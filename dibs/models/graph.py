import igraph as ig
import random as pyrandom_std # Renamed to avoid conflict
import torch
import math # For log

# --- Assumed Helper Functions (PyTorch stubs/versions) ---

def mat_to_graph_torch(adj_matrix: torch.Tensor) -> ig.Graph:
    """Converts a PyTorch adjacency matrix to an igraph.Graph."""
    if adj_matrix.is_cuda:
        adj_matrix = adj_matrix.cpu()
    # igraph.Graph.Adjacency expects list of lists or NumPy array.
    # It's safer to convert to NumPy then to list of lists.
    np_matrix = adj_matrix.numpy()
    return ig.Graph.Adjacency(np_matrix.tolist(), mode=ig.ADJ_DIRECTED)

def graph_to_mat_torch(graph: ig.Graph, n_vars: int) -> torch.Tensor:
    """Converts an igraph.Graph to a PyTorch adjacency matrix."""
    if n_vars == 0:
        return torch.empty((0,0), dtype=torch.int32)
    adj_matrix_list = graph.get_adjacency(type=ig.GET_ADJACENCY_BOOL).data # Get boolean matrix
    return torch.tensor(adj_matrix_list, dtype=torch.int32) # Convert to int

def mat_is_dag_torch(adj_matrix: torch.Tensor) -> bool:
    """Checks if a PyTorch adjacency matrix represents a DAG."""
    if adj_matrix.numel() == 0: # Empty graph is a DAG
        return True
    # Convert to igraph and use its is_dag() method for simplicity
    temp_graph = mat_to_graph_torch(adj_matrix)
    return temp_graph.is_dag()

def zero_diagonal_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Sets the diagonal of a PyTorch tensor to zero and returns a new tensor."""
    if matrix.numel() == 0:
        return matrix.clone()
    # Ensure matrix is square for fill_diagonal_
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to zero its diagonal.")
    
    cloned_matrix = matrix.clone()
    cloned_matrix.fill_diagonal_(0)
    return cloned_matrix

# --- PyTorch Graph Distributions ---

class ErdosReniDAGDistributionTorch:
    def __init__(self, n_vars: int, n_edges_per_node: int = 2, device: str = 'cpu'):
        self.n_vars = n_vars
        self.device = torch.device(device)
        self.dtype = torch.float32 # For log_prob results

        if self.n_vars <= 1:
            self.p_edge = 0.0
        else:
            max_possible_edges = (self.n_vars * (self.n_vars - 1)) / 2.0
            # Ensure max_possible_edges is not zero before division
            if max_possible_edges == 0: # Should only happen if n_vars <=1, caught above
                 self.p_edge = 0.0
            else:
                 self.n_edges_expected = n_edges_per_node * n_vars
                 self.p_edge = self.n_edges_expected / max_possible_edges
        self.p_edge = max(0.0, min(1.0, self.p_edge)) # Clamp to [0,1]

        # Precompute log probabilities carefully to handle p_edge = 0 or 1
        if self.p_edge == 0.0:
            self._log_p_edge = -torch.inf
            self._log_1_minus_p_edge = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        elif self.p_edge == 1.0:
            self._log_p_edge = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            self._log_1_minus_p_edge = -torch.inf
        else:
            self._log_p_edge = torch.log(torch.tensor(self.p_edge, device=self.device, dtype=self.dtype))
            self._log_1_minus_p_edge = torch.log(torch.tensor(1.0 - self.p_edge, device=self.device, dtype=self.dtype))


    def sample_G(self, generator: torch.Generator = None, return_mat: bool = False) -> (ig.Graph | torch.Tensor):
        if self.n_vars == 0:
            mat_shape = (0,0)
        else:
            mat_shape = (self.n_vars, self.n_vars)

        prob_tensor = torch.full(mat_shape, self.p_edge, device=self.device, dtype=torch.float32)
        
        rand_fn = torch.bernoulli if generator is None else lambda p: torch.bernoulli(p, generator=generator)
        mat = rand_fn(prob_tensor).to(torch.int32)

        dag_tril = torch.tril(mat, diagonal=-1)

        if self.n_vars > 0:
            perm_indices = torch.randperm(self.n_vars, generator=generator, device=self.device)
            # Efficient permutation of rows and columns
            dag_perm = dag_tril[perm_indices][:, perm_indices]
        else:
            dag_perm = dag_tril # Remains empty if n_vars = 0

        if return_mat:
            return dag_perm
        else:
            return mat_to_graph_torch(dag_perm)


    def _safe_mul(self, count, log_val):
        if count == 0 and torch.isneginf(log_val):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return count * log_val

    def unnormalized_log_prob_single(self, *, g: ig.Graph, j: int) -> torch.Tensor:
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        
        # Potential parents for node j are n_vars - 1 (excluding self)
        # This assumes a dense graph structure possibilities initially.
        # The ER model considers each possible edge. For a node j, it could have up to n_vars-1 parents.
        # The original JAX code uses `self.n_vars - n_parents - 1` for non-parents.
        # This counts nodes that are not j and not parents of j.
        n_non_parents_for_j = (self.n_vars - 1) - n_parents

        term1 = self._safe_mul(n_parents, self._log_p_edge)
        term2 = self._safe_mul(n_non_parents_for_j, self._log_1_minus_p_edge)
        return term1 + term2

    def unnormalized_log_prob(self, *, g: ig.Graph) -> torch.Tensor:
        if self.n_vars <= 1:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
        N_max_edges = self.n_vars * (self.n_vars - 1) / 2.0
        E_actual_edges = float(len(g.es))

        term1 = self._safe_mul(E_actual_edges, self._log_p_edge)
        term2 = self._safe_mul(N_max_edges - E_actual_edges, self._log_1_minus_p_edge)
        return term1 + term2

    def unnormalized_log_prob_soft(self, *, soft_g: torch.Tensor) -> torch.Tensor:
        soft_g_on_dev = soft_g.to(device=self.device, dtype=self.dtype)
        
        if self.n_vars <= 1:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Sum of probabilities in the upper/lower triangle for DAGs
        # Ensure no processing if n_vars is too small for triu
        E_eff = torch.triu(soft_g_on_dev, diagonal=1).sum() if self.n_vars > 1 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Total possible unique directed edges in a simple graph (not necessarily DAG)
        N_max_edges = self.n_vars * (self.n_vars - 1) / 2.0

        # Original JAX used E = soft_g.sum(), (N-E). This implies soft_g is already triangular or represents something else.
        # The cross-entropy formulation Sum [ G_ij log p + (1-G_ij) log(1-p) ] over potential edges is more standard.
        # Here, we stick to the JAX code's direct formula using sum of probabilities as E_eff.
        term1 = self._safe_mul(E_eff, self._log_p_edge)
        term2 = self._safe_mul(N_max_edges - E_eff, self._log_1_minus_p_edge)
        
        return term1 + term2


class ScaleFreeDAGDistributionTorch:
    def __init__(self, n_vars: int, n_edges_per_node: int = 2, device: str = 'cpu'):
        self.n_vars = n_vars
        self.n_edges_per_node = n_edges_per_node
        self.device = torch.device(device)
        self.dtype = torch.float32 # For log_prob results

    def sample_G(self, generator: torch.Generator = None, return_mat: bool = False) -> (ig.Graph | torch.Tensor):
        if self.n_vars == 0:
            g = ig.Graph(directed=True)
            perm_indices_list = []
        else:
            if generator:
                # Use generator to create a seed for pyrandom_std
                # Generate a random 32-bit integer.
                seed_for_pyrandom = torch.empty((), dtype=torch.int64).random_(generator=generator).item() % (2**32)
            else:
                seed_for_pyrandom = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()
            pyrandom_std.seed(seed_for_pyrandom)
            
            perm_indices_list = torch.randperm(self.n_vars, generator=generator).tolist() # igraph permutation needs list

            # ig.Graph.Barabasi uses Python's global random.
            # It requires n >= m for m > 0. If m=0, it's fine.
            actual_m = self.n_edges_per_node
            if self.n_vars < actual_m and self.n_vars > 0 : # if n_vars is 0, m should be 0
                actual_m = self.n_vars
            if self.n_vars == 0: # if n_vars is 0, m must be 0
                actual_m = 0

            g = ig.Graph.Barabasi(n=self.n_vars, m=actual_m, directed=True)
            if self.n_vars > 0 : # permute_vertices fails on empty list for 0 nodes
                g = g.permute_vertices(perm_indices_list)

        if return_mat:
            return graph_to_mat_torch(g, self.n_vars).to(self.device)
        else:
            return g

    def unnormalized_log_prob_single(self, *, g: ig.Graph, j: int) -> torch.Tensor:
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return -3.0 * torch.log(torch.tensor(1.0 + n_parents, device=self.device, dtype=self.dtype))

    def unnormalized_log_prob(self, *, g: ig.Graph) -> torch.Tensor:
        if self.n_vars == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        scores = [self.unnormalized_log_prob_single(g=g, j=node_idx) for node_idx in range(self.n_vars)]
        return torch.stack(scores).sum()

    def unnormalized_log_prob_soft(self, *, soft_g: torch.Tensor) -> torch.Tensor:
        soft_g_on_dev = soft_g.to(device=self.device, dtype=self.dtype)
        if self.n_vars == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        soft_indegree = soft_g_on_dev.sum(dim=0) # Sum over rows (dim 0) to get in-degree for each column node
        return torch.sum(-3.0 * torch.log(1.0 + soft_indegree))


class UniformDAGDistributionRejectionTorch:
    def __init__(self, n_vars: int, device: str = 'cpu'):
        self.n_vars = n_vars
        self.device = torch.device(device)
        self.dtype = torch.float32 # For log_prob results

    def sample_G(self, generator: torch.Generator = None, return_mat: bool = False) -> (ig.Graph | torch.Tensor):
        if self.n_vars == 0:
            if return_mat: return torch.empty((0,0), dtype=torch.int32, device=self.device)
            else: return ig.Graph(directed=True)

        while True:
            prob_tensor = torch.full((self.n_vars, self.n_vars), 0.5, device=self.device, dtype=torch.float32)
            rand_fn = torch.bernoulli if generator is None else lambda p: torch.bernoulli(p, generator=generator)
            mat = rand_fn(prob_tensor).to(torch.int32)
            
            mat = zero_diagonal_torch(mat)

            if mat_is_dag_torch(mat): # mat_is_dag_torch will handle CPU conversion if needed
                if return_mat:
                    return mat
                else:
                    return mat_to_graph_torch(mat) # mat_to_graph_torch also handles CPU

    def unnormalized_log_prob_single(self, *, g: ig.Graph, j: int) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def unnormalized_log_prob(self, *, g: ig.Graph) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def unnormalized_log_prob_soft(self, *, soft_g: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)