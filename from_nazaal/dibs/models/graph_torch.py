import random
import torch
import igraph as ig # type: ignore
from typing import List, Tuple

# Function Calling Structure for graph.py:
# - scalefree_dag_gmat:
#   - Uses `torch.randperm` (for permutation)
#   - Calls `graph_to_mat` (to convert igraph to tensor)
# - graph_to_mat: Converts igraph graph to PyTorch tensor.
# - mat_to_graph: Converts PyTorch tensor to igraph graph.
# - topological_sort: (Kahn's algorithm)
#   - Iteratively finds nodes with in-degree 0.
# - adjacency_to_igraph_directed: Converts PyTorch tensor to a directed igraph graph.

def scalefree_dag_gmat(
    d: int,
    n_edges_per_node: int = 2,
    seed: int = None
) -> torch.Tensor:
    """
    Generates an adjacency matrix for a scale-free directed acyclic graph (DAG).

    Args:
        d: Number of nodes.
        n_edges_per_node: Number of edges to add for each new node in Barabasi-Albert.
        seed: Optional random seed for reproducibility.

    Returns:
        torch.Tensor: Adjacency matrix of shape [d, d].
    """
    if seed is not None:
        random.seed(seed)
        # For torch.randperm, we should also seed PyTorch if we want full control
        # from a single seed, or pass a generator.
        # For simplicity with the original JAX code's use of `random.seed(int(jnp.sum(key)))`,
        # we'll primarily rely on Python's random for igraph's generation if it uses it internally,
        # and torch's own seeding for torch operations.
        # If igraph's Barabasi uses Python's global random, this will affect it.
        torch.manual_seed(seed)


    # Generate a random permutation for nodes to break symmetries and ensure DAGness
    # by adding edges according to the permuted order.
    # The original JAX code permutes vertices *after* Barabasi graph generation,
    # which makes it a DAG if edges are added respecting the original order before permuting.
    # igraph's Barabasi by default can create cycles if not careful.
    # A common way to ensure DAG from Barabasi-Albert is to add edges from new nodes to existing ones.
    # ig.Graph.Barabasi(directed=True) already does this, making it a DAG if power=1.
    # The permutation then shuffles node labels.

    # Create a permutation of node indices
    perm = torch.randperm(d).tolist()

    # Generate a Barabasi-Albert graph
    # `directed=True` ensures edges go from older to newer nodes (or vice-versa depending on convention),
    # making it a DAG.
    g = ig.Graph.Barabasi(n=d, m=n_edges_per_node, directed=True)

    # Permute the vertices according to the generated permutation
    # This re-labels the nodes but keeps the DAG structure relative to the new labels.
    g = g.permute_vertices(perm)

    dag_mat = graph_to_mat(g)
    return dag_mat

def graph_to_mat(g: ig.Graph) -> torch.Tensor:
    """
    Converts an igraph.Graph object to a PyTorch tensor representing its adjacency matrix.
    """
    # g.get_adjacency() returns a Matrix object, .data gives list of lists
    adj_list_of_lists = g.get_adjacency().data
    return torch.tensor(adj_list_of_lists, dtype=torch.float32)

def mat_to_graph(mat: torch.Tensor) -> ig.Graph:
    """
    Converts a PyTorch tensor representing an adjacency matrix to an igraph.Graph object.
    Assumes a weighted adjacency matrix if values are not just 0 or 1.
    """
    assert len(mat.shape) == 2, "igraph can only handle d x d matrix inputs"
    assert mat.shape[0] == mat.shape[1], "igraph can only handle d x d matrix inputs"
    # igraph.Graph.Weighted_Adjacency expects a list of lists
    return ig.Graph.Weighted_Adjacency(mat.tolist())

def topological_sort(adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Performs a topological sort (Kahn's algorithm) on a graph represented by an adjacency matrix.
    The input adj_matrix should have A[i, j] = 1 if there's an edge from i to j.

    Args:
        adj_matrix (torch.Tensor): The DxD adjacency matrix (binary or weighted, values > 0 mean edge).
                                   A[i,j] = 1 means i -> j.

    Returns:
        Tuple[torch.Tensor, bool]:
            - Tensor containing the topological order of nodes.
            - Boolean indicating if the graph is a DAG (True if DAG, False if cycle detected).
    """
    d = adj_matrix.shape[0]
    adj = (adj_matrix > 0).to(torch.int32) # Ensure binary for in-degree calculation

    # In-degree: sum over columns for A_ij = i -> j (i.e. sum over sources for each target)
    # Or sum over rows for A_ij = j -> i (i.e. sum over targets for each source)
    # Standard Kahn's: in_degree[j] = sum_i A[i,j] (number of incoming edges to j)
    in_degree = torch.sum(adj, dim=0) # in_degree[j] = sum_i adj[i,j]

    # Queue for nodes with in-degree 0
    queue = torch.where(in_degree == 0)[0].tolist()

    topological_order = []
    visited_count = 0

    while queue:
        u = queue.pop(0)
        topological_order.append(u)
        visited_count += 1

        # For each neighbor v of u (i.e., u -> v edge)
        # Neighbors are where adj[u, v] == 1
        for v_idx in range(d):
            if adj[u, v_idx] > 0:
                in_degree[v_idx] -= 1
                if in_degree[v_idx] == 0:
                    queue.append(v_idx)

    if visited_count != d:
        # Graph has a cycle
        return torch.tensor(topological_order, dtype=torch.long), False
    else:
        return torch.tensor(topological_order, dtype=torch.long), True


def adjacency_to_igraph_directed(adj_matrix: torch.Tensor) -> ig.Graph:
    """
    Converts a PyTorch tensor (adjacency matrix) to a directed igraph.Graph object.
    Weights are preserved.
    A[i,j] != 0 means edge i -> j.
    """
    g = ig.Graph(directed=True)
    num_vertices = adj_matrix.shape[0]
    g.add_vertices(num_vertices)

    edges = []
    weights = []
    for i in range(num_vertices):
        for j in range(num_vertices):  # Consider all entries for directed edges
            if adj_matrix[i, j] != 0:
                edges.append((i, j))
                # Round weights for cleaner representation if desired, or use raw
                weights.append(round(adj_matrix[i, j].item(), 3))

    g.add_edges(edges)
    if weights: # Only add weights if there are edges with non-zero weights
        g.es["weight"] = weights

    return g