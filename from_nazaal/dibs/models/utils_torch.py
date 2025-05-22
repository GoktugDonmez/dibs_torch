import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

# Attempt to import the topological_sort_util from the graph_pytorch_translation
# If they are in the same directory and graph_pytorch_translation.py exists:
# from .graph_pytorch_translation import topological_sort as topological_sort_util
# For this standalone block, I'll redefine a PyTorch version of topological_sort.
# If you have it in another file, you should import it.

# --- PyTorch Topological Sort (Kahn's Algorithm) ---
def topological_sort_util(adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Performs a topological sort on a graph represented by an adjacency matrix.
    adj_matrix[i, j] = 1 implies edge i -> j.
    """
    d = adj_matrix.shape[0]
    # Ensure adj is binary for in-degree calculation
    adj = (adj_matrix > 0).to(dtype=torch.int32, device=adj_matrix.device)

    # In-degree: sum over rows for A[i,j] = i -> j (number of incoming edges to j)
    in_degree = torch.sum(adj, dim=0)
    
    queue = torch.where(in_degree == 0)[0].tolist()
    
    topological_order_list = []
    visited_count = 0

    while queue:
        u = queue.pop(0)
        topological_order_list.append(u)
        visited_count += 1

        # For each neighbor v of u (i.e., u -> v edge)
        # Neighbors are where adj[u, v_idx] == 1
        for v_idx in range(d):
            if adj[u, v_idx] > 0: # Edge u -> v_idx
                in_degree[v_idx] -= 1
                if in_degree[v_idx] == 0:
                    queue.append(v_idx)
    
    is_dag = (visited_count == d)
    return torch.tensor(topological_order_list, dtype=torch.long, device=adj_matrix.device), is_dag
# --- End of Topological Sort ---


def stable_mean(fxs: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False, jitter: float = 1e-30) -> torch.Tensor:
    """
    Computes a numerically stable mean of a tensor `fxs`.
    This version is closer to the JAX original, handling positive and negative parts separately
    by averaging their magnitudes in log-space.

    Args:
        fxs (torch.Tensor): Input tensor.
        dim (Optional[Union[int, Tuple[int, ...]]]): Dimension(s) to reduce over. If None, reduces over all elements.
        keepdim (bool): Whether the output tensor has `dim` retained or not.
        jitter (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: The stable mean.
    """

    def stable_mean_positive_only_global(fs_positive_flat: torch.Tensor, n_positive_total: torch.Tensor) -> torch.Tensor:
        if n_positive_total == 0:
            return torch.tensor(0.0, device=fs_positive_flat.device, dtype=fs_positive_flat.dtype)
        
        # Filter out non-positive values before log, only consider actual positive values for sum
        actual_positives = fs_positive_flat[fs_positive_flat > 0]
        if actual_positives.numel() == 0: # Handles case where all f_xs_psve elements are zero
             return torch.tensor(0.0, device=fs_positive_flat.device, dtype=fs_positive_flat.dtype)

        log_fs = torch.log(actual_positives + jitter) # Add jitter to actual positives before log
        lse_log_fs = torch.logsumexp(log_fs, dim=0) # Sum over all elements of filtered positives
        log_n = torch.log(n_positive_total + jitter)
        return torch.exp(lse_log_fs - log_n)

    if dim is not None:
        # This case is more complex. The JAX code structure implies it might apply the pos/neg split globally
        # even when averaging over a dimension. A true per-slice stable mean would be more involved.
        # For now, this will perform a global-style stable mean then average, or interpret fxs.size as total elements.
        # The original JAX's fxs.size implies it always considers the total number of elements for proportions.
        # This might not be what's intended if `dim` is specified for a slice-wise mean.
        # Let's assume the JAX intent was a global-style proportioning.
        print(f"Warning: stable_mean with specific `dim` might not perfectly match JAX version's global fxs.size proportioning. Using simple torch.mean for dim-wise operations as a fallback or use dim=None for JAX-like global stable_mean.")
        return torch.mean(fxs, dim=dim, keepdim=keepdim)


    # Global mean (dim is None) - this matches the JAX code structure more closely
    is_positive = fxs > 0.0
    is_negative = fxs < 0.0
    n_total = torch.tensor(fxs.numel(), device=fxs.device, dtype=torch.float32)

    if n_total == 0:
        return torch.tensor(0.0, device=fxs.device, dtype=fxs.dtype)

    f_xs_psve_flat = fxs[is_positive] # Flattened positive values
    f_xs_ngve_flat = -fxs[is_negative] # Flattened positive magnitudes of negative values

    n_psve = torch.sum(is_positive).float()
    n_ngve = torch.sum(is_negative).float()

    avg_psve = stable_mean_positive_only_global(f_xs_psve_flat, n_psve)
    avg_ngve = stable_mean_positive_only_global(f_xs_ngve_flat, n_ngve)
    
    # Ensure proportions are calculated correctly even if n_psve or n_ngve is 0
    term_psve = (n_psve / n_total) * avg_psve if n_total > 0 else torch.tensor(0.0, device=fxs.device)
    term_ngve = (n_ngve / n_total) * avg_ngve if n_total > 0 else torch.tensor(0.0, device=fxs.device)
    
    return term_psve - term_ngve


def log_stable_mean_from_logs(log_fxs: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    """
    Computes log(mean(exp(log_fxs))) stably.
    log( (1/N) * sum(exp(log_fxs_i)) ) = logsumexp(log_fxs) - log(N)
    """
    if dim is None:
        # Reduce over all dimensions
        target_dims = list(range(log_fxs.ndim))
        if not target_dims: # scalar input
            return log_fxs 
        lse = torch.logsumexp(log_fxs, dim=target_dims)
        log_n = torch.log(torch.tensor(log_fxs.numel(), dtype=log_fxs.dtype, device=log_fxs.device))
    else:
        lse = torch.logsumexp(log_fxs, dim=dim)
        if isinstance(dim, int):
            num_elements_in_dim = log_fxs.shape[dim]
        else: # tuple of dims
            num_elements_in_dim = 1
            for d_idx in dim:
                num_elements_in_dim *= log_fxs.shape[d_idx]
        log_n = torch.log(torch.tensor(num_elements_in_dim, dtype=log_fxs.dtype, device=log_fxs.device))
    return lse - log_n


def acyclic_constr(g_mat: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
    """
    Computes the acyclicity constraint h(G) = tr((I + (1/d)*G)^d) - d.
    g_mat: [..., d, d] tensor, the graph adjacency matrix (can be soft).
    d: Number of nodes. If None, inferred from g_mat.shape[-1].
    """
    if d is None:
        d = g_mat.shape[-1]

    if g_mat.shape[-1] != d or g_mat.shape[-2] != d:
        raise ValueError(f"g_mat shape {g_mat.shape} inconsistent with d={d}")

    alpha = 1.0 / float(d)
    eye = torch.eye(d, device=g_mat.device, dtype=g_mat.dtype)
    
    # Add batch dimension if g_mat is a single matrix for consistent processing
    is_batched = g_mat.ndim == 3
    if not is_batched:
        g_mat = g_mat.unsqueeze(0) # [1, D, D]

    m = eye.unsqueeze(0) + alpha * g_mat # [B, D, D]

    try:
        # matrix_power expects integer exponent
        m_mult = torch.linalg.matrix_power(m, int(d)) # [B, D, D]
        # Compute trace for each matrix in the batch
        h_batch = torch.vmap(torch.trace)(m_mult) - d # torch.vmap requires PyTorch 1.8+
        # Fallback for trace if vmap not available or for older PyTorch:
        # h_batch = torch.stack([torch.trace(m_b_mult) for m_b_mult in m_mult]) - d
    except Exception as e:
        print(f"Warning: acyclic_constr matrix_power failed (d={d}). Error: {e}. Returning large penalty.")
        return torch.full((g_mat.shape[0],), float('inf'), device=g_mat.device, dtype=g_mat.dtype)

    return h_batch if is_batched else h_batch.squeeze(0)


def sample_y(true_gmat: torch.Tensor, edges: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Samples expert beliefs 'y' for given edges based on a true graph and error rate rho.
    Assumes Bernoulli expert. true_gmat[i,j]=1 means i->j.
    edges: [num_edges, 2] tensor of (row_idx, col_idx) for edges.
    Returns: y_samples [num_edges] tensor of 0s and 1s.
    """
    gmat_values_at_edges = true_gmat[edges[:, 0], edges[:, 1]] # P(edge exists) or 0/1

    # bernoulli_p: probability that expert says "edge exists" (y=1)
    # If true edge exists (gmat_value_at_edge = 1), expert says 1 with prob (1-rho).
    # If true edge does not exist (gmat_value_at_edge = 0), expert says 1 with prob rho.
    # So, P(expert_says_1 | gmat_value) = gmat_value * (1-rho) + (1-gmat_value) * rho
    prob_expert_says_1 = gmat_values_at_edges * (1.0 - rho) + (1.0 - gmat_values_at_edges) * rho
    
    y = torch.bernoulli(prob_expert_says_1)
    return y.to(torch.int32)


# --- Neural Network Layers ---
def layer_with_activation(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, activation_fn: Callable = F.relu) -> torch.Tensor:
    """Forward pass for a hidden layer. Assumes x is [..., N, InFeatures]."""
    # JAX: weights @ x + bias where x is [D,1] (features are rows)
    # PyTorch nn.Linear: x @ W^T + b where x is [N, InFeatures]
    # If weights are [Out, In], then x @ weights.T or weights @ x (if x is column)
    # Assuming weights: [out_features, in_features], bias: [out_features]
    # x: [Batch, InFeatures] or [InFeatures]
    # Output: [Batch, OutFeatures] or [OutFeatures]
    
    # Ensure x is at least 2D for matmul with weights.T
    original_ndim = x.ndim
    if original_ndim == 1:
        x = x.unsqueeze(0) # [1, InFeatures]

    # Linear transformation: y = xW^T + b
    output = F.linear(x, weights, bias) # PyTorch's F.linear handles xW^T + b
    
    if activation_fn is not None:
      output = activation_fn(output)

    if original_ndim == 1:
        output = output.squeeze(0)
    return output

def linear_layer_nn(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Forward pass for an output layer (linear). PyTorch F.linear convention."""
    return layer_with_activation(x, weights, bias, activation_fn=None)

def forward_nn(x_input: torch.Tensor, weights_list: List[torch.Tensor], bias_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Forward pass through the network.
    x_input: [Batch, InFeatures] or [InFeatures]
    The final layer is linear. Activation is ReLU for hidden.
    """
    activation = x_input
    num_layers = len(weights_list)
    for i in range(num_layers - 1):
        activation = layer_with_activation(activation, weights_list[i], bias_list[i], activation_fn=F.relu)
    
    # Apply final layer without activation (linear output)
    activation = linear_layer_nn(activation, weights_list[-1], bias_list[-1])
    return activation

def forward_pytree_nn(x_input: torch.Tensor, params_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Forward pass that takes a pytree-like list of layer parameters."""
    weights = [p['weights'] for p in params_list]
    biases = [p['bias'] for p in params_list]
    return forward_nn(x_input, weights, biases)

# ResNet specific layers
def resnet_layer_nn(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Residual layer: relu(Wx+b) + x. Assumes input x and output of linear part match shape for residual."""
    # x: [Batch, Features]
    # Wx+b also [Batch, Features]
    h = layer_with_activation(x, weights, bias, activation_fn=F.relu)
    # Ensure shapes match for addition, might need projection if they don't.
    # The JAX code assumes they match: h + x
    if h.shape != x.shape:
        # This would require a projection layer for x, or ensure W is identity-like for shape matching.
        # For simplicity, assume shapes match or a projection is part of 'weights' for h.
        raise ValueError(f"Shape mismatch in ResNet layer: h.shape={h.shape}, x.shape={x.shape}")
    return h + x

def forward_resnet_nn(x_input: torch.Tensor, weights_list: List[torch.Tensor], bias_list: List[torch.Tensor]) -> torch.Tensor:
    """ResNet forward pass."""
    activation = x_input
    num_layers = len(weights_list)

    for i in range(num_layers - 1): # Hidden ResNet layers
        activation = resnet_layer_nn(activation, weights_list[i], bias_list[i])
    
    # Final layer (linear, no residual connection in this JAX version)
    activation = linear_layer_nn(activation, weights_list[-1], bias_list[-1])
    return activation

def forward_resnet_pytree_nn(x_input: torch.Tensor, params_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """ResNet forward pass using PyTree-like parameters."""
    weights = [p['weights'] for p in params_list]
    biases = [p['bias'] for p in params_list]
    return forward_resnet_nn(x_input, weights, biases)

# --- SCM Sampling ---
def sample_x(
    hard_gmat: torch.Tensor, # A[i,j]=1 means i->j
    theta: Any, 
    n_samples: int, 
    hparams: Dict[str, Any], 
    intervs: Optional[torch.Tensor] = None, # [target_idx, value]
    iscm: bool = False,
    device: str = 'cpu'
) -> torch.Tensor:
    """Samples data from an SCM, handling linear or NN mechanisms."""
    d = hard_gmat.shape[0]
    noise_std_per_node = hparams["noise_std"].to(device)       # [D]
    parent_means_per_node = hparams["parent_means"].to(device) # [D], typically base means for root nodes

    topological_order, is_dag = topological_sort_util(hard_gmat) # Needs A[i,j] for i->j
    if not is_dag:
        # print("Warning: sample_x called with a cyclic graph. Returning zeros.")
        return torch.zeros((n_samples, d), device=device) # Or raise error

    pop_means = torch.zeros(d, device=device)
    pop_stds = torch.ones(d, device=device)
    if iscm:
        assert "pop_means" in hparams and "pop_stds" in hparams, "pop_means/stds required for iSCM"
        pop_means = hparams["pop_means"].to(device)
        pop_stds = hparams["pop_stds"].to(device)
        pop_stds = torch.where(pop_stds == 0, torch.ones_like(pop_stds), pop_stds) # Avoid div by zero

    interv_var_idx, interv_val = -1, 0.0
    has_intervention = False
    if intervs is not None:
        interv_var_idx = int(intervs[0].item())
        interv_val = intervs[1].item()
        has_intervention = True

    x_samples = torch.zeros((n_samples, d), device=device)
    
    # Determine if theta represents a linear model or NN parameters
    is_linear_model = torch.is_tensor(theta) and theta.ndim == 2 and theta.shape == (d,d)
    is_nn_model = isinstance(theta, list) # Assuming list of dicts for NN params per node

    for node_idx_in_order in range(d):
        current_var_idx = topological_order[node_idx_in_order].item()
        
        exog_noise = noise_std_per_node[current_var_idx] * torch.randn(n_samples, device=device)
        
        # Parents of current_var_idx are nodes j where hard_gmat[j, current_var_idx] == 1
        parent_mask = hard_gmat[:, current_var_idx] == 1.0
        parent_indices = torch.where(parent_mask)[0]

        val_from_scm_mech: torch.Tensor
        if not torch.any(parent_mask): # Root node
            val_from_scm_mech = parent_means_per_node[current_var_idx] + exog_noise
        else: # Has parents
            parent_data = x_samples[:, parent_indices] # [N_samples, Num_parents]
            
            if is_linear_model:
                # Linear model: X_i = sum_{j in Parents(i)} X_j * theta[j,i] + noise_i
                # theta[j,i] is the coefficient for edge j->i
                coeffs_for_parents = theta[parent_indices, current_var_idx] # [Num_parents]
                mean_from_parents = torch.matmul(parent_data, coeffs_for_parents) # [N_samples]
                val_from_scm_mech = mean_from_parents + exog_noise
            elif is_nn_model:
                # NN model: X_i = NN_i(Parents_values; theta_i) + noise_i
                # theta[current_var_idx] should contain NN parameters for node i
                # The JAX code's NN input: x_sample_values * hard_gmat[:, current_var_idx]
                # This passes all D variables, with non-parents zeroed out.
                
                # We need to apply the NN for node `current_var_idx` to its parent values.
                # The `forward_resnet_pytree_nn` takes `x_input` and `params_list`.
                # JAX version's `scm_forward` was:
                # `jax.vmap(lambda x_one_sample: forward_resnet_pytree(x_one_sample * hard_gmat[:, i], theta[i], d))(xs)`
                # This means the NN for node `i` takes a D-dimensional vector as input.
                
                nn_params_for_node = theta[current_var_idx]
                
                # Create input for NN: [N_samples, D], where non-parents are zeroed out
                # This is inefficient if NNs are small and only take few parents.
                # A more efficient way would be for NN_i to only take values of Parents(X_i).
                # But to match JAX code's `x * hard_gmat[:,i]` input style:
                input_to_nn_batch = torch.zeros_like(x_samples, device=device) # [N_samples, D]
                input_to_nn_batch[:, parent_indices] = parent_data

                # The forward_resnet_pytree_nn expects [Batch, InFeatures] or [InFeatures]
                # If the NN for node `i` is defined to take D features (masked):
                mean_from_parents = forward_resnet_pytree_nn(input_to_nn_batch, nn_params_for_node) # [N_samples]
                val_from_scm_mech = mean_from_parents + exog_noise
            else:
                raise ValueError(f"Unsupported theta type for SCM sampling at node {current_var_idx}")

        # Apply iSCM normalization/scaling
        normalized_val = (val_from_scm_mech - pop_means[current_var_idx]) / pop_stds[current_var_idx]
        x_samples[:, current_var_idx] = normalized_val

        if has_intervention and current_var_idx == interv_var_idx:
            x_samples[:, current_var_idx] = interv_val
            
    return x_samples

# sample_x_old is a simplified version of sample_x, primarily for linear models and without iSCM.
# Its translation would follow sample_x closely, removing iSCM logic and simplifying the scm_forward part.

# --- Intervention utilities (Conceptual translations, as these are higher-level) ---
# These functions often use vmap over particles or intervention values.
# In PyTorch, this would typically be Python loops or ensuring batch-compatibility of inner functions.

def f_distr_per_intervention_i(
    i_node_to_intervene: int, 
    hard_gmat: torch.Tensor, 
    theta: Any, 
    hparams: Dict[str, Any],
    target_node_idx: int = -1, # Index of the target variable f(x_target)
    device: str = 'cpu'
) -> torch.Tensor:
    """
    For a given model (hard_gmat, theta), and intervention on node `i_node_to_intervene`,
    computes samples of f(x_target) over a grid of intervention values.
    hparams must contain: 'n_scm_samples', 'f' (the function f(x_target_value)),
                         'xmin', 'xmax', 'interv_discretization'.
    """
    interv_values = torch.linspace(hparams['xmin'], hparams['xmax'], hparams['interv_discretization'], device=device)
    n_scm_samples = hparams['n_scm_samples']
    
    f_on_target_node_fn = hparams['f'] # Example: lambda x_target_val: x_target_val (if f is identity)

    results_for_ival = []
    for ival in interv_values:
        current_interv = torch.tensor([i_node_to_intervene, ival.item()], device=device)
        # Sample X under do(node_i = ival)
        x_intervened_samples = sample_x(hard_gmat, theta, n_scm_samples, hparams, intervs=current_interv, device=device)
        # Get values of the target node
        target_node_values = x_intervened_samples[:, target_node_idx] # [n_scm_samples]
        
        # Apply function f. If f_on_target_node_fn is already vectorized for tensor input:
        f_outputs = f_on_target_node_fn(target_node_values) # [n_scm_samples]
        results_for_ival.append(f_outputs) # List of [n_scm_samples] tensors
        
    return torch.stack(results_for_ival) # [interv_discretization, n_scm_samples]

def best_intervention(
    hard_gmat: torch.Tensor, 
    theta: Any, 
    hparams: Dict[str, Any],
    target_node_idx: int = -1, # Which node is the "target" for function f
    num_nodes_to_intervene_on: Optional[int] = None, # Usually d or d-1
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Given a single model (hard_gmat, theta), finds the best (node, value) intervention
    that maximizes the mean of f(x_target).
    """
    d = hard_gmat.shape[-1]
    if num_nodes_to_intervene_on is None:
        num_nodes_to_intervene_on = d # Intervene on all nodes (except possibly target if d-1)
        # JAX code uses d-1, perhaps excluding the target node itself from intervention if it's fixed.

    interv_domain = torch.linspace(hparams["xmin"], hparams["xmax"], hparams["interv_discretization"], device=device)
    
    all_avg_f_outputs = [] # Store avg_f for each (intervened_node, interv_value)

    # Iterate over nodes to intervene on (0 to num_nodes_to_intervene_on - 1)
    for i_node in range(num_nodes_to_intervene_on):
        # Get f_distr for intervening on node i_node: shape [interv_discretization, n_scm_samples]
        f_samples_for_node_i = f_distr_per_intervention_i(i_node, hard_gmat, theta, hparams, target_node_idx, device)
        # Mean of f samples at each intervention value for this node: shape [interv_discretization]
        avg_f_for_node_i_at_each_ival = torch.mean(f_samples_for_node_i, dim=1)
        all_avg_f_outputs.append(avg_f_for_node_i_at_each_ival)
        
    # all_avg_f_outputs is list of tensors, stack to [num_intervened_nodes, interv_discretization]
    avg_fs_matrix = torch.stack(all_avg_f_outputs)

    # Find max and its indices
    max_val = torch.max(avg_fs_matrix)
    indices = torch.where(avg_fs_matrix == max_val) # Returns tuple of tensors
    
    # Take the first occurrence if multiple maxima
    best_node_to_intervene = indices[0][0].item()
    best_val_idx = indices[1][0].item()
    
    best_interv_value = interv_domain[best_val_idx]
    
    return torch.tensor([best_node_to_intervene, best_interv_value.item()], device=device)


def process_dag_util(gmat: torch.Tensor) -> torch.Tensor:
    """
    Returns an empty graph if the input graph `gmat` has cycles.
    Otherwise, returns `gmat`.
    This version assumes gmat is a single [D,D] matrix.
    The JAX `process_dag` also had a `target_has_no_parents` check which might be specific.
    """
    d = gmat.shape[-1]
    h_G = acyclic_constr(gmat.unsqueeze(0) if gmat.ndim==2 else gmat, d).squeeze() # acyclic_constr expects batch
    is_cyclic = h_G > 1e-6  # Add a small tolerance for float precision

    # Original JAX code had:
    # target_has_no_parents = (jnp.sum(gmat[:, -1]) == 0).astype(int) # if 1, thats bad.
    # zero_out = jax.lax.bitwise_or(is_cyclic, target_has_no_parents)
    # if jnp.allclose(zero_out, 1.0): return 0.0 * gmat
    # This `target_has_no_parents` seems very specific to a causal objective where the last node is the target.
    # For general DAG processing, only cyclicity is usually checked.

    if is_cyclic.item(): # .item() if h_G is scalar after squeeze
        return torch.zeros_like(gmat)
    else:
        return gmat