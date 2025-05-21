import torch
from torch.distributions import Normal
from torch.special import lgamma
import math

# Helper for slogdet on submatrices (PyTorch equivalent for _slogdet_jax)
def _slogdet_pytorch(matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes logabsdet of a submatrix selected by a boolean mask.
    The submatrix is formed by selecting rows and columns where mask is True.

    Args:
        matrix (torch.Tensor): The input matrix [d, d].
        mask (torch.Tensor): A 1D boolean tensor of shape [d] indicating rows/columns to select.

    Returns:
        torch.Tensor: The log absolute determinant of the submatrix. Returns 0.0 for an empty submatrix.
    """
    if not mask.dtype == torch.bool:
        mask = mask.bool() # Ensure mask is boolean for indexing
        
    num_selected = mask.sum()
    if num_selected == 0: # No elements selected, determinant of 0x0 matrix is 1.
        return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)
    
    sub_matrix = matrix[mask][:, mask]
    
    # If sub_matrix becomes 0x0 despite num_selected > 0 (should not happen with proper mask)
    # or if it's otherwise problematic for slogdet.
    # This check is more robust for the 0x0 case specifically.
    if sub_matrix.numel() == 0 :
         return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)

    _sign, logabsdet = torch.linalg.slogdet(sub_matrix)
    return logabsdet


class BGeTorch:
    """
    PyTorch implementation of the BGe score.
    Linear Gaussian BN model with Normal-Wishart conjugate prior.
    """
    def __init__(self, *,
                 n_vars: int,
                 mean_obs: torch.Tensor = None,
                 alpha_mu: float = None,
                 alpha_lambd: float = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.n_vars = n_vars
        self.device = torch.device(device)
        self.dtype = dtype

        if mean_obs is None:
            self.mean_obs = torch.zeros(self.n_vars, device=self.device, dtype=self.dtype)
        else:
            # Ensure input tensor is on the correct device and type
            self.mean_obs = mean_obs.clone().detach().to(device=self.device, dtype=self.dtype)

        self.alpha_mu = alpha_mu if alpha_mu is not None else 1.0
        
        # Default alpha_lambd from JAX code: self.n_vars + 2
        # JAX code asserts self.alpha_lambd > self.n_vars + 1
        default_alpha_lambd = float(self.n_vars + 2)
        self.alpha_lambd = alpha_lambd if alpha_lambd is not None else default_alpha_lambd
        
        if not self.alpha_lambd > self.n_vars + 1:
            raise ValueError(
                f"alpha_lambd ({self.alpha_lambd}) must be greater than n_vars + 1 ({self.n_vars + 1})"
            )

        self.no_interv_targets = torch.zeros(self.n_vars, device=self.device, dtype=torch.bool)

    def get_theta_shape(self, *, n_vars):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussianTorch` model instead.")

    def sample_parameters(self, *, generator, n_vars, n_particles=0, batch_size=0):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussianTorch` model instead.")

    def sample_obs(self, *, generator, n_samples, g, theta, toporder=None, interv=None):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussianTorch` model instead.")

    def _log_marginal_likelihood_single(self, j_node_idx: int, n_parents_j: float, g: torch.Tensor, 
                                        x: torch.Tensor, interv_targets: torch.Tensor) -> torch.Tensor:
        """
        Computes node-specific score of BGe marginal likelihood for node j_node_idx.
        """
        d = x.shape[-1] # Number of variables
        
        # Ensure hyperparameters are float for calculations
        _alpha_mu = float(self.alpha_mu)
        _alpha_lambd = float(self.alpha_lambd)

        # small_t: scalar part of the prior precision matrix T0
        small_t = (_alpha_mu * (_alpha_lambd - d - 1.0)) / (_alpha_mu + 1.0)
        if small_t <= 0:
            raise ValueError(f"small_t ({small_t}) must be positive. Check alpha_lambd and alpha_mu values.")

        T0 = small_t * torch.eye(d, device=self.device, dtype=self.dtype)

        # Mask rows of `x` where node j is intervened upon
        x = x * (1.0 - interv_targets[..., j_node_idx].unsqueeze(-1))
        N = (1.0 - interv_targets[..., j_node_idx]).sum()
        
        is_N_zero = torch.isclose(N, torch.tensor(0.0, device=self.device, dtype=self.dtype))

        # Compute mean of non-intervened data
        x_bar = torch.where(is_N_zero.unsqueeze(0),
                           torch.zeros((1, d), device=self.device, dtype=self.dtype),
                           x.sum(dim=0, keepdim=True) / N)
        
        # Center data and compute scatter matrix
        x_center = (x - x_bar) * (1.0 - interv_targets[..., j_node_idx].unsqueeze(-1))
        s_N = x_center.T @ x_center  # Scatter matrix [d, d]

        # Compute R matrix (posterior precision related matrix)
        R = T0 + s_N + ((N * _alpha_mu) / (N + _alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))

        # Get parent masks
        parents = g[:, j_node_idx]
        parents_and_j = (g + torch.eye(d, device=self.device, dtype=torch.bool))[:, j_node_idx]

        # Log gamma term components
        log_gamma_term = (
            0.5 * (torch.log(torch.tensor(_alpha_mu, device=self.device, dtype=self.dtype)) 
                   - torch.log(N + _alpha_mu))
            + lgamma(0.5 * (N + _alpha_lambd - d + n_parents_j + 1.0))
            - lgamma(0.5 * (_alpha_lambd - d + n_parents_j + 1.0))
            - 0.5 * N * math.log(math.pi)
            + 0.5 * (_alpha_lambd - d + 2.0 * n_parents_j + 1.0) * torch.log(torch.tensor(small_t, device=self.device, dtype=self.dtype))
        )

        # Log determinant terms
        log_term_r = (
            0.5 * (N + _alpha_lambd - d + n_parents_j) * _slogdet_pytorch(R, parents)
            - 0.5 * (N + _alpha_lambd - d + n_parents_j + 1.0) * _slogdet_pytorch(R, parents_and_j)
        )

        # Return 0 if no observations (N=0)
        return torch.where(is_N_zero, torch.tensor(0.0, device=self.device, dtype=self.dtype), 
                          log_gamma_term + log_term_r)

    def log_marginal_likelihood(self, *, g: torch.Tensor, x: torch.Tensor, interv_targets: torch.Tensor) -> torch.Tensor:
        _N_obs, d = x.shape
        
        # Ensure inputs are on the correct device and type
        g = g.to(device=self.device, dtype=self.dtype if g.is_floating_point() else torch.int) # g can be int or float for sum, then bool
        x = x.to(device=self.device, dtype=self.dtype)
        interv_targets = interv_targets.to(device=self.device, dtype=x.dtype) # interventions often bool or float

        # Number of parents for each node [d]
        n_parents_all = g.sum(dim=0).float() # Ensure float for arithmetic in _log_marginal_likelihood_single

        # Loop over nodes (PyTorch equivalent of the JAX vmap in this context)
        node_scores = []
        for j_idx in range(d):
            score_j = self._log_marginal_likelihood_single(
                j_idx, 
                n_parents_all[j_idx].item(), # .item() to get Python float from 0-dim tensor
                g, 
                x, 
                interv_targets
            )
            node_scores.append(score_j)
        
        total_score = torch.stack(node_scores).sum()
        return total_score

    def interventional_log_marginal_prob(self, g: torch.Tensor, _: torch.Tensor, x: torch.Tensor, 
                                         interv_targets: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # Parameter `_` (theta) and `generator` are dummies for BGe (marginal likelihood)
        return self.log_marginal_likelihood(g=g, x=x, interv_targets=interv_targets)


class LinearGaussianTorch:
    """
    PyTorch implementation of the Linear Gaussian model.
    """
    def __init__(self, *, 
                 n_vars: int, 
                 obs_noise: float = 0.1, 
                 mean_edge: float = 0.0, 
                 sig_edge: float = 1.0, 
                 min_edge: float = 0.5,
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float32):
        self.n_vars = n_vars
        self.obs_noise = obs_noise
        self.obs_noise_sqrt = math.sqrt(obs_noise)
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.min_edge = min_edge
        self.device = torch.device(device)
        self.dtype = dtype

        self.no_interv_targets = torch.zeros(self.n_vars, device=self.device, dtype=torch.bool)

    def get_theta_shape(self, *, n_vars: int) -> tuple:
        return (n_vars, n_vars)

    def sample_parameters(self, *, generator: torch.Generator = None, n_vars: int, 
                          n_particles: int = 0, batch_size: int = 0) -> torch.Tensor:
        shape_dims = []
        if batch_size > 0: shape_dims.append(batch_size)
        if n_particles > 0: shape_dims.append(n_particles)
        shape_dims.extend(self.get_theta_shape(n_vars=n_vars))
        final_shape = tuple(shape_dims)

        rand_normal_fn = torch.randn if generator is None else lambda *args, **kwargs: torch.normal(0., 1., *args, **kwargs, generator=generator)
        
        theta_normal = rand_normal_fn(final_shape, device=self.device, dtype=self.dtype)
        theta = self.mean_edge + self.sig_edge * theta_normal
        theta = theta + torch.sign(theta) * self.min_edge # Add min_edge to non-zero weights
        return theta

    def sample_obs(self, *, generator: torch.Generator, n_samples: int, g_adj: torch.Tensor, 
                   theta: torch.Tensor, toporder: list, interv: dict = None) -> torch.Tensor:
        g_adj = g_adj.to(device=self.device, dtype=self.dtype) # Ensure correct device/type
        theta = theta.to(device=self.device, dtype=self.dtype)

        if interv is None: interv = {}

        x = torch.zeros((n_samples, self.n_vars), device=self.device, dtype=self.dtype)
        
        rand_normal_fn = torch.randn if generator is None else lambda *args, **kwargs: torch.normal(0., 1., *args, **kwargs, generator=generator)
        z_unit_noise = rand_normal_fn((n_samples, self.n_vars), device=self.device, dtype=self.dtype)
        z = self.obs_noise_sqrt * z_unit_noise

        for j_idx_int in toporder: # Iterate through nodes in topological order
            j_idx = int(j_idx_int) # Ensure integer for indexing and dict keys
            
            if j_idx in interv:
                x[:, j_idx] = interv[j_idx] # Apply intervention
                continue

            parents_mask = g_adj[:, j_idx].bool() # Parents of node j
            parent_indices = torch.where(parents_mask)[0]

            if len(parent_indices) > 0:
                # Mean is sum over parents: X_pa @ Theta_pa,j
                mean_val = x[:, parent_indices] @ theta[parent_indices, j_idx]
                x[:, j_idx] = mean_val + z[:, j_idx]
            else: # Node has no parents
                x[:, j_idx] = z[:, j_idx]
        return x

    def log_prob_parameters(self, *, theta: torch.Tensor, g_adj: torch.Tensor) -> torch.Tensor:
        g_adj = g_adj.to(device=self.device, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)
        
        # Gaussian prior N(mean_edge, sig_edge) on existing edges
        dist = Normal(loc=self.mean_edge, scale=self.sig_edge)
        log_probs_all_edges = dist.log_prob(theta)
        
        # Only consider edges present in g_adj
        return torch.sum(g_adj * log_probs_all_edges)

    def log_likelihood(self, *, x: torch.Tensor, theta: torch.Tensor, g_adj: torch.Tensor, 
                       interv_targets: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        theta = theta.to(device=self.device, dtype=self.dtype)
        g_adj = g_adj.to(device=self.device, dtype=self.dtype)
        interv_targets_bool = interv_targets.bool().to(device=self.device)


        # Predicted mean for each node: X_mean = X @ (G * Theta)
        # (G * Theta) is the effective weight matrix.
        predicted_means = x @ (g_adj * theta) # [N_obs, n_vars]
        
        dist = Normal(loc=predicted_means, scale=self.obs_noise_sqrt)
        log_likelihoods_all_nodes_samples = dist.log_prob(x) # [N_obs, n_vars]

        # Zero out log-likelihood for intervened node-samples
        final_log_likelihoods = torch.where(interv_targets_bool,
                                            torch.tensor(0.0, device=self.device, dtype=self.dtype),
                                            log_likelihoods_all_nodes_samples)
        return torch.sum(final_log_likelihoods)

    def interventional_log_joint_prob(self, g_adj: torch.Tensor, theta: torch.Tensor, x: torch.Tensor, 
                                      interv_targets: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # `generator` is a dummy here, not used by underlying functions
        log_prior_theta = self.log_prob_parameters(g_adj=g_adj, theta=theta)
        log_lik_data = self.log_likelihood(g_adj=g_adj, theta=theta, x=x, interv_targets=interv_targets)
        return log_prior_theta + log_lik_data