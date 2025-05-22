import torch
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# --- Assumed utility functions (would need to be translated from models/utils.py) ---
def acyclic_constr(g_mat, d):
    """
    Computes the acyclicity constraint h(G) = tr((I + alpha*G)^d) - d.
    g_mat: [d, d] tensor, the graph adjacency matrix (can be soft)
    d: number of nodes
    """
    alpha = 1.0 / d
    eye = torch.eye(d, device=g_mat.device, dtype=g_mat.dtype)
    m = eye + alpha * g_mat
    # Efficiently compute matrix power for integer d
    # For non-integer d, or for very large d where this is slow,
    # other methods like repeated squaring or eigendecomposition might be needed
    # if d is small, torch.linalg.matrix_power is fine.
    # For larger d, this can be computationally intensive.
    # The original DiBS paper uses d in the exponent.
    try:
        m_mult = torch.linalg.matrix_power(m, d)
    except Exception as e:
        # Fallback for potential issues with matrix_power if d is too large or matrix is ill-conditioned
        # This is a placeholder; a robust solution might involve iterative multiplication
        # or checking matrix properties. For now, we'll re-raise or return a large penalty.
        print(f"Warning: torch.linalg.matrix_power failed with d={d}. Error: {e}. Returning large penalty.")
        return torch.tensor(float('inf'), device=g_mat.device, dtype=g_mat.dtype)


    h = torch.trace(m_mult) - d
    return h

def stable_mean(fxs, dim=0, keepdim=False):
    """
    Computes a more stable mean, especially if fxs contains very small positive numbers.
    The JAX version had a more complex handling for positive and negative values.
    This is a simplified version assuming fxs are mostly positive or their log is taken.
    If fxs can be negative, the JAX version's logic for splitting positive/negative
    and using logsumexp on their absolute values would be needed.
    For now, we'll use a simple mean, assuming inputs are well-behaved for averaging
    or that the calling function handles logs appropriately for logsumexp-style averaging.
    """
    # A common way to stabilize mean of log-probabilities (if fxs are log-probs)
    # is logsumexp(fxs) - log(N)
    # If fxs are probabilities, torch.mean is usually fine.
    # The original `stable_mean` in JAX was more elaborate.
    # For gradients, a simple mean is often what's needed if the terms are already gradients.
    return torch.mean(fxs, dim=dim, keepdim=keepdim)

def expand_by(arr, n):
    """
    Expands the tensor by n dimensions at the end.
    arr: input tensor
    n: number of dimensions to add
    """
    for _ in range(n):
        arr = arr.unsqueeze(-1)
    return arr
# --- End of assumed utility functions ---


# === Core DiBS Logic in PyTorch ===

# Function Calling Structure:
# SVGD loop (not shown here, but would be in `fit_svgd`) calls:
#   `grad_log_joint` (to get gradients for Z and Theta)
#     `grad_log_joint` calls:
#       `update_dibs_hparams` (to anneal hyperparameters)
#       `grad_z_log_joint_gumbel` (for dZ)
#         `grad_z_log_joint_gumbel` calls:
#           `gumbel_grad_acyclic_constr_mc` (for prior gradient part)
#             `gumbel_grad_acyclic_constr_mc` calls (conceptually, via torch.autograd):
#               `gumbel_soft_gmat` (to get G_soft from Z)
#               `acyclic_constr` (the constraint function)
#           `log_full_likelihood` (for likelihood gradient part, via torch.autograd)
#             `log_full_likelihood` calls:
#               `log_gaussian_likelihood`
#               `log_bernoulli_likelihood` (if expert data `y` is present)
#           `log_theta_prior` (for likelihood gradient part, via torch.autograd)
#           `gumbel_soft_gmat` (multiple times for MC estimates)
#       `grad_theta_log_joint` (for dTheta)
#         `grad_theta_log_joint` calls:
#           `log_full_likelihood` (via torch.autograd)
#           `log_theta_prior` (via torch.autograd)
#           `bernoulli_soft_gmat` (multiple times for MC estimates)
#
# The `log_joint` function itself (calculating the value, not gradient) calls:
#   `update_dibs_hparams`
#   `log_full_likelihood`
#   `gumbel_acyclic_constr_mc` (or `bernoulli_soft_gmat` depending on formulation)
#   `log_theta_prior`

def log_gaussian_likelihood(x, pred_mean, sigma=0.1):
    """
    Calculates the log Gaussian likelihood.
    x: observed data [N, D] or [D]
    pred_mean: predicted mean [N, D] or [D]
    sigma: standard deviation (scalar or [D])
    """
    if not isinstance(sigma, float) and not isinstance(sigma, int):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
        sigma = sigma.reshape(-1)
        assert sigma.shape[0] == pred_mean.shape[-1], "Sigma shape mismatch"
    
    # Ensure sigma is a tensor for Normal distribution
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)

    # Create Normal distribution object
    # Ensure loc and scale are broadcastable with x
    if x.shape != pred_mean.shape:
        if x.ndim == 1 and pred_mean.ndim > 1 and x.shape[0] == pred_mean.shape[-1]: # x is [D], pred_mean is [N,D]
             pred_mean_expanded = pred_mean
        elif pred_mean.ndim == 1 and x.ndim > 1 and pred_mean.shape[0] == x.shape[-1]: # pred_mean is [D], x is [N,D]
             pred_mean_expanded = pred_mean.expand_as(x)
        else:
            raise ValueError(f"Shape mismatch between x ({x.shape}) and pred_mean ({pred_mean.shape}) that cannot be broadcasted.")
    else:
        pred_mean_expanded = pred_mean

    if sigma.ndim == 1 and sigma.shape[0] == x.shape[-1] and x.ndim > pred_mean_expanded.ndim : # sigma is [D]
        sigma_expanded = sigma.expand_as(x)
    elif sigma.numel() == 1 :
        sigma_expanded = sigma.expand_as(x)
    else:
        sigma_expanded = sigma


    normal_dist = Normal(loc=pred_mean_expanded, scale=sigma_expanded)
    log_prob = normal_dist.log_prob(x)
    return torch.sum(log_prob) # Sum over all dimensions (N and D)

def log_bernoulli_likelihood(y_expert_edge, soft_gmat_entry, rho, jitter=1e-5):
    """
    Log likelihood for a single expert edge belief.
    y_expert_edge: scalar tensor, 0 or 1, expert's belief about edge presence.
    soft_gmat_entry: scalar tensor, probability of edge from soft_gmat.
    rho: expert's error rate.
    """
    # p_tilde is the probability of observing y_expert_edge given g_ij (soft_gmat_entry) and rho
    # If y_expert_edge == 1 (expert says edge exists):
    #   P(y=1|g_ij) = g_ij*(1-rho) + (1-g_ij)*rho = rho + g_ij - 2*rho*g_ij (if g_ij is P(edge=1))
    # If y_expert_edge == 0 (expert says edge absent):
    #   P(y=0|g_ij) = g_ij*rho + (1-g_ij)*(1-rho) = 1 - rho - g_ij + 2*rho*g_ij
    # The JAX code uses: p_tilde = rho + g_ij - 2 * rho * g_ij
    # and then loglik = y * log(1-p_tilde) + (1-y) * log(p_tilde)
    # This implies p_tilde is P(expert is wrong OR edge state contradicts expert if expert was right)
    # Let's re-derive based on common interpretation:
    # P(expert_says_1 | true_edge_1) = 1 - rho
    # P(expert_says_1 | true_edge_0) = rho
    # P(expert_says_0 | true_edge_1) = rho
    # P(expert_says_0 | true_edge_0) = 1 - rho
    # log P(expert_y | g_ij) = expert_y * log(P(says_1|g_ij)) + (1-expert_y) * log(P(says_0|g_ij))
    # P(says_1|g_ij) = g_ij * (1-rho) + (1-g_ij) * rho
    # P(says_0|g_ij) = g_ij * rho     + (1-g_ij) * (1-rho)

    prob_expert_says_1 = soft_gmat_entry * (1.0 - rho) + (1.0 - soft_gmat_entry) * rho
    prob_expert_says_0 = soft_gmat_entry * rho + (1.0 - soft_gmat_entry) * (1.0 - rho)

    loglik = y_expert_edge * torch.log(prob_expert_says_1 + jitter) + \
             (1.0 - y_expert_edge) * torch.log(prob_expert_says_0 + jitter)
    return loglik


def scores(z, alpha_hparam):
    """
    Computes the raw scores S_ij = alpha * u_i^T v_j from latent Z.
    z: latent variables [..., D, K, 2]
    alpha_hparam: scalar hyperparameter
    Returns: scores [..., D, D]
    """
    u = z[..., 0]  # [..., D, K]
    v = z[..., 1]  # [..., D, K]
    # Einsum for batched dot product: sum over K
    # u_bdi = u[b,d,i], v_bdj = v[b,d,j]
    # scores_bd1d2 = sum_k u_bd1k * u_bd2k
    raw_scores = alpha_hparam * torch.einsum('...ik,...jk->...ij', u, v) # if u,v are [...,D,K]
    
    # Mask diagonal (no self-loops)
    if raw_scores.ndim >= 2:
        diag_mask = 1.0 - torch.eye(raw_scores.shape[-1], device=z.device, dtype=z.dtype)
        return raw_scores * diag_mask
    return raw_scores


def bernoulli_soft_gmat(z, hparams):
    """
    Generates a soft adjacency matrix (probabilities) using Bernoulli-Sigmoid.
    P(G_ij=1|Z) = sigmoid(alpha * u_i^T v_j)
    z: latent variables [N_particles, D, K, 2] or [D, K, 2]
    hparams: dictionary containing 'alpha'
    Returns: soft_gmat [N_particles, D, D] or [D,D] (probabilities)
    """
    raw_scores = scores(z, hparams['alpha'])
    return torch.sigmoid(raw_scores) # tau is not used here unlike gumbel_soft_gmat


def gumbel_soft_gmat(z, hparams, device='cpu'):
    """
    Generates a soft adjacency matrix using Gumbel-Sigmoid reparameterization.
    G_ij = sigmoid(tau * (logistic_noise_ij + alpha * u_i^T v_j))
    z: latent variables [D, K, 2] (for a single particle)
    hparams: dictionary containing 'alpha', 'tau'
    Returns: soft_gmat [D, D] (differentiable soft samples)
    """
    d = z.shape[0]
    raw_scores = scores(z, hparams['alpha']) # [D, D]

    # Sample Logistic noise L_ij ~ Logistic(0,1)
    # PyTorch's Logistic distribution is location=0, scale=1 by default
    logistic_dist = Logistic(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    # JAX uses jax.random.logistic(key, shape=(d,d)).
    # PyTorch equivalent:
    # Standard logistic U(0,1) -> log(U) - log(1-U)
    # Or use logistic_dist.sample((d,d)).squeeze(-1) if it adds extra dim
    logistic_noise = logistic_dist.sample((d, d)).squeeze(-1)

    # Gumbel-Sigmoid
    # The JAX code uses `(_l[i,j] + score) * tau`.
    # Here, _l is logistic noise.
    gumbel_input = logistic_noise + raw_scores # JAX version adds score to logistic noise
    soft_gmat = torch.sigmoid(hparams['tau'] * gumbel_input)

    # Mask diagonal
    diag_mask = 1.0 - torch.eye(d, device=device, dtype=z.dtype)
    return soft_gmat * diag_mask


def log_full_likelihood(data, soft_gmat, current_theta, hparams, device='cpu'):
    """
    Computes log P(Data | Z, Theta).
    data: dict containing 'x' (observational data [N,D]) and optionally 'y' (expert edges list of [i,j,val])
    soft_gmat: The soft adjacency matrix [D,D], typically generated by gumbel_soft_gmat or bernoulli_soft_gmat.
    current_theta: parameters Theta for the current particle [D, D] (for linear model)
    hparams: dict of hyperparameters
    """
    # For likelihood, typically use Bernoulli soft gmat (probabilities)
    # or Gumbel soft gmat if that's part of the likelihood definition (e.g. for certain gradient paths)
    # The JAX log_joint uses bernoulli_soft_gmat for the likelihood term.
    # The JAX grad_z_log_joint_gumbel uses gumbel_soft_gmat for its internal log_density_z.
    # Let's assume for calculating the likelihood value, bernoulli_soft_gmat is appropriate.
    # If this function is used inside a gradient computation that requires Gumbel, that needs to be handled.
    # The `log_density_z` in JAX's `grad_z_log_joint_gumbel` passes `gumbel_soft_gmat(k, z, ...)`
    # to `log_full_likelihood`. So this function needs to accept a pre-computed soft_gmat.

    # Let's modify signature to accept soft_gmat directly,
    # as the caller (`log_density_z` or `log_joint`) will decide which type of soft_gmat.
    # def log_full_likelihood(data, soft_gmat_from_z, current_theta, hparams, device='cpu'):

    # For now, let's stick to the original call and decide inside or assume it's for log_joint value
    # The JAX `log_joint` uses `bernoulli_soft_gmat` for its likelihood.
    # The JAX `grad_z_log_joint_gumbel`'s `log_density_z` uses `gumbel_soft_gmat`.
    # This means `log_full_likelihood` must be flexible or called with the correct `soft_gmat`.
    # Let's assume the `soft_gmat` argument is passed in, matching `log_density_z`'s usage.
    # This function is called by `log_density_z` (for dZ) and `theta_log_joint` (for dTheta).
    # `log_density_z` uses `gumbel_soft_gmat`.
    # `theta_log_joint` uses `bernoulli_soft_gmat`.
    # So, the `soft_gmat` argument is essential.

    # Let's assume the signature is:
    # def log_full_likelihood(data, soft_gmat_arg, current_theta, hparams, device='cpu'):
    # And the original JAX code was:
    # log_full_likelihood(data, soft_gmat=gumbel_soft_gmat(k, z, ...), theta=nonopt_params["theta"], ...)
    # This means the `soft_gmat` argument in the JAX code is the result of `gumbel_soft_gmat` or `bernoulli_soft_gmat`.
    # So, the PyTorch version should also expect `soft_gmat_arg`.
    # The original `dibs.py` has `log_full_likelihood(data, soft_gmat, hard_gmat, theta, hparams)`
    # The `hard_gmat` argument is often None.

    # Let's stick to the JAX function signature as much as possible.
    # `soft_gmat` here is the one passed by the caller.
    # `hard_gmat` is often None.

    # Renaming arguments for clarity based on JAX version
    # def log_full_likelihood(data_dict, soft_gmat_tensor, hard_gmat_tensor_or_none, theta_tensor, hparams_dict, device='cpu'):
    
    # Using names from the original JAX function for consistency in this file
    # data, soft_gmat, hard_gmat (often None), theta, hparams

    # Observational likelihood (Gaussian)
    # pred_mean = data['x'] @ (current_theta * soft_gmat) # Assumes data['x'] is [N, D]
    # The JAX code uses data["x"] @ (theta * soft_gmat)
    # This implies data['x'] is [N, D], theta is [D, D], soft_gmat is [D, D]
    # Resulting pred_mean is [N, D]
    # This is for a linear model where theta contains coefficients.
    # For PyTorch, ensure data['x'] is a tensor.
    x_data = data['x'] # Shape [N, D]
    
    # Effective weighted adjacency matrix
    # Theta contains potential coefficients, soft_gmat gates/weights them
    effective_W = current_theta * soft_gmat # [D, D]
    
    # Predicted mean: X_pred = X_obs @ W_eff
    # This is a common formulation for Structural Equation Models / linear autoregressive processes
    # X_pred_i = sum_j X_obs_j * W_eff_ji
    pred_mean = torch.matmul(x_data, effective_W) # [N, D]

    # sigma_obs is a hyperparameter, e.g., hparams.get('sigma_obs_noise', 0.1)
    sigma_obs = hparams.get('sigma_obs_noise', 0.1) # Get from hparams or default
    log_obs_likelihood = log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

    log_expert_likelihood_total = torch.tensor(0.0, device=device)
    inv_temperature_expert = 0.0

    if data.get("y", None) is not None and len(data["y"]) > 0:
        inv_temperature_expert = hparams.get("temp_ratio", 0.0) # JAX uses hparams["temp_ratio"]
        
        # data["y"] is a list of [i, j, val]
        # We need to iterate through these expert beliefs.
        expert_log_probs = []
        for expert_belief in data["y"]:
            i, j, val = int(expert_belief[0]), int(expert_belief[1]), expert_belief[2].item() # Ensure indices are int
            
            # Get the corresponding P(edge_ij=1) from the soft_gmat
            g_ij_prob = soft_gmat[i, j]
            
            # Calculate log likelihood for this single expert belief
            # Ensure val is a tensor for log_bernoulli_likelihood
            y_val_tensor = torch.tensor(val, dtype=g_ij_prob.dtype, device=g_ij_prob.device)
            expert_log_probs.append(
                log_bernoulli_likelihood(y_val_tensor, g_ij_prob, hparams["rho"])
            )
        
        if expert_log_probs:
            log_expert_likelihood_total = torch.sum(torch.stack(expert_log_probs))

    return inv_temperature_expert * log_expert_likelihood_total + log_obs_likelihood


def log_theta_prior(theta_effective, theta_prior_mean, theta_prior_sigma):
    """
    Log prior probability of Theta (Gaussian).
    theta_effective: The parameters being penalized (e.g., theta * G_soft)
    theta_prior_mean: Mean of the Gaussian prior (often zeros)
    theta_prior_sigma: Std dev of the Gaussian prior
    """
    # Ensure theta_prior_mean is a tensor with compatible shape or broadcastable
    if not torch.is_tensor(theta_prior_mean):
        theta_prior_mean = torch.full_like(theta_effective, theta_prior_mean)
        
    return log_gaussian_likelihood(theta_effective, theta_prior_mean, sigma=theta_prior_sigma)


# For acyclicity constraint, we need gumbel_soft_gmat and acyclic_constr
# The JAX code has `gumbel_acyclic_constr_mc` which vmaps `acyclic_constr(jax.random.bernoulli(k, soft_gmat), d)`
# or `gumbel_acyclic_constr_mc` which vmaps `acyclic_constr(soft_gmat(k), d)`
# Let's implement the version that takes soft_gmat samples.

def gumbel_acyclic_constr_mc(z_particle, d, hparams, n_mc_samples, device='cpu'):
    """
    Monte Carlo estimate of E_{G_soft ~ p(G_soft|Z)} [h(G_soft)] using Gumbel-soft samples.
    Or, if h(G) is defined on hard graphs, E_{G_hard ~ Bernoulli(G_soft)} [h(G_hard)].
    The JAX `gumbel_acyclic_constr_mc` uses `acyclic_constr(soft_gmat(k), d)`.
    The JAX `acyclic_constr_mc` uses `acyclic_constr(jax.random.bernoulli(k, bernoulli_soft_gmat(z,d,hparams)), d)`.
    Let's assume the one for `log_joint` which is `gumbel_acyclic_constr_mc` in JAX,
    meaning we average h(G_soft) where G_soft are Gumbel samples.
    z_particle: single Z particle [D, K, 2]
    d: number of nodes
    hparams: hyperparameters
    n_mc_samples: number of MC samples for the expectation
    """
    h_samples = []
    for _ in range(n_mc_samples):
        # Each Gumbel sample needs a fresh noise sample L, but z is fixed for this expectation.
        # The JAX `gumbel_soft_gmat(k, z, ...)` takes a key `k` for randomness.
        # Here, we generate noise inside gumbel_soft_gmat.
        g_soft = gumbel_soft_gmat(z_particle, hparams, device=device)
        h_samples.append(acyclic_constr(g_soft, d))
    
    if not h_samples: # Should not happen if n_mc_samples > 0
        return torch.tensor(0.0, device=device)
        
    return torch.mean(torch.stack(h_samples))


# --- Gradient Computations ---
# These are the most complex to translate due to JAX's `grad` and `vmap`.
# We'll use `torch.autograd.grad` and manual looping or batching for vmap.

def gumbel_grad_acyclic_constr_mc(z_particle_opt, d, hparams_dict_nonopt, n_mc_samples, create_graph=False):
    """
    Computes E_L [grad_Z h(G_soft(Z,L))]
    z_particle_opt: The Z tensor [D,K,2] for which we need gradients. Must have requires_grad=True.
    d: number of nodes
    hparams_dict_nonopt: hyperparameters
    n_mc_samples: number of MC samples for the expectation
    create_graph: bool, for torch.autograd.grad
    """
    # This function computes the gradient of E_L[h(G_soft(Z,L))] w.r.t Z.
    # = E_L[grad_Z h(G_soft(Z,L))]
    # We need to sample L, compute G_soft, compute h(G_soft), then get grad_Z h(G_soft).
    
    grad_h_samples = []
    for _ in range(n_mc_samples):
        # Ensure z_particle_opt is a leaf and requires grad for this specific sample's computation
        # If z_particle_opt is already a parameter being optimized by an outer loop, this is fine.
        # If it's an intermediate tensor, it might need to be cloned and set to require_grad.
        # For now, assume z_particle_opt is the parameter we are differentiating.

        # Generate G_soft(Z,L). This operation must be part of the graph for autograd.
        g_soft = gumbel_soft_gmat(z_particle_opt, hparams_dict_nonopt, device=z_particle_opt.device)
        h_val = acyclic_constr(g_soft, d)
        
        # Compute gradient of h_val w.r.t. z_particle_opt for this one MC sample
        # `h_val` is a scalar. `z_particle_opt` is the tensor of interest.
        grad_h_val_wrt_z, = torch.autograd.grad(
            outputs=h_val,
            inputs=z_particle_opt,
            grad_outputs=torch.ones_like(h_val), # for scalar output
            retain_graph=True, # Potentially needed if z_particle_opt is used in multiple h_samples
                               # or if the graph is needed later. Set to False if not.
                               # If this is the only use of z_particle_opt for this h_val, False is fine.
                               # For an expectation, we sum gradients, so True is safer if z_particle_opt is the same object.
            create_graph=create_graph # If we need higher-order derivatives
        )
        grad_h_samples.append(grad_h_val_wrt_z)

    if not grad_h_samples:
        return torch.zeros_like(z_particle_opt)
        
    # Average the gradients
    # The JAX code uses jnp.mean(..., axis=0)
    # Here, stack and mean over the sample dimension (dim=0)
    avg_grad_h = torch.mean(torch.stack(grad_h_samples), dim=0)
    return avg_grad_h


def grad_z_log_joint_gumbel(current_z_opt, current_theta_nonopt, data_dict, hparams_full, device='cpu'):
    """
    Computes gradient of log P(Z, Theta, Data) w.r.t. Z. (Eq 10 in DiBS paper, adapted)
    current_z_opt: Z tensor [D,K,2] requiring grad.
    current_theta_nonopt: Theta tensor [D,D], treated as fixed for this grad computation.
    data_dict: Data.
    hparams_full: Combined hyperparameters and current non-optimal params (like current_theta_nonopt).
    """
    d = current_z_opt.shape[0]
    beta = hparams_full['beta']
    sigma_z_sq = hparams_full['sigma_z']**2
    n_grad_mc_samples = hparams_full['n_grad_mc_samples'] # For likelihood part

    # 1. Gradient of log prior on Z: grad_Z [ -beta * E[h(G_soft)] - 0.5/sigma_z^2 * ||Z||^2 ]
    # 1a. grad_Z E[h(G_soft)]
    # The JAX code uses `gumbel_grad_acyclic_constr_mc` which averages grads.
    # n_nongrad_mc_samples from hparams is used by JAX's gumbel_grad_acyclic_constr_mc.
    n_acyclic_mc_samples = hparams_full.get('n_nongrad_mc_samples', hparams_full['n_grad_mc_samples'])
    grad_expected_h = gumbel_grad_acyclic_constr_mc(
        current_z_opt, d, hparams_full, n_acyclic_mc_samples, create_graph=False # No higher order grads needed here
    )
    grad_log_z_prior_acyclicity = -beta * grad_expected_h
    
    # 1b. grad_Z [- 1/(2*sigma_z^2) * sum(Z_flat^2) ] = - Z / sigma_z^2
    # (Assuming prior is N(0, sigma_z^2) for each element of Z, so logpdf is -0.5*z^2/sigma_z^2 - log(sigma_z*sqrt(2pi)))
    # The JAX code has `-(1 / nonopt_params["sigma_z"] ** 2) * opt_params["z"]`. This implies sum over elements.
    # If prior is sum_elements N(0, sigma_z), then log prior is sum (-0.5 * (z_ij/sigma_z)^2)
    # grad is -z_ij / sigma_z^2.
    grad_log_z_prior_gaussian = -current_z_opt / sigma_z_sq
    
    grad_log_z_prior_total = grad_log_z_prior_acyclicity + grad_log_z_prior_gaussian

    # 2. Gradient of E_G_soft [ log P(Theta, Data | G_soft(Z)) ] (log of expectation, using score identity)
    #    This is the term: (grad_Z E[P(Theta,Data|G_soft)]) / E[P(Theta,Data|G_soft)]
    #    The JAX code implements this using logsumexp for stability with MC samples of
    #    log P(Theta, Data | G_soft(Z, L)) and grad_Z log P(Theta, Data | G_soft(Z, L)).

    # Define log_density_z_fn(z_arg, l_noise_idx_for_gumbel_key)
    # This function calculates log P(Theta, Data | G_soft(z_arg, L)) + log P(Theta_eff | G_soft(z_arg,L))
    # JAX version: log_density_z = lambda k_gumbel, z_lambda: log_full_likelihood(...) + log_theta_prior(...)
    # where soft_gmat in log_full_likelihood is gumbel_soft_gmat(k_gumbel, z_lambda, ...)
    # and theta_effective in log_theta_prior is current_theta_nonopt * gumbel_soft_gmat(k_gumbel, z_lambda, ...)

    log_density_values_for_mc = []
    # For grad_z_of_log_density_values_for_mc, we need to compute grad w.r.t. current_z_opt.
    # This requires current_z_opt to be part of the computation graph for each MC sample.
    
    # Store gradients of log_density w.r.t current_z_opt for each MC sample
    # PyTorch autograd.grad is better here than .backward() if we want specific gradients.
    
    # For the stable gradient computation (similar to JAX logsumexp trick for gradients):
    # We need:
    #   log_p_samples: [n_grad_mc_samples] - values of log_density_z(k, current_z_opt)
    #   grad_log_p_samples: [n_grad_mc_samples, D, K, 2] - values of grad_z log_density_z(k, z) |_{z=current_z_opt}

    log_p_samples_list = []
    grad_log_p_wrt_z_list = []

    for i in range(n_grad_mc_samples):
        # Ensure z_opt is used in a way that its gradient can be taken for this sample
        # If current_z_opt is a parameter, it's fine.
        
        # G_soft for this MC sample, depends on current_z_opt and fresh Logistic noise
        # This G_soft must be constructed in a way that tracks gradients back to current_z_opt
        g_soft_mc = gumbel_soft_gmat(current_z_opt, hparams_full, device=device) # differentiable wrt current_z_opt
        
        # Calculate log P(Data | G_soft, Theta_nonopt)
        log_lik_val = log_full_likelihood(data_dict, g_soft_mc, current_theta_nonopt, hparams_full, device=device)
        
        # Calculate log P(Theta_eff | G_soft)
        # Theta_eff = Theta_nonopt * G_soft
        theta_eff_mc = current_theta_nonopt * g_soft_mc
        # Prior mean and sigma for theta from hparams
        theta_prior_mean_val = torch.zeros_like(current_theta_nonopt, device=device) # Example
        theta_prior_sigma_val = hparams_full.get('theta_prior_sigma', 1.0)
        log_theta_prior_val = log_theta_prior(theta_eff_mc, theta_prior_mean_val, theta_prior_sigma_val)
        
        current_log_density = log_lik_val + log_theta_prior_val
        log_p_samples_list.append(current_log_density.detach()) # Detach for storing value, grad comes next

        # Gradient of current_log_density w.r.t current_z_opt
        if current_z_opt.grad is not None:
            current_z_opt.grad.zero_() # Zero out previous grads if any (though torch.autograd.grad doesn't accumulate like .backward())
            
        grad_curr_log_density_wrt_z, = torch.autograd.grad(
            outputs=current_log_density,
            inputs=current_z_opt,
            retain_graph=True, # Important: current_z_opt is used across MC samples
            create_graph=False # Not taking higher-order derivatives of this gradient
        )
        grad_log_p_wrt_z_list.append(grad_curr_log_density_wrt_z)

    # Stack the collected tensors
    log_p_samples = torch.stack(log_p_samples_list) # [n_grad_mc_samples]
    grad_log_p_wrt_z_samples = torch.stack(grad_log_p_wrt_z_list) # [n_grad_mc_samples, D, K, 2]

    # Clean up graph for current_z_opt if retain_graph was True for the last grad computation
    # This is tricky. If current_z_opt is a leaf node in an outer optimization,
    # its graph should be retained until the optimizer step.
    # If torch.autograd.grad was used with retain_graph=True, need to be careful.
    # A common pattern is to ensure ops inside the loop don't affect subsequent iterations' grad computations
    # unless intended. Cloning z_opt for each loop or careful grad zeroing might be needed if issues arise.
    # For now, assume `retain_graph=True` in `autograd.grad` is handled correctly by SVGD's structure.

    # Compute stable gradient: sum_s (w_s * grad_s) where w_s = exp(log_p_s - logsumexp(log_p_all))
    # This is E_{L_s}[grad_Z log P(Theta,Data|G(Z,L_s))] where the expectation is weighted by P(Theta,Data|G(Z,L_s))
    # JAX code:
    # lse_numerator = tree_map(lambda leaf_z: logsumexp(a=expand_by(log_p_samples, leaf_z.ndim-1), b=leaf_z, axis=0, return_sign=True)[0], grad_log_p_wrt_z_samples)
    # sign_lse_numerator = ... [1]
    # lse_denominator = logsumexp(a=log_p_samples, axis=0)
    # stable_grad_lik_part = sign * exp(lse_num - log(N) - lse_den + log(N)) = sign * exp(lse_num - lse_den)

    # PyTorch equivalent:
    # log_weights = F.log_softmax(log_p_samples, dim=0) # log_p_s - logsumexp(log_p_all)
    # weights = torch.exp(log_weights) # [n_grad_mc_samples]
    # grad_lik_part = torch.sum(weights.reshape(-1, 1, 1, 1) * grad_log_p_wrt_z_samples, dim=0)
    
    # Let's follow JAX more closely for the "gradient of log of expectation"
    # grad_log_E[X] = (grad E[X]) / E[X] = (E[grad X]) / E[X] if reparam, or (E[X grad log P]) / E[X] if score fn.
    # The JAX code is essentially computing E_L[grad_z log_density(Z,L)] / E_L[1] but weighted.
    # It's an estimator for grad_Z log E_L [ exp(log_density(Z,L)) ]
    # = ( E_L [ exp(log_density(Z,L)) * grad_Z log_density(Z,L) ] ) / E_L [ exp(log_density(Z,L)) ]

    # To avoid numerical issues with exp(log_p_samples):
    log_p_max = torch.max(log_p_samples)
    shifted_log_p = log_p_samples - log_p_max # For stability
    exp_shifted_log_p = torch.exp(shifted_log_p) # These are proportional to actual probabilities

    # Numerator term for each sample: exp_shifted_log_p[s] * grad_log_p_wrt_z_samples[s]
    # Sum these up, then divide by sum(exp_shifted_log_p)
    
    # Reshape for broadcasting: exp_shifted_log_p needs to be [N, 1, 1, 1]
    exp_shifted_log_p_reshaped = exp_shifted_log_p.reshape(-1, *([1]*(current_z_opt.ndim)))

    numerator_sum = torch.sum(exp_shifted_log_p_reshaped * grad_log_p_wrt_z_samples, dim=0)
    denominator_sum = torch.sum(exp_shifted_log_p)

    if denominator_sum < 1e-9: # Avoid division by zero
        grad_log_likelihood_part = torch.zeros_like(current_z_opt)
    else:
        grad_log_likelihood_part = numerator_sum / denominator_sum
        
    return grad_log_z_prior_total + grad_log_likelihood_part


def grad_theta_log_joint(current_z_nonopt, current_theta_opt, data_dict, hparams_full, device='cpu'):
    """
    Computes gradient of log P(Z, Theta, Data) w.r.t. Theta. (Eq 11 in DiBS paper, adapted)
    current_z_nonopt: Z tensor [D,K,2], treated as fixed.
    current_theta_opt: Theta tensor [D,D] requiring grad.
    data_dict: Data.
    hparams_full: Combined hyperparameters.
    """
    d = current_z_nonopt.shape[0]
    n_grad_mc_samples = hparams_full['n_grad_mc_samples']

    # For grad_theta, the paper (Eq A.33) suggests using Bernoulli soft gmat (not Gumbel).
    # This is because Theta's gradient doesn't involve differentiating the graph sampling process itself,
    # but rather differentiating the likelihood and theta prior for a given graph distribution from Z.
    # The expectation is over G ~ Bernoulli(sigma(Z)).
    # The JAX code's `theta_log_joint` lambda uses `bernoulli_soft_gmat(nonopt_params["z"], ...)`

    log_p_samples_list = []
    grad_log_p_wrt_theta_list = []

    # The `bernoulli_soft_gmat` depends on Z, which is fixed here. So, it's computed once.
    # However, the JAX `theta_log_joint` lambda takes a key `_k` which is unused, implying
    # the stochasticity for this expectation might come from sampling hard graphs G ~ Bernoulli(G_soft(Z)).
    # Let's re-check Eq 11 and A.33.
    # Eq 11: E_{p(G|Z)} [ grad_Theta p(Theta,D|G) ] / E_{p(G|Z)} [ p(Theta,D|G) ]
    # where p(G|Z) is the Bernoulli model.
    # This means we sample hard G's from Bernoulli(sigma(Z)).
    # For each hard G, we compute log P(D|G,Theta_opt) + log P(Theta_opt|G) and its gradient wrt Theta_opt.

    g_soft_from_fixed_z = bernoulli_soft_gmat(current_z_nonopt, hparams_full) # [D,D] probabilities

    for i in range(n_grad_mc_samples):
        # Sample a hard G ~ Bernoulli(g_soft_from_fixed_z)
        # This hard_g_mc is what's used as `soft_gmat` in log_full_likelihood and `theta_eff`
        # if we follow the "expectation over G" literally.
        # However, the JAX code's `theta_log_joint` passes `bernoulli_soft_gmat` (which is soft)
        # directly to `log_full_likelihood` and `log_theta_prior`.
        # This implies the expectation might be implicitly handled by using the soft matrix directly
        # as an expected adjacency, or the "G" in E[p(Theta,D|G)] is the soft G.
        # Given the JAX code structure, it seems they use the *soft* bernoulli_soft_gmat directly.
        # This is a subtle point. If E_p(G|Z) means G is hard, then we sample.
        # If it means G is the expected adjacency (soft), we use it directly.
        # The paper's Eq. 6 defines p(G|Z) for discrete G.
        # The gradient estimators in Sec 4.3 are for E_{p(G|Z)}[f(G)].
        # Let's assume for grad_theta, they use the soft bernoulli matrix directly as G_soft.
        # This matches the JAX `theta_log_joint` lambda.
        
        g_soft_for_lik = g_soft_from_fixed_z # Use the same soft matrix for all "samples" if not sampling hard G

        log_lik_val = log_full_likelihood(data_dict, g_soft_for_lik, current_theta_opt, hparams_full, device=device)
        
        theta_eff_mc = current_theta_opt * g_soft_for_lik # Effective theta based on this g_soft
        theta_prior_mean_val = torch.zeros_like(current_theta_opt, device=device)
        theta_prior_sigma_val = hparams_full.get('theta_prior_sigma', 1.0)
        log_theta_prior_val = log_theta_prior(theta_eff_mc, theta_prior_mean_val, theta_prior_sigma_val)
        
        current_log_density = log_lik_val + log_theta_prior_val
        log_p_samples_list.append(current_log_density.detach())

        if current_theta_opt.grad is not None:
            current_theta_opt.grad.zero_()
            
        grad_curr_log_density_wrt_theta, = torch.autograd.grad(
            outputs=current_log_density,
            inputs=current_theta_opt,
            retain_graph=True, # current_theta_opt is used across MC samples
            create_graph=False
        )
        grad_log_p_wrt_theta_list.append(grad_curr_log_density_wrt_theta)

    log_p_samples = torch.stack(log_p_samples_list)
    grad_log_p_wrt_theta_samples = torch.stack(grad_log_p_wrt_theta_list)
    
    # Stable gradient computation (same logic as for Z)
    log_p_max = torch.max(log_p_samples)
    shifted_log_p = log_p_samples - log_p_max
    exp_shifted_log_p = torch.exp(shifted_log_p)
    
    exp_shifted_log_p_reshaped = exp_shifted_log_p.reshape(-1, *([1]*(current_theta_opt.ndim)))

    numerator_sum = torch.sum(exp_shifted_log_p_reshaped * grad_log_p_wrt_theta_samples, dim=0)
    denominator_sum = torch.sum(exp_shifted_log_p)

    if denominator_sum < 1e-9:
        grad_log_likelihood_part_theta = torch.zeros_like(current_theta_opt)
    else:
        grad_log_likelihood_part_theta = numerator_sum / denominator_sum
        
    return grad_log_likelihood_part_theta


def grad_log_joint(params, data_dict, hparams_dict_config, device='cpu'):
    """
    Computes gradients of the log joint P(Z, Theta, Data) w.r.t Z and Theta.
    This is the main gradient function used by SVGD.
    params: dict containing current 'z' and 'theta' tensors for one particle.
            These tensors should have requires_grad=True.
    data_dict: dict of data.
    hparams_dict_config: dict of hyperparameters from config.
    """
    # Ensure params require grad
    current_z = params['z'] # Should be [D, K, 2] for a single particle
    current_theta = params['theta'] # Should be [D, D] for a single particle
    
    # It's crucial that current_z and current_theta are the actual parameters
    # from the optimizer, or that gradients can flow back to them.
    # If they are detached copies, set requires_grad=True.
    if not current_z.requires_grad:
        current_z.requires_grad_(True)
    if not current_theta.requires_grad:
        current_theta.requires_grad_(True)

    t_anneal = params.get('t', torch.tensor([0.0], device=device)).item() # Annealing step
    # JAX uses params['t'].reshape(-1)[0].astype(int)
    # Ensure t_anneal is an int or float suitable for hparam update
    # The JAX code adds 1 to t in the loop before passing to update_dibs_hparams
    
    # Create a mutable copy for hparams to be updated
    hparams_updated_for_step = hparams_dict_config.copy() # Shallow copy
    # If hparams_dict_config contains nested dicts, deepcopy might be safer if they are modified.
    # For DiBS, update_dibs_hparams modifies top-level keys.
    hparams_updated_for_step = update_dibs_hparams(hparams_updated_for_step, t_anneal)


    # Combine current params (like theta if needed by grad_z, or z if needed by grad_theta)
    # with hparams_updated_for_step for the `nonopt_params` or `hparams_full` arguments.
    # JAX uses `params | hparams` which merges dicts.
    hparams_and_current_params = {**hparams_updated_for_step, **params} # Python 3.5+
    # Or:
    # hparams_and_current_params = hparams_updated_for_step.copy()
    # hparams_and_current_params.update(params)


    # angelo_fortuin_anneal = 1.0 # As in JAX code for now

    # Compute grad_z
    # For grad_z_log_joint_gumbel, current_theta is non-optimal (fixed)
    grad_z = grad_z_log_joint_gumbel(
        current_z_opt=current_z,
        current_theta_nonopt=current_theta.detach(), # Treat theta as fixed for dZ
        data_dict=data_dict,
        hparams_full=hparams_and_current_params, # Pass merged params
        device=device
    )
    
    # Compute grad_theta
    # For grad_theta_log_joint, current_z is non-optimal (fixed)
    grad_theta = grad_theta_log_joint(
        current_z_nonopt=current_z.detach(), # Treat z as fixed for dTheta
        current_theta_opt=current_theta,
        data_dict=data_dict,
        hparams_full=hparams_and_current_params, # Pass merged params
        device=device
    )
    
    # The 't' parameter in JAX is a dummy for annealing, its gradient is 0.
    return {"z": grad_z, "theta": grad_theta, "t": torch.tensor([0.0], device=device)}


def log_joint(params, data_dict, hparams_dict_config, device='cpu'):
    """
    Computes the log joint density log P(Z, Theta, Data).
    params: dict containing 'z' and 'theta' for one particle.
    data_dict: Data.
    hparams_dict_config: Hyperparameters.
    """
    current_z = params['z']
    current_theta = params['theta']
    t_anneal = params.get('t', torch.tensor([0.0], device=device)).item()

    hparams_updated = update_dibs_hparams(hparams_dict_config.copy(), t_anneal)
    
    d = current_z.shape[0] # Assuming z is [D, K, 2]

    # 1. Log Likelihood part: log P(Data | Z, Theta)
    #    Uses bernoulli_soft_gmat as per JAX log_joint
    g_soft_for_lik = bernoulli_soft_gmat(current_z, hparams_updated)
    log_lik = log_full_likelihood(data_dict, g_soft_for_lik, current_theta, hparams_updated, device=device)

    # 2. Log Prior part: log P(Z) + log P(Theta | Z)
    # 2a. Log P(Z) = log P_gaussian(Z) + log P_acyclic(Z)
    #     log P_gaussian(Z)
    log_prior_z_gaussian = torch.sum(Normal(0.0, hparams_updated['sigma_z']).log_prob(current_z))
    
    #     log P_acyclic(Z) = -beta * E_G_soft~p(G|Z) [h(G_soft)]
    #     The JAX log_joint uses `gumbel_acyclic_constr_mc` for the prior.
    #     This uses Gumbel soft matrices for the expectation of h(G).
    n_acyclic_mc_samples = hparams_updated.get('n_nongrad_mc_samples', hparams_updated['n_grad_mc_samples'])
    expected_h_val = gumbel_acyclic_constr_mc(
        current_z, d, hparams_updated, n_acyclic_mc_samples, device=device
    )
    log_prior_z_acyclic = -hparams_updated['beta'] * expected_h_val
    
    log_prior_z_total = log_prior_z_gaussian + log_prior_z_acyclic

    # 2b. Log P(Theta | Z)
    #     Prior on theta_effective = theta * G_soft_bernoulli
    g_soft_for_theta_prior = bernoulli_soft_gmat(current_z, hparams_updated)
    theta_eff = current_theta * g_soft_for_theta_prior
    theta_prior_mean = torch.zeros_like(current_theta, device=device) # Example
    theta_prior_sigma = hparams_updated.get('theta_prior_sigma', 1.0)
    log_prior_theta_given_z = log_theta_prior(theta_eff, theta_prior_mean, theta_prior_sigma)
    
    total_log_prior = log_prior_z_total + log_prior_theta_given_z
    
    return log_lik + total_log_prior


def update_dibs_hparams(hparams_dict, t_step):
    """
    Updates hyperparameters that anneal with time step t_step.
    Modifies hparams_dict in place or returns a new one.
    JAX version returns a new dict.
    """
    # Ensure t_step is not zero to avoid division by zero if using (t + 1/t)
    # The JAX SVGD loop adds 1 to t before passing to update_dibs_hparams.
    # So, t_step here corresponds to (actual_iteration + 1).
    
    # Make a copy to avoid modifying the original config dict if it's reused.
    updated_hparams = hparams_dict.copy()

    # Original JAX annealing: factor = (t + 1/t)
    # This factor decreases then increases. For annealing, we usually want monotonic increase/decrease.
    # Example: linear annealing or exponential decay/increase.
    # If t_step is iter_num + 1:
    annealing_factor = 1.0 # Default, no annealing
    if t_step > 0: # Basic check
        # A common annealing schedule is to increase beta, tau, alpha over time.
        # Let's use a simple linear increase for demonstration, or keep JAX's if intended.
        # factor = t_step / total_steps (for linear increase from 0 to 1)
        # factor = initial_value * decay_rate ** t_step (for exponential)
        # The JAX code uses `hparams["param"] * (t + 1 / t)` which is unusual for typical annealing.
        # It might be specific to their empirical findings or a typo.
        # If t is large, (t + 1/t) approx t.
        # If t is small (e.g., 1), factor is 2. If t=0.5, factor is 2.5. If t=0.1, factor is 10.1.
        # This suggests t should be > 0 and perhaps not too small.
        # Given the SVGD loop sets t = iteration_number + 1, t_step >= 1.
        # For t_step=1, factor=2. For t_step=2, factor=2.5. For t_step=10, factor=10.1.
        # This is an increasing factor.
        if t_step == 0: # Should not happen if t = iter + 1
            factor = 1.0 # Or some initial large value if it's 1/t like
        else:
            factor = t_step + (1.0 / t_step)
            # To prevent excessive amplification at early stages if base values are large:
            # factor = min(factor, some_cap_if_needed) 
            # Or simply ensure base tau/alpha/beta are small.
            # The JAX code directly multiplies: `hparams["tau"] * (t + 1 / t)`
            # This means the base hparams['tau'] should be the value at factor=1.
            # The JAX code in `update_dibs_hparams` was:
            # updated_hparams["tau"] = hparams["tau"] # No annealing in provided snippet, but paper implies it
            # updated_hparams["alpha"] = hparams["alpha"] * (t + 1 / t)
            # updated_hparams["beta"] = hparams["beta"] * (t + 1 / t)
            # Let's follow this for alpha and beta. Tau is often annealed too.
            # The paper mentions annealing alpha and beta to infinity, and tau for Gumbel-softmax.

    # Assuming base hparams are for t=0 or some reference point.
    # If hparams already contains the annealed value, this logic is wrong.
    # The JAX `update_dibs_hparams` seems to take the *base* hparams and current t.
    
    # Let's assume hparams_dict contains the *base* values.
    if 'tau_base' in hparams_dict: # Example: if we want to anneal tau
        updated_hparams["tau"] = hparams_dict.get("tau_base", 1.0) * factor
    else: # If 'tau' is already the value to be used or not annealed by this factor
        updated_hparams["tau"] = hparams_dict.get("tau", 1.0) # Default if not present

    if 'alpha_base' in hparams_dict:
        updated_hparams["alpha"] = hparams_dict["alpha_base"] * factor
    elif 'alpha' in hparams_dict: # If alpha itself is the base to be multiplied
         updated_hparams["alpha"] = hparams_dict["alpha"] * factor


    if 'beta_base' in hparams_dict:
        updated_hparams["beta"] = hparams_dict["beta_base"] * factor
    elif 'beta' in hparams_dict:
        updated_hparams["beta"] = hparams_dict["beta"] * factor
        
    return updated_hparams


def hard_gmat_particles_from_z(z_particles, alpha_hparam_for_scores=1.0):
    """
    Converts Z particles to hard adjacency matrices by thresholding scores.
    z_particles: [N_particles, D, K, 2]
    alpha_hparam_for_scores: alpha used in scores() function.
    Returns: hard_gmat_particles [N_particles, D, D]
    """
    # scores() computes alpha * u^T v and masks diagonal
    # For hard_gmat, we typically use the raw scores before sigmoid.
    # The JAX `hard_gmat_particles_from_z` uses `scores(z, d, {"alpha":1.0}) > 0.0`.
    # This implies alpha=1.0 for this specific conversion, or it's passed.
    
    # Get scores for each particle
    # `scores` function handles batching if z_particles is [N, D, K, 2]
    s = scores(z_particles, alpha_hparam=alpha_hparam_for_scores) # [N, D, D]
    
    hard_gmats = (s > 0.0).to(torch.float32) # Convert boolean to float (0.0 or 1.0)
    return hard_gmats


class Logistic(Distribution):
    """
    Logistic distribution implementation for PyTorch.
    """
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, torch.Tensor):
            batch_shape = self.loc.size()
        else:
            batch_shape = torch.Size()
        super(Logistic, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * (torch.log(u) - torch.log(1 - u))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return -z - 2 * torch.log(1 + torch.exp(-z)) - torch.log(self.scale)

