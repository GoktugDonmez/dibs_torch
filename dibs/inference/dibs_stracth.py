"""
    MAIN STRACTH FOR IMPLEMENTING DIBS
    START FROM THE CORE FUNCTIONS
    
    FIRST TRY TO IMPLEMENT THE JOINT PROB COMPONENTES (Z, G|Z, THETA|G, D|Z, THETA)
    THEN ITS GRADIENTS
    PRIOR GRAPH


    AFTER IMPLEMENT THE SVGD INFERENCE
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.func import grad, vmap

# TODO later the packages
#from dibs.utils import zero_diagonal  # Import from package
## TODO LOGSUMEXP
def zero_diagonal(g: torch.Tensor) -> torch.Tensor:
    """
    Returns the argument matrix with its diagonal set to zero.

    Args:
        g (torch.Tensor): matrix of shape [..., d, d]
    """
    
    d = g.shape[-1]
    diag_mask = ~torch.eye(d, dtype=torch.bool, device=g.device)
    
    return g * diag_mask


class DiBS(nn.Module):
    ## TODO LATER
    def __init__(self):
        super(DiBS, self).__init__()
        self.tau = torch.tensor(1.0) ## TODO related to gumbel-softmax
        self.alpha = lambda t: torch.tensor(1.0) ## TODO related to gumbel-softmax

        # TODO: Initialize self.x (data), self.interv_mask (intervention mask),
        #       and define or assign self.log_joint_prob method.
        # Example placeholders (replace with actual data/logic):
        # self.x = torch.randn(100, 10)  # Example data: 100 samples, 10 features
        # self.interv_mask = torch.zeros(10, dtype=torch.bool) # Example mask
        # self.log_joint_prob = self._my_log_joint_prob_implementation
        self.grad_estimator_z = 'reparam' # Default, or set as needed. TODO: User updated this to be a comment.

        # TODO: Set values for n_grad_mc_samples and score_function_baseline
        self.n_grad_mc_samples = 16 # Example value, set as needed
        self.score_function_baseline = 0.5 # Example value, set as needed (e.g. 0.0 to disable, >0 to enable)

    def particle_to_g_lim(self, z):
        """
        Returns G corresponding to alpha = infinity for particles z

        Args:
            z: latent variables [..., d, k, 2]

        Returns:
            graph adjacency matrices of shape [..., d, d]
        """
        # Split z into u,v components along last dimension
        u, v = z[..., 0], z[..., 1]
        
        # Compute inner products between u,v vectors
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        
        # Convert to binary adjacency matrix
        g_samples = (scores > 0).to(torch.int32)
        
        
        return zero_diagonal(g_samples)
    
    def sample_g(self, p_probs: torch.Tensor, n_samples) -> torch.Tensor:
        """
        Sample Bernoulli matrix according to matrix of probabilities

        Args:
            p_probs (torch.Tensor): matrix of probabilities [d, d]
            n_samples (int): number of samples

        Returns:
            torch.Tensor: an array of matrices sampled according to p_probs of shape [n_samples, d, d]
        """
        d = p_probs.shape[-1]
        # Sample from Bernoulli for each entry
        g_samples = torch.bernoulli(p_probs.expand(n_samples, -1, -1))
        g_samples = g_samples.to(torch.int32)
        # Mask diagonal
        return zero_diagonal(g_samples)



    def particle_to_soft_graph(self, z: torch.Tensor, uniform_noise: torch.Tensor, t: int) -> torch.Tensor:
        """
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples ``eps``

        Args:
            z (torch.Tensor): a single latent tensor :math:`Z` of shape ``[d, k, 2]``
            eps (torch.Tensor): random i.i.d. Logistic(0,1) noise  of shape ``[d, d]``
            t (int): step

        Returns:
            torch.Tensor: Gumbel-softmax sample of adjacency matrix [d, d]
        """
        # z has shape [d, k, 2], so z[..., 0] and z[..., 1] have shape [d, k]
        # scores will have shape [d, d]
        scores = torch.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        # self.alpha(t) is expected to return a scalar or a tensor compatible with broadcasting
        # self.tau is expected to be a scalar tensor or float
        soft_graph = torch.sigmoid(self.tau * (uniform_noise + self.alpha(t) * scores))

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(soft_graph)


    def particle_to_hard_graph(self, z: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        """
        Bernoulli sample of :math:`G` using probabilities implied by latent ``z``

        Args:
            z (torch.Tensor): a single latent tensor :math:`Z` of shape ``[d, k, 2]``
            eps (torch.Tensor): random i.i.d. Logistic(0,1) noise  of shape ``[d, d]``
            t (int): step

        Returns:
            torch.Tensor: Gumbel-max (hard) sample of adjacency matrix ``[d, d]``
        """
        # z has shape [d, k, 2], scores will have shape [d, d]
        scores = torch.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # simply take hard limit of sigmoid in gumbel-softmax/concrete distribution
        hard_graph = ((eps + self.alpha(t) * scores) > 0.0).to(z.dtype)

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(hard_graph)

    def edge_probs(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """
        Edge probabilities encoded by latent representation

        Args:
            z (torch.Tensor): latent tensors :math:`Z`  ``[..., d, k, 2]``
            t (int): step

        Returns:
            torch.Tensor: edge probabilities of shape ``[..., d, d]``
        """
        u, v = z[..., 0], z[..., 1]
        # z can have batch dims [..., d, k, 2], so u, v are [..., d, k]
        # scores will have shape [..., d, d]
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        probs = torch.sigmoid(self.alpha(t) * scores)

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(probs)

    def edge_log_probs(self, z: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Edge log probabilities encoded by latent representation

        Args:
            z (torch.Tensor): latent tensors :math:`Z` ``[..., d, k, 2]``
            t (int): step

        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        u, v = z[..., 0], z[..., 1]
        # z can have batch dims [..., d, k, 2], so u, v are [..., d, k]
        # scores will have shape [..., d, d]
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        
        # log_sigmoid(x) = -softplus(-x)
        # log(1-sigmoid(x)) = log_sigmoid(-x) = -softplus(x)
        
        # Using torch.nn.functional.logsigmoid for numerical stability
        log_probs = torch.nn.functional.logsigmoid(self.alpha(t) * scores)
        log_probs_neg = torch.nn.functional.logsigmoid(self.alpha(t) * -scores)

        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        return zero_diagonal(log_probs), zero_diagonal(log_probs_neg)

    def latent_log_prob(self, single_g: torch.Tensor, single_z: torch.Tensor, t: int) -> torch.Tensor:
        """
        Log likelihood of generative graph model

        Args:
            single_g (torch.Tensor): single graph adjacency matrix ``[d, d]``
            single_z (torch.Tensor): single latent tensor ``[d, k, 2]``
            t (int): step

        Returns:
            torch.Tensor: log likelihood :math:`log p(G | Z)` of shape ``[1,]``
        """
        # [d, d], [d, d]
        log_p, log_1_p = self.edge_log_probs(single_z, t)

        # [d, d]
        # Ensure single_g is a tensor for broadcasting and operations
        
        log_prob_g_ij = single_g * log_p + (1.0 - single_g) * log_1_p

        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = torch.sum(log_prob_g_ij)

        return log_prob_g

    def eltwise_grad_latent_log_prob(self, gs: torch.Tensor, single_z: torch.Tensor, t: int) -> torch.Tensor:
        """
        Gradient of log likelihood of generative graph model w.r.t. :math:`Z`
        i.e. :math:`\nabla_Z \log p(G | Z)`
        Batched over samples of :math:`G` given a single :math:`Z`.

        Args:
            gs (torch.Tensor): batch of graph matrices ``[n_graphs, d, d]``
            single_z (torch.Tensor): latent variable ``[d, k, 2]``
            t (int): step

        Returns:
            torch.Tensor: batch of gradients of shape ``[n_graphs, d, k, 2]``
        """
        # Ensure single_z requires grad for the gradient computation
        # Create a new tensor that requires gradients if single_z doesn't already.
        # This is important because the original single_z might come from a part of the graph
        # that doesn't require gradients, or it might be a non-leaf tensor.
        single_z_clone = single_z.clone().requires_grad_(True)

        # Define a function to compute gradients for a single g
        # grad_fn will compute the gradient of self.latent_log_prob w.r.t. its second argument (single_z)
        grad_fn = grad(self.latent_log_prob, argnums=1)
        
        # Use vmap to apply grad_fn over the batch of graphs gs
        # in_dims=(0, None, None) specifies that gs is batched (0), single_z_clone is not (None), and t is not (None)
        # out_dims=0 specifies that the output should be batched along the first dimension
        # Note: latent_log_prob expects single_z as the second argument, so argnums=1 in grad() and the order in vmap matters.
        # We are passing single_z_clone to vmap, which will be used by grad_fn.
        batched_grads = vmap(grad_fn, in_dims=(0, None, None), out_dims=0)(gs, single_z_clone, t)
        
        return batched_grads

    def eltwise_log_joint_prob(self, gs: torch.Tensor, single_theta: torch.Tensor) -> torch.Tensor:
        """
        Joint likelihood :math:`\log p(\Theta, D | G)` batched over samples of :math:`G`

        Args:
            gs (torch.Tensor): batch of graphs ``[n_graphs, d, d]``

        Returns:
            torch.Tensor: batch of logprobs of shape ``[n_graphs, ]``
        """
        # Assuming self.x and self.interv_mask are initialized in __init__
        # The vmap signature maps over gs (0-th dim), while single_theta, self.x, self.interv_mask, an.
        return vmap(self.log_joint_prob, in_dims=(0, None, None, None), out_dims=0)(gs, single_theta, self.x, self.interv_mask)

    def log_join_prob_soft(self, single_z: torch.Tensor, single_theta: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:

        """
         This is the composition of :math:`\\log p(\\Theta, D | G) `
        and :math:`G(Z, U)`  (Gumbel-softmax graph sample given :math:`Z`)

        
        return 
            logprob shape [1,]
        
        """

        soft_g_sample = self.particle_to_soft_graph(single_z, eps, t)
        # Assuming self.log_joint_prob is defined and expects (graph, theta, x, interv_mask)
        # TODO: Confirm if self.x and self.interv_mask should be accessed here or passed.
        # For now, assuming they are attributes of self, similar to eltwise_log_joint_prob
        return self.log_joint_prob(soft_g_sample, single_theta, self.x, self.interv_mask)

    def grad_z_likelihood_score_function(self, single_z: torch.Tensor, single_theta: torch.Tensor, single_sf_baseline: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score function estimator (aka REINFORCE) for the score :math:`\\nabla_Z \\log p(\\Theta, D | Z)`
        Uses the same :math:`G \\sim p(G | Z)` samples for expectations in numerator and denominator.

        This does not use :math:`\\nabla_G \\log p(\\Theta, D | G)` and is hence applicable when
        the gradient w.r.t. the adjacency matrix is not defined (as e.g. for the BGe score).

        Args:
            single_z (torch.Tensor): single latent tensor ``[d, k, 2]``
            single_theta (Any): single parameter PyTree (adjust type as needed for PyTorch)
            single_sf_baseline (torch.Tensor): ``[1, ]`` or scalar
            t (int): step

        Returns:
            tuple of gradient, baseline  ``[d, k, 2], [1, ]`` (or scalar baseline)
        """

        # [d, d]
        p = self.edge_probs(single_z, t)
        n_vars, k_dim = single_z.shape[0:2] # Corrected n_dim to k_dim for clarity

        # [n_grad_mc_samples, d, d]
        # PyTorch's sample_g doesn't use explicit keys like JAX
        g_samples = self.sample_g(p, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        # n_mc_numerator = self.n_grad_mc_samples (implicit)
        # n_mc_denominator = self.n_grad_mc_samples (implicit)

        # [n_grad_mc_samples, ]
        # PyTorch's eltwise_log_joint_prob also doesn't use explicit keys
        # Ensure eltwise_log_joint_prob is adapted for single_theta if it expects batched theta
        # For now, assume single_theta is correctly handled or passed as is.
        # If eltwise_log_joint_prob expects a batch of thetas, this needs adjustment.
        # Based on its signature eltwise_log_joint_prob(self, gs, single_theta), this should be fine.
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta)
        logprobs_denominator = logprobs_numerator # shape [n_grad_mc_samples]

        # variance_reduction
        # Ensure single_sf_baseline is a scalar or correctly broadcastable
        if self.score_function_baseline > 0.0:
            logprobs_numerator_adjusted = logprobs_numerator - single_sf_baseline.squeeze()
        else:
            logprobs_numerator_adjusted = logprobs_numerator
        
        # [n_grad_mc_samples, d, k, 2]
        # grad_latent_log_prob returns grads of shape [n_graphs, d, k, 2]
        grad_z_batched = self.eltwise_grad_latent_log_prob(g_samples, single_z, t)
        
        # [n_grad_mc_samples, d * k * 2]
        grad_z_flat = grad_z_batched.reshape(self.n_grad_mc_samples, -1)
        
        # Transpose to [d * k * 2, n_grad_mc_samples]
        grad_z = grad_z_flat.transpose(0, 1)

        # stable computation of exp/log/divide for log(sum(b * exp(a))) with sign
        # a is logprobs_numerator_adjusted (shape [N])
        # b is grad_z (shape [M, N]) where N = n_grad_mc_samples, M = d*k*2
        
        # terms = b * exp(a)
        # logprobs_numerator_adjusted needs to be [1, N] to broadcast with grad_z [M, N]
        terms = grad_z * torch.exp(logprobs_numerator_adjusted.unsqueeze(0))
        
        sum_terms = torch.sum(terms, dim=1) # Sum over N (axis=1), result shape [M]
        
        # Handle cases where sum_terms is zero to avoid log(0)
        # A small epsilon can be added, or handle sign separately.
        # JAX's logsumexp with return_sign handles this robustly.
        # For PyTorch:
        log_abs_sum_terms = torch.log(torch.abs(sum_terms) + 1e-38) # Add epsilon for stability
        sign_sum_terms = torch.sign(sum_terms)
        
        log_numerator = log_abs_sum_terms
        sign = sign_sum_terms

        # log_denominator: scalar
        log_denominator = torch.logsumexp(logprobs_denominator, dim=0) # Sum over n_grad_mc_samples

        # If n_mc_numerator == n_mc_denominator, the log(n_mc) terms cancel.
        # stable_sf_grad = sign * torch.exp(log_numerator - log_denominator)
        # Original was: sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))
        # This simplifies to sign * exp(log_numerator - log_denominator) if n_mc_numerator == n_mc_denominator
        # which they are (self.n_grad_mc_samples)

        stable_sf_grad_flat = sign * torch.exp(log_numerator - log_denominator)

        # [d, k, 2]
        stable_sf_grad_shaped = stable_sf_grad_flat.reshape(n_vars, k_dim, 2)

        # update baseline
        # Ensure single_sf_baseline is a scalar for this calculation
        if self.score_function_baseline > 0.0: # only update if baseline is active
            new_baseline = (self.score_function_baseline * logprobs_numerator.mean() +
                                (1.0 - self.score_function_baseline) * single_sf_baseline.squeeze())
        else:
            new_baseline = single_sf_baseline.squeeze()


        return stable_sf_grad_shaped, new_baseline.view(1) # Ensure baseline is [1,]

    def grad_z_likelihood_gumbel(self, single_z: torch.Tensor, single_theta: torch.Tensor, baseline: torch.Tensor, t: int): # Matches vmap
        """
        Placeholder for Gumbel reparameterization estimator for nabla_Z log p(Theta, D | Z).
        Signature matches the vmap in eltwise_grad_z_likelihood.
        """
        # Expected to return a tuple: (gradient_estimate, baseline_output_or_metric)
        # gradient_estimate: [d, k, 2]
        # baseline_output_or_metric: scalar or tensor that vmap can handle for the second output dim
        raise NotImplementedError("grad_z_likelihood_gumbel needs to be implemented.")

    def eltwise_grad_z_likelihood(self, zs: torch.Tensor, thetas: torch.Tensor, baselines: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes batch of estimators for score :math:`\\nabla_Z \\log p(\\Theta, D | Z)`
        Selects corresponding estimator used for the term :math:`\\nabla_Z E_{p(G|Z)}[ p(\\Theta, D | G) ]`
        and executes it in batch.

        Args:
            zs (torch.Tensor): batch of latent tensors :math:`Z` ``[n_particles, d, k, 2]``
            thetas (Any): batch of parameters PyTree with ``n_particles`` as leading dim (adjust type as needed)
            baselines (torch.Tensor): array of score function baseline values of shape ``[n_particles, ]``
            t (int): step
            subkeys (torch.Tensor): batch of JAX-like PRNG keys, or seeds/noise for PyTorch ``[n_particles, ...]``

        Returns:
            tuple batch of (gradient estimates, baselines_outputs) of shapes ``[n_particles, d, k, 2], [n_particles, ...]``
        """

        # select the chosen gradient estimator
        if self.grad_estimator_z == 'score':
            grad_z_likelihood_fn = self.grad_z_likelihood_score_function
        elif self.grad_estimator_z == 'reparam':
            grad_z_likelihood_fn = self.grad_z_likelihood_gumbel
        else:
            raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')

        # vmap the selected function
        # (0, 0, 0, None, 0) -> input dimensions for (zs, thetas, baselines, t, subkeys)
        # (0, 0) -> output dimensions for the tuple returned by grad_z_likelihood_fn
        return vmap(grad_z_likelihood_fn, in_dims=(0, 0, 0, None), out_dims=(0, 0))(zs, thetas, baselines, t)

    ## estimator for score d/dz log p(theta,D | Z)



    # --- Test Methods ---
    # Note: These test methods assume that the DiBS instance (`self`)
    # has `tau` attribute and `alpha` method (e.g., `self.alpha = lambda t: torch.tensor(1.0)`)
    # initialized appropriately, for example in the `__init__` method of the DiBS class.

    def test_sample_g(self):
        print("\nRunning test_sample_g...")
        d = 3
        n_samples = 5
        p_probs = torch.rand(d, d) # Probabilities between 0 and 1

        g_samples = self.sample_g(p_probs, n_samples)

        assert g_samples.shape == (n_samples, d, d), f"Expected shape ({n_samples}, {d}, {d}), got {g_samples.shape}"
        assert g_samples.dtype == torch.int32, f"Expected dtype torch.int32, got {g_samples.dtype}"
        assert torch.all((g_samples == 0) | (g_samples == 1)), "Values are not binary"
        for i in range(n_samples):
            assert torch.all(torch.diag(g_samples[i]) == 0), f"Diagonal not zero for sample {i}"
        print(f"test_sample_g: Output shape {g_samples.shape}. All checks passed.")

    def test_particle_to_soft_graph(self):
        print("\nRunning test_particle_to_soft_graph...")
        d = 3
        k = 2
        t_val = 1 # Example time step
        
        # Ensure self.tau and self.alpha are available, e.g. by initializing them in DiBS.__init__
        # For this test, we'll assume they exist. Example:
        # self.tau = torch.tensor(1.0)
        # self.alpha = lambda t: torch.tensor(1.0)

        z = torch.randn(d, k, 2)
        # Sample Logistic(0,1) noise using inverse sigmoid (logit) trick
        uniform_noise = torch.rand(d, d)
        eps_logistic = torch.log(uniform_noise) - torch.log(1 - uniform_noise)

        soft_graph = self.particle_to_soft_graph(z, eps_logistic, t_val)

        assert soft_graph.shape == (d, d), f"Expected shape ({d}, {d}), got {soft_graph.shape}"
        assert torch.all(soft_graph >= 0) and torch.all(soft_graph <= 1), "Values not in [0, 1]"
        assert torch.all(torch.diag(soft_graph) == 0), "Diagonal not zero"
        print(f"test_particle_to_soft_graph: Output shape {soft_graph.shape}. All checks passed.")

    def test_particle_to_hard_graph(self):
        print("\nRunning test_particle_to_hard_graph...")
        d = 3
        k = 2
        t_val = 1 # Example time step
        
        # Ensure self.alpha is available. Example:
        # self.alpha = lambda t: torch.tensor(1.0)

        z = torch.randn(d, k, 2, dtype=torch.float32) # Specify dtype
        # Sample Logistic(0,1) noise using inverse sigmoid (logit) trick
        uniform_noise = torch.rand(d, d)
        eps_logistic = torch.log(uniform_noise) - torch.log(1 - uniform_noise)

        hard_graph = self.particle_to_hard_graph(z, eps_logistic, t_val)

        assert hard_graph.shape == (d, d), f"Expected shape ({d}, {d}), got {hard_graph.shape}"
        assert hard_graph.dtype == z.dtype, f"Expected dtype {z.dtype}, got {hard_graph.dtype}"
        assert torch.all((hard_graph == 0) | (hard_graph == 1)), "Values are not binary"
        assert torch.all(torch.diag(hard_graph) == 0), "Diagonal not zero"
        print(f"test_particle_to_hard_graph: Output shape {hard_graph.shape}. All checks passed.")

    def test_edge_probs(self):
        print("\nRunning test_edge_probs...")
        d = 3
        k = 2
        t_val = 1 # Example time step

        # Ensure self.alpha is available. Example:
        # self.alpha = lambda t: torch.tensor(1.0)

        # Test with unbatched z
        z_unbatched = torch.randn(d, k, 2)
        probs_unbatched = self.edge_probs(z_unbatched, t_val)
        
        assert probs_unbatched.shape == (d, d), f"Unbatched: Expected shape ({d}, {d}), got {probs_unbatched.shape}"
        assert torch.all(probs_unbatched >= 0) and torch.all(probs_unbatched <= 1), "Unbatched: Values not in [0, 1]"
        assert torch.all(torch.diag(probs_unbatched) == 0), "Unbatched: Diagonal not zero"
        print(f"test_edge_probs (unbatched): Output shape {probs_unbatched.shape}. Checks passed.")

        # Test with batched z
        batch_size = 4
        z_batched = torch.randn(batch_size, d, k, 2)
        probs_batched = self.edge_probs(z_batched, t_val)

        assert probs_batched.shape == (batch_size, d, d), f"Batched: Expected shape ({batch_size}, {d}, {d}), got {probs_batched.shape}"
        assert torch.all(probs_batched >= 0) and torch.all(probs_batched <= 1), "Batched: Values not in [0, 1]"
        for i in range(batch_size):
            assert torch.all(torch.diag(probs_batched[i]) == 0), f"Batched: Diagonal not zero for sample {i}"
        print(f"test_edge_probs (batched): Output shape {probs_batched.shape}. All checks passed.")


##test
s = DiBS()
#s.test_sample_g()
s.test_particle_to_soft_graph()
#s.test_particle_to_hard_graph()