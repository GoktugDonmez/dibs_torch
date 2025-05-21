import functools
import numpy as onp # Original NumPy, used for onp.unique in JAX
import torch
import math
from typing import Callable, Any, Tuple, List, Dict, Optional, Union
from .dibs import DiBS

# Assuming DiBSTorch and its helpers (zero_diagonal, acyclic_constr_nograd_pytorch etc.)
# are defined as in the previous step. For this snippet, imagine they are available.
# from .dibs_torch import DiBSTorch 
# from .torch_helpers import ... (logsumexp_pytorch_for_dibs, etc.)

# --- Placeholder for ParticleDistribution (can be a simple dataclass or named tuple) ---
from dataclasses import dataclass

@dataclass
class ParticleDistributionTorch:
    logp: torch.Tensor
    g: torch.Tensor
    theta: Optional[Any] = None # For JointDiBS (not used by Marginal)

# --- Optimizer Implementations (JAX-like API for PyTorch) ---
class TorchOpt:
    """Base class for PyTorch optimizers with a JAX-like functional API."""
    def __init__(self, step_size: float):
        self.step_size = step_size

    def init(self, params: Any) -> Any:
        return params  # For SGD, state is just params

    def get_params(self, opt_state: Any) -> Any:
        return opt_state

    def update(self, step_idx: int, grads: Any, opt_state: Any) -> Any:
        # `grads` is expected to be the SVGD update vector phi (pre-negated for descent)
        # params_new = params_current - step_size * grads (if grads is like dL/dx)
        # Or, params_new = params_current + step_size * phi_ascent_direction
        # The JAX code's _z_update returns - (svgd_direction), so 'grads' here is for minimization.
        
        # This method will be overridden by specific optimizers
        if isinstance(opt_state, torch.Tensor):
            return opt_state + self.step_size * grads # Assuming grads is the pre-negated SVGD update
        # PyTree handling for more complex states would go here or in subclasses
        raise NotImplementedError("Base TorchOpt update called.")

class TorchSGD(TorchOpt):
    def update(self, step_idx: int, grads: Any, opt_state: Any) -> Any:
        # opt_state is the current parameters (e.g., z_particles)
        # grads is the SVGD update phi (already negated for descent)
        if isinstance(opt_state, torch.Tensor):
            return opt_state + self.step_size * grads # JAX DiBS _z_update returns negated phi.
        else: # Basic PyTree (list/tuple of tensors)
            return self._tree_map(lambda p, g: p + self.step_size * g, opt_state, grads)
    
    def _tree_map(self, func, tree1, tree2): # Simplified tree_map for SGD
        if isinstance(tree1, torch.Tensor): return func(tree1, tree2)
        if isinstance(tree1, list): return [self._tree_map(func, t1, t2) for t1, t2 in zip(tree1, tree2)]
        if isinstance(tree1, tuple): return tuple(self._tree_map(func, t1, t2) for t1, t2 in zip(tree1, tree2))
        if isinstance(tree1, dict): return {k: self._tree_map(func, tree1[k], tree2[k]) for k in tree1}
        return tree1


class TorchRMSprop(TorchOpt):
    def __init__(self, step_size: float, alpha: float = 0.9, eps: float = 1e-8):
        super().__init__(step_size)
        self.alpha = alpha
        self.eps = eps

    def init(self, params: Any) -> Tuple[Any, Any]: # Returns (params, avg_sq_grad)
        avg_sq_grad = self._tree_map(lambda p: torch.zeros_like(p), params)
        return (params, avg_sq_grad)

    def get_params(self, opt_state: Tuple[Any, Any]) -> Any:
        return opt_state[0]

    def update(self, step_idx: int, grads: Any, opt_state: Tuple[Any, Any]) -> Tuple[Any, Any]:
        params, avg_sq_grad = opt_state

        def _update_leaf(p, g, avg_sq):
            new_avg_sq = self.alpha * avg_sq + (1.0 - self.alpha) * torch.square(g)
            # Grads (phi) already negated for descent. So, p = p + step_size * (-g_eff) = p - step_size * g_eff
            # JAX RMSprop applies: x - step_size * g / sqrt(avg_sq_g + eps)
            # If grads (phi) is already -svgd_update, then x_new = x - step_size * (-svgd_update / ...)
            # So, x_new = x + step_size * (svgd_update / ...)
            # This matches the JAX DiBS _z_update which returns negated phi.
            p_new = p + self.step_size * grads / (torch.sqrt(new_avg_sq) + self.eps) # Mistake: grads is a pytree, not g
            # Corrected logic for leaf:
            # p_new = p + self.step_size * g / (torch.sqrt(new_avg_sq) + self.eps)
            return p + self.step_size * g / (torch.sqrt(new_avg_sq) + self.eps), new_avg_sq

        # Assuming _tree_map_multi applies func to corresponding leaves of multiple trees
        new_params, new_avg_sq_grad = self._tree_map_multi(_update_leaf, params, grads, avg_sq_grad)
        return (new_params, new_avg_sq_grad)

    # Basic PyTree helpers for RMSprop state
    def _tree_map(self, func, tree):
        if isinstance(tree, torch.Tensor): return func(tree)
        if isinstance(tree, list): return [self._tree_map(func, item) for item in tree]
        if isinstance(tree, tuple): return tuple(self._tree_map(func, item) for item in tree)
        if isinstance(tree, dict): return {k: self._tree_map(func, v) for k, v in tree.items()}
        return tree
    
    def _tree_map_multi(self, func, *trees):
        first_tree = trees[0]
        if isinstance(first_tree, torch.Tensor): return func(*trees)
        if isinstance(first_tree, list): return [self._tree_map_multi(func, *[t[i] for t in trees]) for i in range(len(first_tree))]
        if isinstance(first_tree, tuple): return tuple(self._tree_map_multi(func, *[t[i] for t in trees]) for i in range(len(first_tree)))
        if isinstance(first_tree, dict): return {k: self._tree_map_multi(func, *[t[k] for t in trees]) for k in first_tree}
        return first_tree

# --- MarginalDiBSTorch Class ---
class MarginalDiBS(DiBS): # DiBSTorch from previous response
    def __init__(self, *,
                 x: torch.Tensor,
                 graph_model: Any, # PyTorch equivalent of ErdosReniDAGDistributionTorch
                 likelihood_model: Any, # PyTorch equivalent of BGeTorch
                 interv_mask: Optional[torch.Tensor] = None,
                 kernel_cls: Callable, # E.g., AdditiveFrobeniusSEKernelTorch (needs to be defined)
                 kernel_param: Optional[Dict] = None,
                 optimizer_name: str = "rmsprop",
                 optimizer_param: Optional[Dict] = None,
                 alpha_linear: float = 1.0,
                 beta_linear: float = 1.0,
                 tau: float = 1.0,
                 n_grad_mc_samples: int = 128,
                 n_acyclicity_mc_samples: int = 32,
                 grad_estimator_z: str = "score", # Default for MarginalDiBS in JAX
                 score_function_baseline_factor: float = 0.0,
                 latent_prior_std: Optional[float] = None,
                 k_latent_dim: Optional[int] = None,
                 device: str = 'cpu',
                 verbose: bool = False):

        # Default mutable args
        if kernel_param is None: kernel_param = {"h": 5.0} # Example for AdditiveFrobeniusSEKernel
        if optimizer_param is None: optimizer_param = {"step_size": 0.005}
        
        effective_device = torch.device(device)
        if interv_mask is None:
            interv_mask = torch.zeros_like(x, dtype=torch.bool, device=effective_device) # Use bool for masks
        else:
            interv_mask = interv_mask.to(dtype=torch.bool, device=effective_device)

        super().__init__(
            x=x, interv_mask=interv_mask,
            log_graph_prior=graph_model.unnormalized_log_prob_soft,
            # In MarginalDiBS, log_joint_prob is likelihood_model.interventional_log_marginal_prob
            # This takes (g, _theta, x, interv_mask, rng). _theta is None.
            log_joint_prob=likelihood_model.interventional_log_marginal_prob,
            alpha_linear=alpha_linear, beta_linear=beta_linear, tau=tau,
            n_grad_mc_samples=n_grad_mc_samples, n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z, score_function_baseline_factor=score_function_baseline_factor,
            latent_prior_std=latent_prior_std, k_latent_dim=k_latent_dim,
            device=device, verbose=verbose
        )

        self.likelihood_model = likelihood_model
        self.graph_model = graph_model
        self.kernel = kernel_cls(**kernel_param, device=self.device) # Pass device to kernel if it uses tensors

        # Optimizer setup (manual, JAX-like API)
        opt_step_size = optimizer_param.get('step_size', 0.005)
        if optimizer_name.lower() == 'sgd':
            self.opt_manual_impl = TorchSGD(step_size=opt_step_size)
        elif optimizer_name.lower() == 'rmsprop':
            rms_decay = optimizer_param.get('decay', 0.9)
            rms_eps = optimizer_param.get('eps', 1e-8)
            self.opt_manual_impl = TorchRMSprop(step_size=opt_step_size, alpha=rms_decay, eps=rms_eps)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # JAX-like optimizer functions
        self.opt_init = self.opt_manual_impl.init
        self.opt_update = self.opt_manual_impl.update
        self.get_params = self.opt_manual_impl.get_params # Jit in JAX, not here unless whole class is scripted

    def _sample_initial_random_particles(self, *, generator: Optional[torch.Generator], 
                                         n_particles: int, n_dim: Optional[int] = None) -> torch.Tensor:
        effective_n_dim = n_dim if n_dim is not None else self.n_vars
        if effective_n_dim <= 0: effective_n_dim = 1 # Avoid sqrt(0) or empty dim if n_vars is 0

        # Use self.latent_prior_std_val which is set in DiBSTorch.__init__
        std_z = self.latent_prior_std_val
        if std_z is None or std_z <=0: # Fallback if not properly set from k_latent_dim
            std_z = 1.0 / math.sqrt(effective_n_dim) if effective_n_dim > 0 else 1.0
            if self.verbose: print(f"MarginalDiBS using fallback std_z: {std_z} for Z init.")

        shape_z = (n_particles, self.n_vars, effective_n_dim, 2)
        rand_fn = torch.randn if generator is None else lambda *s, **kw: torch.normal(0.,1.,size=s,generator=kw.get('generator'),device=kw.get('device'),dtype=kw.get('dtype'))
        z = rand_fn(shape_z, generator=generator, device=self.device, dtype=torch.float32) * std_z
        return z

    def _f_kernel(self, x_latent: torch.Tensor, y_latent: torch.Tensor) -> torch.Tensor:
        return self.kernel.eval(x=x_latent, y=y_latent) # kernel.eval should take two [d,k,2] tensors

    def _f_kernel_mat(self, x_latents: torch.Tensor, y_latents: torch.Tensor) -> torch.Tensor:
        A, B = x_latents.shape[0], y_latents.shape[0]
        kernel_matrix = torch.empty((A, B), device=self.device, dtype=x_latents.dtype)
        for i in range(A):
            for j in range(B):
                kernel_matrix[i, j] = self._f_kernel(x_latents[i], y_latents[j])
        return kernel_matrix

    def _eltwise_grad_kernel_z(self, x_latents_batch: torch.Tensor, y_latent_single: torch.Tensor) -> torch.Tensor:
        grads_list = []
        for i in range(x_latents_batch.shape[0]):
            x_l_s = x_latents_batch[i].detach().requires_grad_(True)
            kernel_val = self._f_kernel(x_l_s, y_latent_single)
            grad_x_l = torch.autograd.grad(kernel_val, x_l_s, allow_unused=True)[0]
            grads_list.append(grad_x_l if grad_x_l is not None else torch.zeros_like(x_l_s))
            x_l_s.requires_grad_(False)
        return torch.stack(grads_list)

    def _z_update_single_particle(self, current_particle_z: torch.Tensor, 
                                  all_particles_z: torch.Tensor, 
                                  kernel_row_for_current_z: torch.Tensor, # k(current_z, other_z_j) for all j
                                  grad_log_prob_all_other_z: torch.Tensor) -> torch.Tensor:
        # current_particle_z: [d,k,2] (z_i)
        # all_particles_z:    [N,d,k,2] (all z_j)
        # kernel_row_for_current_z: [N] (k(z_i, z_j) for all j)
        # grad_log_prob_all_other_z: [N,d,k,2] (grad_log_p(z_j) for all j)

        # Term 1: sum_j k(z_i, z_j) * grad_log_p(z_j)
        term1 = torch.sum(kernel_row_for_current_z.view(-1,1,1,1) * grad_log_prob_all_other_z, dim=0)

        # Term 2: sum_j grad_zj k(z_j, z_i)
        # _eltwise_grad_kernel_z(all_particles_z, current_particle_z) gives [N,d,k,2] 
        # where element [j,:,:,:] is grad_zj k(z_j, z_i)
        term2_all_grads = self._eltwise_grad_kernel_z(all_particles_z, current_particle_z)
        term2 = torch.sum(term2_all_grads, dim=0)
        
        svgd_phi_for_current_z = (term1 + term2) / all_particles_z.shape[0]
        return -svgd_phi_for_current_z # Return negated for descent-based optimizer

    def _parallel_update_z(self, all_zs: torch.Tensor, kxx_full: torch.Tensor, grad_log_p_zs: torch.Tensor) -> torch.Tensor:
        # all_zs: [N,d,k,2], kxx_full: [N,N], grad_log_p_zs: [N,d,k,2] -> Output [N,d,k,2]
        phi_list = [self._z_update_single_particle(all_zs[i], all_zs, kxx_full[i,:], grad_log_p_zs) for i in range(all_zs.shape[0])]
        return torch.stack(phi_list)

    def _svgd_step(self, t: Union[int, float], opt_state_z: Any, 
                   current_iter_generators: List[Optional[torch.Generator]], 
                   sf_baseline_batch: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        
        z_particles = self.get_params(opt_state_z)
        
        # dz_log_likelihood for p(D|Z) where theta is None for marginal
        dz_log_likelihood, sf_baseline_updated = self.eltwise_grad_z_likelihood(
            z_particles, None, sf_baseline_batch, t, current_iter_generators)
        
        dz_log_prior = self.eltwise_grad_latent_prior(z_particles, current_iter_generators, t)
        dz_log_target = dz_log_prior + dz_log_likelihood

        k_matrix = self._f_kernel_mat(z_particles, z_particles)
        
        # phi_z is the SVGD update vector (already negated for descent)
        phi_z_batch = self._parallel_update_z(z_particles, k_matrix, dz_log_target)
        
        new_opt_state_z = self.opt_update(t, phi_z_batch, opt_state_z)
        return new_opt_state_z, sf_baseline_updated


    def _svgd_loop_python(self, start_iter: int, n_steps_in_loop: int, init_args: Tuple, 
                          base_generator: torch.Generator) -> Tuple[Any, torch.Tensor]:
        opt_state_z, sf_baseline = init_args
        
        for i in range(start_iter, start_iter + n_steps_in_loop):
            n_particles = self.get_params(opt_state_z).shape[0]
            step_generators = []
            for _ in range(n_particles): # One generator per particle for this SVGD step
                 seed_val = torch.empty((), dtype=torch.int64, device=self.device).random_(generator=base_generator).item()
                 g = torch.Generator(device=self.device); g.manual_seed(seed_val)
                 step_generators.append(g)
            
            opt_state_z, sf_baseline = self._svgd_step(i, opt_state_z, step_generators, sf_baseline)
        return opt_state_z, sf_baseline

    def sample(self, *, main_generator: torch.Generator, n_particles: int, steps: int, 
               n_dim_particles: Optional[int] = None, callback: Optional[Callable] = None, 
               callback_every: Optional[int] = None) -> torch.Tensor:
        # Determine k_latent_dim for DiBSTorch's latent_prior_std default logic
        # This k is used in _sample_initial_random_particles and can influence latent_prior_std_val
        if self.k_latent_dim is None : # If not set during __init__
            self.k_latent_dim = n_dim_particles if n_dim_particles is not None else self.n_vars
            if self.latent_prior_std_val is None and self.k_latent_dim > 0 : # Update std if it was pending k
                self.latent_prior_std_val = 1.0 / math.sqrt(self.k_latent_dim)
                if self.verbose: print(f"MarginalDiBS sample() updated latent_prior_std to {self.latent_prior_std_val} based on k_dim={self.k_latent_dim}")


        init_z = self._sample_initial_random_particles(generator=main_generator, n_particles=n_particles, n_dim=n_dim_particles)
        sf_baseline_state = torch.zeros(n_particles, device=self.device)
        opt_state_z = self.opt_init(init_z)

        effective_callback_every = callback_every if callback_every is not None else steps
        if steps == 0: effective_callback_every = 0

        for t_loop_start in (range(0, steps, effective_callback_every) if steps > 0 else range(0)):
            steps_this_call = min(effective_callback_every, steps - t_loop_start)
            if steps_this_call <= 0: break
            
            opt_state_z, sf_baseline_state = self._svgd_loop_python(
                t_loop_start, steps_this_call, (opt_state_z, sf_baseline_state), main_generator)

            if callback:
                current_z_particles = self.get_params(opt_state_z)
                callback_kwargs = {"dibs": self, "t": t_loop_start + steps_this_call, "zs": current_z_particles.detach().cpu()}
                if hasattr(self, 'likelihood_model') and hasattr(self.likelihood_model, 'get_params') and isinstance(self.get_params(opt_state_z), tuple) : # JointDiBS like signature check
                     callback_kwargs["thetas"] = self.get_params(opt_state_z)[1].detach().cpu() # Example for joint
                callback(**callback_kwargs)
        
        final_z = self.get_params(opt_state_z)
        return self.particle_to_g_lim(final_z.detach())

    def get_empirical(self, g_samples: torch.Tensor) -> ParticleDistributionTorch:
        N = g_samples.shape[0]
        if N == 0: return ParticleDistributionTorch(logp=torch.tensor([], device=self.device), g=torch.tensor([], device=self.device))
        
        g_flat = g_samples.reshape(N, -1)
        # Note: torch.unique behavior on CUDA for bool can differ or be slower. Convert to int if issues.
        unique_graphs_flat, _, counts = torch.unique(g_flat.cpu(), dim=0, return_inverse=True, return_counts=True) # Perform on CPU
        
        unique_graphs = unique_graphs_flat.reshape(-1, self.n_vars, self.n_vars).to(self.device)
        logp = torch.log(counts.float()) - math.log(N)
        return ParticleDistributionTorch(logp=logp.to(self.device), g=unique_graphs)

    def get_mixture(self, g_samples: torch.Tensor, generator: Optional[torch.Generator] = None) -> ParticleDistributionTorch:
        N = g_samples.shape[0]
        if N == 0: return ParticleDistributionTorch(logp=torch.tensor([], device=self.device), g=torch.tensor([], device=self.device))

        logp_list = [self.log_joint_prob_fn(g_samples[i], None, self.x, self.interv_mask, generator) for i in range(N)]
        logp = torch.stack(logp_list)
        logp_normalized = logp - torch.logsumexp(logp, dim=0)
        return ParticleDistributionTorch(logp=logp_normalized, g=g_samples)


# --- JointDiBSTorch Class (Structure) ---
class JointDiBSTorch(DiBS): # DiBSTorch from previous response
    def __init__(self, *,
                 x: torch.Tensor,
                 graph_model: Any, # PyTorch version
                 likelihood_model: Any, # PyTorch version, needs sample_parameters, interventional_log_joint_prob
                 interv_mask: Optional[torch.Tensor] = None,
                 kernel_cls: Callable, # E.g., JointAdditiveFrobeniusSEKernelTorch
                 kernel_param: Optional[Dict] = None,
                 optimizer_name: str = "rmsprop",
                 optimizer_param: Optional[Dict] = None,
                 alpha_linear: float = 0.05, # JAX default for Joint
                 beta_linear: float = 1.0,
                 tau: float = 1.0,
                 n_grad_mc_samples: int = 128,
                 n_acyclicity_mc_samples: int = 32,
                 grad_estimator_z: str = "reparam", # Default for JointDiBS in JAX
                 score_function_baseline_factor: float = 0.0,
                 latent_prior_std: Optional[float] = None,
                 k_latent_dim: Optional[int] = None,
                 device: str = 'cpu',
                 verbose: bool = False):

        if kernel_param is None: kernel_param = {"h_latent": 5.0, "h_theta": 500.0} # Example
        if optimizer_param is None: optimizer_param = {"step_size": 0.005}
        
        effective_device = torch.device(device)
        if interv_mask is None:
            interv_mask = torch.zeros_like(x, dtype=torch.bool, device=effective_device)
        else:
            interv_mask = interv_mask.to(dtype=torch.bool, device=effective_device)

        super().__init__(
            x=x, interv_mask=interv_mask,
            log_graph_prior=graph_model.unnormalized_log_prob_soft,
            log_joint_prob=likelihood_model.interventional_log_joint_prob, # This is correct for JointDiBS
            alpha_linear=alpha_linear, beta_linear=beta_linear, tau=tau,
            n_grad_mc_samples=n_grad_mc_samples, n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z, score_function_baseline_factor=score_function_baseline_factor,
            latent_prior_std=latent_prior_std, k_latent_dim=k_latent_dim,
            device=device, verbose=verbose
        )

        self.likelihood_model = likelihood_model # Has .sample_parameters(), .interventional_log_joint_prob()
        self.graph_model = graph_model
        self.kernel = kernel_cls(**kernel_param, device=self.device) # Kernel instance for (z, theta)

        opt_step_size = optimizer_param.get('step_size', 0.005)
        if optimizer_name.lower() == 'sgd':
            self.opt_manual_impl = TorchSGD(step_size=opt_step_size)
        elif optimizer_name.lower() == 'rmsprop':
            rms_decay = optimizer_param.get('decay', 0.9)
            rms_eps = optimizer_param.get('eps', 1e-8)
            self.opt_manual_impl = TorchRMSprop(step_size=opt_step_size, alpha=rms_decay, eps=rms_eps)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        self.opt_init = self.opt_manual_impl.init
        self.opt_update = self.opt_manual_impl.update
        self.get_params = self.opt_manual_impl.get_params
    
    def _sample_initial_random_particles(self, *, generator: Optional[torch.Generator], 
                                     n_particles: int, n_dim: Optional[int] = None) -> Tuple[torch.Tensor, Any]:
        # Sample Z
        effective_n_dim = n_dim if n_dim is not None else self.n_vars
        if effective_n_dim <= 0: effective_n_dim = 1
        
        std_z = self.latent_prior_std_val
        if std_z is None or std_z <=0:
            std_z = 1.0 / math.sqrt(effective_n_dim) if effective_n_dim > 0 else 1.0
        
        shape_z = (n_particles, self.n_vars, effective_n_dim, 2)
        rand_fn_z = torch.randn if generator is None else lambda *s, **kw: torch.normal(0.,1.,size=s,generator=kw.get('generator'),device=kw.get('device'),dtype=kw.get('dtype'))
        z_particles = rand_fn_z(shape_z, generator=generator, device=self.device, dtype=torch.float32) * std_z

        # Sample Theta using likelihood_model.sample_parameters
        # This method in JAX returns a PyTree with leading dim n_particles.
        # The PyTorch equivalent should do the same.
        theta_particles = self.likelihood_model.sample_parameters(
            generator=generator, n_particles=n_particles, n_vars=self.n_vars
        ) # This method needs to be PyTorch compatible.
        return z_particles, theta_particles

    def _f_kernel(self, x_latent: torch.Tensor, x_theta: Any, 
                  y_latent: torch.Tensor, y_theta: Any) -> torch.Tensor:
        return self.kernel.eval(x_latent=x_latent, x_theta=x_theta, 
                                y_latent=y_latent, y_theta=y_theta)

    def _f_kernel_mat(self, x_latents: torch.Tensor, x_thetas: Any, 
                      y_latents: torch.Tensor, y_thetas: Any) -> torch.Tensor:
        # x_latents [A,d,k,2], x_thetas PyTree (leading A), y_latents [B,d,k,2], y_thetas PyTree (leading B)
        # Output: [A,B]
        A, B = x_latents.shape[0], y_latents.shape[0]
        is_x_thetas_list = isinstance(x_thetas, (list,tuple))
        is_y_thetas_list = isinstance(y_thetas, (list,tuple))

        kernel_matrix = torch.empty((A,B), device=self.device, dtype=x_latents.dtype)
        for i in range(A):
            x_th_i = x_thetas[i] if is_x_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, x_thetas)
            for j in range(B):
                y_th_j = y_thetas[j] if is_y_thetas_list else self._tree_map(lambda leaf: leaf[j] if torch.is_tensor(leaf) else leaf, y_thetas)
                kernel_matrix[i,j] = self._f_kernel(x_latents[i], x_th_i, y_latents[j], y_th_j)
        return kernel_matrix

    def _eltwise_grad_kernel_z(self, x_latents_b: torch.Tensor, x_thetas_b: Any, 
                               y_latent_s: torch.Tensor, y_theta_s: Any) -> torch.Tensor:
        # grad w.r.t. x_latent (arg 0 of _f_kernel)
        grads_list = []
        is_x_thetas_list = isinstance(x_thetas_b, (list,tuple))
        for i in range(x_latents_b.shape[0]):
            x_l_s = x_latents_b[i].detach().requires_grad_(True)
            x_th_s = x_thetas_b[i] if is_x_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, x_thetas_b)
            
            kernel_val = self._f_kernel(x_l_s, x_th_s, y_latent_s, y_theta_s)
            grad_val = torch.autograd.grad(kernel_val, x_l_s, allow_unused=True)[0]
            grads_list.append(grad_val if grad_val is not None else torch.zeros_like(x_l_s))
            x_l_s.requires_grad_(False)
        return torch.stack(grads_list)

    def _eltwise_grad_kernel_theta(self, x_latents_b: torch.Tensor, x_thetas_b: Any, 
                                   y_latent_s: torch.Tensor, y_theta_s: Any) -> Any: # Returns PyTree of Grads
        # grad w.r.t. x_theta (arg 1 of _f_kernel)
        # This is complex due to PyTrees. Result should be list of grad_pytrees.
        all_particle_grad_pytrees = []
        is_x_thetas_list = isinstance(x_thetas_b, (list,tuple))

        for i in range(x_latents_b.shape[0]):
            x_l_s = x_latents_b[i] # Does not need grad here
            x_th_s_template = x_thetas_b[i] if is_x_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, x_thetas_b)
            
            theta_s_leaves, orig_s_states = self._prepare_theta_for_grad(x_th_s_template) # Makes leaves require_grad
            
            if not theta_s_leaves: # No differentiable parameters in this particle's theta
                grad_pytree_s = self._tree_map(lambda p: torch.zeros_like(p) if torch.is_tensor(p) else p, x_th_s_template)
            else:
                kernel_val = self._f_kernel(x_l_s, x_th_s_template, y_latent_s, y_theta_s)
                grads_for_s_leaves = torch.autograd.grad(kernel_val, theta_s_leaves, allow_unused=True)
                filled_grads_s_leaves = [g if g is not None else torch.zeros_like(p) for g,p in zip(grads_for_s_leaves, theta_s_leaves)]
                grad_pytree_s = self._reconstruct_theta_grads(x_th_s_template, filled_grads_s_leaves, theta_s_leaves)
            
            self._restore_theta_grad_state(orig_s_states)
            all_particle_grad_pytrees.append(grad_pytree_s)
        
        # Stack the PyTrees:
        if not all_particle_grad_pytrees: return self._tree_map(lambda p: torch.empty_like(p, shape=(0,*p.shape)) if torch.is_tensor(p) else p, x_thetas_b) # Example empty structure
        return self._tree_map_multi(lambda *leaves: torch.stack(leaves, dim=0) if all(torch.is_tensor(l) for l in leaves) else leaves[0], *all_particle_grad_pytrees)


    def _z_update_single_particle(self, current_particle_z: torch.Tensor, current_particle_theta: Any,
                                  all_particles_z: torch.Tensor, all_particles_theta: Any,
                                  kernel_row_for_current_particle: torch.Tensor, # k((zi,ti),(zj,tj)) for fixed i, all j
                                  grad_log_prob_all_zs: torch.Tensor) -> torch.Tensor:
        # grad_log_prob_all_zs: [N,d,k,2] ( d/dz_j log p(zj,tj) )

        term1 = torch.sum(kernel_row_for_current_particle.view(-1,1,1,1) * grad_log_prob_all_zs, dim=0)
        
        # repulsion: sum_j grad_zj k((zj,tj), (zi,ti))
        # _eltwise_grad_kernel_z(all_particles_z, all_particles_theta, current_particle_z, current_particle_theta)
        # This gives [N,d,k,2] where N-th entry is grad_zn k((zn,tn),(zi,ti))
        term2_all_grads = self._eltwise_grad_kernel_z(all_particles_z, all_particles_theta, current_particle_z, current_particle_theta)
        term2 = torch.sum(term2_all_grads, dim=0)
        
        svgd_phi_z = (term1 + term2) / all_particles_z.shape[0]
        return -svgd_phi_z

    def _parallel_update_z(self, all_zs: torch.Tensor, all_thetas: Any, kxx_full: torch.Tensor, grad_log_p_zs: torch.Tensor) -> torch.Tensor:
        phi_list = []
        is_all_thetas_list = isinstance(all_thetas, (list,tuple))
        for i in range(all_zs.shape[0]):
            theta_i = all_thetas[i] if is_all_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, all_thetas)
            phi_i = self._z_update_single_particle(all_zs[i], theta_i, all_zs, all_thetas, kxx_full[i,:], grad_log_p_zs)
            phi_list.append(phi_i)
        return torch.stack(phi_list)

    def _theta_update_single_particle(self, current_particle_z: torch.Tensor, current_particle_theta: Any,
                                      all_particles_z: torch.Tensor, all_particles_theta: Any,
                                      kernel_row_for_current_particle: torch.Tensor, 
                                      grad_log_prob_all_thetas: Any) -> Any: # Returns PyTree
        # grad_log_prob_all_thetas: PyTree, leading dim N ( d/dtj log p(zj,tj) )
        
        # Term 1: sum_j k((zi,ti),(zj,tj)) * grad_log_p(tj)
        # kernel_row: [N]. grad_log_prob_all_thetas: PyTree with Tensors [N,...]
        # We need to expand kernel_row for each leaf in grad_log_prob_all_thetas
        def weighted_sum_fn(grad_leaf_all_j): # grad_leaf_all_j is [N, *leaf_shape]
            k_expanded = expand_by_pytorch(kernel_row_for_current_particle, grad_leaf_all_j.ndim - 1)
            return torch.sum(k_expanded * grad_leaf_all_j, dim=0)
        term1 = self._tree_map(weighted_sum_fn, grad_log_prob_all_thetas)
        
        # Term 2: sum_j grad_ti k((zj,tj), (zi,ti))
        # _eltwise_grad_kernel_theta(all_particles_z, all_particles_theta, current_particle_z, current_particle_theta)
        # Returns PyTree with leading dim N. Each leaf [N, *leaf_shape] is grad_tj k((zj,tj), (zi,ti))
        term2_all_grads_pytree = self._eltwise_grad_kernel_theta(
            all_particles_z, all_particles_theta, current_particle_z, current_particle_theta)
        term2 = self._tree_map(lambda leaf_all_j_grads: torch.sum(leaf_all_j_grads, dim=0), term2_all_grads_pytree)
        
        num_particles = float(all_particles_z.shape[0])
        svgd_phi_theta = self._tree_map_multi(lambda t1, t2: (t1 + t2) / num_particles, term1, term2)
        return self._tree_map(lambda leaf: -leaf, svgd_phi_theta) # Negate

    def _parallel_update_theta(self, all_zs: torch.Tensor, all_thetas: Any, kxx_full: torch.Tensor, grad_log_p_thetas: Any) -> Any:
        phi_list_pytrees = []
        is_all_thetas_list = isinstance(all_thetas, (list,tuple))
        for i in range(all_zs.shape[0]):
            z_i = all_zs[i]
            theta_i = all_thetas[i] if is_all_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, all_thetas)
            phi_i_pytree = self._theta_update_single_particle(
                z_i, theta_i, all_zs, all_thetas, kxx_full[i,:], grad_log_p_thetas)
            phi_list_pytrees.append(phi_i_pytree)
        
        if not phi_list_pytrees: return self._tree_map(lambda p: torch.empty_like(p, shape=(0,*p.shape)) if torch.is_tensor(p) else p, all_thetas)
        return self._tree_map_multi(lambda *leaves: torch.stack(leaves, dim=0) if all(torch.is_tensor(l) for l in leaves) else leaves[0], *phi_list_pytrees)


    def _svgd_step(self, t: Union[int, float], opt_state_z: Any, opt_state_theta: Any,
                   current_iter_generators: List[Optional[torch.Generator]], 
                   sf_baseline_batch: torch.Tensor) -> Tuple[Any, Any, torch.Tensor]:
        
        z_particles = self.get_params(opt_state_z) # [N,d,k,2]
        theta_particles = self.get_params(opt_state_theta) # PyTree with leading N

        # Grad w.r.t theta: d/dtheta log p(theta,D|Z)
        dtheta_log_target = self.eltwise_grad_theta_likelihood(
            z_particles, theta_particles, t, current_iter_generators) # PyTree with leading N

        # Grad w.r.t Z: d/dz log p(theta,D|Z)
        dz_log_likelihood, sf_baseline_updated = self.eltwise_grad_z_likelihood(
            z_particles, theta_particles, sf_baseline_batch, t, current_iter_generators) # [N,d,k,2]
        
        dz_log_prior = self.eltwise_grad_latent_prior(z_particles, current_iter_generators, t) # [N,d,k,2]
        dz_log_target = dz_log_prior + dz_log_likelihood

        k_matrix = self._f_kernel_mat(z_particles, theta_particles, z_particles, theta_particles) # [N,N]

        phi_z_batch = self._parallel_update_z(z_particles, theta_particles, k_matrix, dz_log_target)
        phi_theta_batch = self._parallel_update_theta(z_particles, theta_particles, k_matrix, dtheta_log_target)

        new_opt_state_z = self.opt_update(t, phi_z_batch, opt_state_z)
        new_opt_state_theta = self.opt_update(t, phi_theta_batch, opt_state_theta)
        
        return new_opt_state_z, new_opt_state_theta, sf_baseline_updated

    def _svgd_loop_python(self, start_iter: int, n_steps_in_loop: int, init_args: Tuple, 
                          base_generator: torch.Generator) -> Tuple[Any, Any, torch.Tensor]:
        opt_state_z, opt_state_theta, sf_baseline = init_args
        
        for i in range(start_iter, start_iter + n_steps_in_loop):
            n_particles = self.get_params(opt_state_z).shape[0]
            step_generators = []
            for _ in range(n_particles):
                 seed_val = torch.empty((), dtype=torch.int64, device=self.device).random_(generator=base_generator).item()
                 g = torch.Generator(device=self.device); g.manual_seed(seed_val)
                 step_generators.append(g)
            
            opt_state_z, opt_state_theta, sf_baseline = self._svgd_step(
                i, opt_state_z, opt_state_theta, step_generators, sf_baseline)
        return opt_state_z, opt_state_theta, sf_baseline


    def sample(self, *, main_generator: torch.Generator, n_particles: int, steps: int, 
               n_dim_particles: Optional[int] = None, callback: Optional[Callable] = None, 
               callback_every: Optional[int] = None) -> Tuple[torch.Tensor, Any]:

        if self.k_latent_dim is None:
            self.k_latent_dim = n_dim_particles if n_dim_particles is not None else self.n_vars
            if self.latent_prior_std_val is None and self.k_latent_dim > 0:
                self.latent_prior_std_val = 1.0 / math.sqrt(self.k_latent_dim)

        init_z, init_theta = self._sample_initial_random_particles(
            generator=main_generator, n_particles=n_particles, n_dim=n_dim_particles)
        
        sf_baseline_state = torch.zeros(n_particles, device=self.device)
        opt_state_z = self.opt_init(init_z)
        opt_state_theta = self.opt_init(init_theta) # Optimizer state for theta particles

        effective_callback_every = callback_every if callback_every is not None else steps
        if steps == 0: effective_callback_every = 0

        for t_loop_start in (range(0, steps, effective_callback_every) if steps > 0 else range(0)):
            steps_this_call = min(effective_callback_every, steps - t_loop_start)
            if steps_this_call <= 0: break
            
            opt_state_z, opt_state_theta, sf_baseline_state = self._svgd_loop_python(
                t_loop_start, steps_this_call, 
                (opt_state_z, opt_state_theta, sf_baseline_state), 
                main_generator)

            if callback:
                current_z = self.get_params(opt_state_z)
                current_theta = self.get_params(opt_state_theta)
                callback(dibs=self, t=t_loop_start + steps_this_call, 
                         zs=current_z.detach().cpu(), 
                         thetas=self._tree_map(lambda t: t.detach().cpu() if torch.is_tensor(t) else t, current_theta))
        
        final_z = self.get_params(opt_state_z).detach()
        final_theta = self._tree_map(lambda t: t.detach() if torch.is_tensor(t) else t, self.get_params(opt_state_theta))
        
        g_final_samples = self.particle_to_g_lim(final_z)
        return g_final_samples, final_theta

    def get_empirical(self, g_samples: torch.Tensor, theta_samples: Any) -> ParticleDistributionTorch:
        N = g_samples.shape[0]
        if N == 0: return ParticleDistributionTorch(logp=torch.tensor([], device=self.device), g=torch.tensor([], device=self.device), theta=None)
        # For joint, (G, Theta) particles are unique due to continuous Theta
        logp = -math.log(N) * torch.ones(N, device=self.device)
        return ParticleDistributionTorch(logp=logp, g=g_samples, theta=theta_samples)

    def get_mixture(self, g_samples: torch.Tensor, theta_samples: Any, generator: Optional[torch.Generator] = None) -> ParticleDistributionTorch:
        N = g_samples.shape[0]
        if N == 0: return ParticleDistributionTorch(logp=torch.tensor([], device=self.device), g=torch.tensor([], device=self.device), theta=None)

        is_thetas_list = isinstance(theta_samples, (list,tuple))
        logp_list = []
        for i in range(N):
            theta_i = theta_samples[i] if is_thetas_list else self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) else leaf, theta_samples)
            val = self.log_joint_prob_fn(g_samples[i], theta_i, self.x, self.interv_mask, generator)
            logp_list.append(val)
        
        logp = torch.stack(logp_list)
        logp_normalized = logp - torch.logsumexp(logp, dim=0)
        return ParticleDistributionTorch(logp=logp_normalized, g=g_samples, theta=theta_samples)