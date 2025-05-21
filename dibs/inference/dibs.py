import torch
import torch.nn.functional as F
import math
from typing import Callable, Any, Tuple, List, Dict, Optional, Union

# Assuming helper functions from previous translations are available or defined here:
# These include:
# - zero_diagonal (zero_diagonal_batch_pytorch_v2)
# - acyclic_constr_nograd_pytorch
# - sample_logistic_pytorch
# - expand_by_pytorch
# - logsumexp_pytorch_for_dibs (a specialized logsumexp for DiBS's usage)

# --- PyTorch Helper Function Implementations (from prior thought process) ---
def zero_diagonal(matrices: torch.Tensor) -> torch.Tensor: # Renamed for consistency
    """Sets the diagonal of a batch of matrices to zero."""
    if matrices.ndim < 2: raise ValueError("Input must be at least 2D.")
    if matrices.shape[-2] != matrices.shape[-1]: raise ValueError("Matrices must be square.")
    if matrices.numel() == 0: return matrices.clone() # Handle empty tensor

    multiplier = torch.ones_like(matrices)
    diag_ones = torch.eye(matrices.shape[-1], device=matrices.device, dtype=matrices.dtype)
    while diag_ones.ndim < matrices.ndim:
        diag_ones = diag_ones.unsqueeze(0)
    
    multiplier_no_diag = 1.0 - diag_ones
    return matrices * multiplier_no_diag

def acyclic_constr_nograd_pytorch(G: torch.Tensor, n_vars: int) -> torch.Tensor:
    if n_vars == 0: return torch.tensor(0.0, device=G.device, dtype=G.dtype)
    G_float = G.float()
    G_odot_G = G_float * G_float 
    # PyTorch 1.8+ for torch.linalg.matrix_exp
    if not hasattr(torch.linalg, 'matrix_exp'):
        raise RuntimeError("torch.linalg.matrix_exp not available. Update PyTorch or implement fallback.")
    exp_G_odot_G = torch.linalg.matrix_exp(G_odot_G)
    return torch.trace(exp_G_odot_G) - n_vars

def sample_logistic_pytorch(shape: Tuple[int, ...], generator: Optional[torch.Generator] = None, 
                            device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    rand_fn = torch.rand if generator is None else lambda *args, **kwargs: torch.rand(*args, **kwargs, generator=generator)
    u = rand_fn(shape, device=device, dtype=dtype)
    eps_val = torch.finfo(u.dtype).eps
    u = torch.clamp(u, eps_val, 1.0 - eps_val)
    return torch.log(u) - torch.log1p(-u)

def expand_by_pytorch(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if n_dims == 0: return tensor
    return tensor.view(*tensor.shape, *((1,) * n_dims))

def logsumexp_pytorch_for_dibs(a_vec: torch.Tensor, b_matrix_or_vec: torch.Tensor, 
                               axis_to_sum: int, return_sign: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Adapting for b_matrix_or_vec potentially being a vector (if b is grad_z which is reshaped from higher dim)
    # Or if b_matrix_or_vec is a scalar constant for all elements in sum.
    # JAX logsumexp(a, b=coeffs) is log(sum(coeffs * exp(a)))
    # If a is logprobs_numerator_adjusted [S], b is grad_z [D, S], axis=1 sums over S
    # This means a is broadcast to [D,S].
    
    A_broadcast = a_vec
    B_effective = b_matrix_or_vec

    if A_broadcast.ndim == 1 and B_effective.ndim == 2 and A_broadcast.shape[0] == B_effective.shape[axis_to_sum]:
        # Common case: A_broadcast [S], B_effective [D,S], axis_to_sum=1 -> A_broadcast needs to be [1,S]
        A_broadcast = A_broadcast.unsqueeze(0) # [1, S]
    elif A_broadcast.ndim == B_effective.ndim: # Shapes should match or be broadcastable
        pass
    else: # Mismatch that needs specific handling or error
         raise ValueError(f"Shape mismatch or unhandled broadcast for logsumexp: a={a_vec.shape}, b={b_matrix_or_vec.shape}")


    # Stability using a_max from the 'a' part of the expression: exp(a_max) * sum (b * exp(a - a_max))
    # The 'a' part here is A_broadcast. Max over the axis_to_sum.
    a_max = torch.max(A_broadcast, dim=axis_to_sum, keepdim=True)[0]
    a_max = torch.where(torch.isneginf(a_max), torch.zeros_like(a_max), a_max)

    sum_val_stable = torch.sum(B_effective * torch.exp(A_broadcast - a_max), dim=axis_to_sum)
    
    # Remove keepdim=True dimension from a_max
    a_max_squeezed = a_max.squeeze(axis_to_sum)

    if return_sign:
        sgn = torch.sign(sum_val_stable)
        abs_sum_val = torch.abs(sum_val_stable)
        log_abs_sum = torch.log(abs_sum_val)
        # Handle sum_val_stable == 0 resulting in log(0) = -inf
        log_abs_sum = torch.where(abs_sum_val == 0, torch.full_like(abs_sum_val, -torch.inf), log_abs_sum)
        final_log_val = a_max_squeezed + log_abs_sum
        return final_log_val, sgn
    else:
        if torch.any(B_effective < 0):
            raise ValueError("If not return_sign, all elements in b (multiplier) must be non-negative.")
        final_log_val = a_max_squeezed + torch.log(sum_val_stable) # log(0) will be -inf
        return final_log_val

# --- DiBS PyTorch Class ---
class DiBS:
    def __init__(self, *,
                 x: torch.Tensor,
                 interv_mask: torch.Tensor,
                 log_graph_prior: Callable[[torch.Tensor], torch.Tensor], # soft_g [d,d] -> scalar_tensor
                 log_joint_prob: Callable[[torch.Tensor, Any, torch.Tensor, torch.Tensor, Optional[torch.Generator]], torch.Tensor], # g [d,d], theta, x, interv, gen -> scalar_tensor
                 alpha_linear: float = 0.05,
                 beta_linear: float = 1.0,
                 tau: float = 1.0,
                 n_grad_mc_samples: int = 128,
                 n_acyclicity_mc_samples: int = 32,
                 grad_estimator_z: str = 'reparam',
                 score_function_baseline_factor: float = 0.0, # Factor for EMA update
                 latent_prior_std: Optional[float] = None,
                 k_latent_dim: Optional[int] = None, # Added: k for Z; needed for default latent_prior_std
                 device: str = 'cpu',
                 verbose: bool = False):
        
        self.device = torch.device(device)
        self.x = x.to(self.device)
        self.interv_mask = interv_mask.to(self.device)
        self.n_vars = x.shape[-1]
        
        self.log_graph_prior_fn = log_graph_prior
        self.log_joint_prob_fn = log_joint_prob

        self.alpha_fn = lambda t: (alpha_linear * float(t))
        self.beta_fn = lambda t: (beta_linear * float(t))
        self.tau = tau
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.grad_estimator_z = grad_estimator_z
        self.score_function_baseline_factor = score_function_baseline_factor
        
        self.k_latent_dim = k_latent_dim # Store k
        if latent_prior_std is None:
            if self.k_latent_dim is not None and self.k_latent_dim > 0:
                self.latent_prior_std_val = 1.0 / math.sqrt(self.k_latent_dim)
            else: # k is not known or invalid, cannot set default std
                self.latent_prior_std_val = None # Will cause error if prior is used without std
        else:
            self.latent_prior_std_val = latent_prior_std

        self.verbose = verbose

    # --- PyTree Helpers (Basic) ---
    def _tree_map(self, func: Callable, tree: Any) -> Any:
        if isinstance(tree, torch.Tensor): return func(tree)
        if isinstance(tree, list): return [self._tree_map(func, item) for item in tree]
        if isinstance(tree, tuple): return tuple(self._tree_map(func, item) for item in tree)
        if isinstance(tree, dict): return {k: self._tree_map(func, v) for k, v in tree.items()}
        return tree # Non-tensor, non-collection leaf

    def _tree_map_multi(self, func: Callable, *trees: Any) -> Any:
        first_tree = trees[0]
        if isinstance(first_tree, torch.Tensor): return func(*trees)
        if isinstance(first_tree, list): return [self._tree_map_multi(func, *[t[i] for t in trees]) for i in range(len(first_tree))]
        if isinstance(first_tree, tuple): return tuple(self._tree_map_multi(func, *[t[i] for t in trees]) for i in range(len(first_tree)))
        if isinstance(first_tree, dict): return {k: self._tree_map_multi(func, *[t[k] for t in trees]) for k in first_tree}
        return first_tree


    # --- Backbone functionality ---
    def particle_to_g_lim(self, z: torch.Tensor) -> torch.Tensor:
        u, v = z[..., 0], z[..., 1]
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        g_samples = (scores > 0).to(torch.int32)
        return zero_diagonal(g_samples)

    def sample_g(self, p_probs: torch.Tensor, generator: Optional[torch.Generator], n_samples: int) -> torch.Tensor:
        p_b = p_probs.unsqueeze(0).expand(n_samples, -1, -1) if p_probs.ndim == 2 else p_probs
        rand_fn = torch.bernoulli if generator is None else lambda pr: torch.bernoulli(pr, generator=generator)
        g_samples = rand_fn(p_b).to(torch.int32)
        return zero_diagonal(g_samples)

    def particle_to_soft_graph(self, z_single: torch.Tensor, eps_noise: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        scores = torch.einsum('ik,jk->ij', z_single[..., 0], z_single[..., 1]) # z_single is [d,k,2]
        alpha_val = self.alpha_fn(t)
        soft_graph = torch.sigmoid(self.tau * (eps_noise + alpha_val * scores)) # eps_noise is [d,d]
        return zero_diagonal(soft_graph.unsqueeze(0)).squeeze(0) # Add/remove batch for zero_diagonal

    def particle_to_hard_graph(self, z_single: torch.Tensor, eps_noise: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        scores = torch.einsum('ik,jk->ij', z_single[..., 0], z_single[..., 1])
        alpha_val = self.alpha_fn(t)
        hard_graph = ((eps_noise + alpha_val * scores) > 0.0).to(z_single.dtype)
        return zero_diagonal(hard_graph.unsqueeze(0)).squeeze(0)

    # --- Generative graph model p(G | Z) ---
    def edge_probs(self, z: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        u, v = z[..., 0], z[..., 1]
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        probs = torch.sigmoid(self.alpha_fn(t) * scores)
        return zero_diagonal(probs)

    def edge_log_probs(self, z: torch.Tensor, t: Union[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        u, v = z[..., 0], z[..., 1]
        scores = torch.einsum('...ik,...jk->...ij', u, v)
        alpha_t_scores = self.alpha_fn(t) * scores
        log_p = F.logsigmoid(alpha_t_scores)
        log_1_p = F.logsigmoid(-alpha_t_scores) # log(1-sigmoid(x)) = log(sigmoid(-x))
        return zero_diagonal(log_p), zero_diagonal(log_1_p)

    def latent_log_prob(self, single_g: torch.Tensor, single_z: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        log_p, log_1_p = self.edge_log_probs(single_z, t)
        log_prob_g_ij = single_g * log_p + (1.0 - single_g) * log_1_p
        return torch.sum(log_prob_g_ij)

    def _get_grads_for_loop(self, target_fn: Callable, loop_iterable: torch.Tensor, static_args: tuple, grad_arg_idx: int, requires_grad_tensor: torch.Tensor):
        """Helper to simulate vmap(grad(target_fn, grad_arg_idx), ...)(loop_iterable, *static_args)"""
        grads_list = []
        # Detach the tensor we differentiate w.r.t. to avoid interference across loop iterations for its .grad field
        # Or, ensure it's a new tensor in each iteration if target_fn modifies it (not typical for JAX grad).
        # Here, grad_arg_idx refers to the argument of target_fn.
        # If that argument is `requires_grad_tensor` itself or part of `static_args`.

        # Example: target_fn(iter_arg, static_arg1, static_arg2_diff) where static_arg2_diff is requires_grad_tensor.
        # grad_arg_idx would point to static_arg2_diff's position in target_fn's signature.

        # For eltwise_grad_latent_log_prob:
        # target_fn = self.latent_log_prob(g_sample, z_for_grad, t)
        # grad_arg_idx = 1 (for z_for_grad)
        # requires_grad_tensor = z_for_grad (which is single_z made to require_grad)
        # loop_iterable = gs_batch (provides g_sample)
        # static_args for latent_log_prob are (z_for_grad, t) excluding g_sample
        # This structure is a bit off. Let's simplify.

        # This helper is for when we vmap a `grad(f, argnum)` call.
        # JAX: vmap(grad(self.latent_log_prob, 1), (0, None, None), 0)(gs, single_z, t)
        # grad is w.r.t single_z (arg 1 of latent_log_prob).
        # vmap is over gs (arg 0 of the grad_fn, which corresponds to arg 0 of latent_log_prob).
        # single_z and t are None in_dims for vmap (passed as is).

        z_local_grad = requires_grad_tensor.detach().requires_grad_(True)
        for item_from_iterable in loop_iterable:
            # Construct args for target_fn for this iteration
            # This depends on how static_args and item_from_iterable map to target_fn's signature
            current_args = []
            grad_input_tensor_in_args = None

            # This part needs to be specific to the function being called.
            # For self.latent_log_prob(g, z, t) with grad w.r.t z (arg 1)
            # item_from_iterable is g_sample. static_args are (t_val). z is z_local_grad.
            if target_fn == self.latent_log_prob: # Specific to eltwise_grad_latent_log_prob
                g_sample, t_val = item_from_iterable, static_args[0] # static_args only contains t
                val = target_fn(g_sample, z_local_grad, t_val)
                grad_input_tensor_in_args = z_local_grad
            elif target_fn == self.log_joint_prob_soft: # Specific to grad_z_likelihood_gumbel
                # grad(self.log_joint_prob_soft, 0) -> grad w.r.t. single_z (arg 0)
                # vmap args: (None, None, 0, None, None) -> single_z, single_theta, eps_sample, t, gen
                # item_from_iterable is eps_sample. static_args are (single_theta, t_val, gen_val)
                # requires_grad_tensor here is single_z.
                single_z_ref, single_theta_ref, t_val_ref, gen_ref = static_args # Unpack static args
                eps_sample_iter = item_from_iterable
                val = target_fn(z_local_grad, single_theta_ref, eps_sample_iter, t_val_ref, gen_ref)
                grad_input_tensor_in_args = z_local_grad
            elif target_fn == self.constraint_gumbel: # Specific to grad_constraint_gumbel
                # grad(self.constraint_gumbel, 0) -> grad w.r.t single_z (arg 0)
                # vmap args: (None, 0, None) -> single_z, eps_sample, t
                single_z_ref, t_val_ref = static_args
                eps_sample_iter = item_from_iterable
                val = target_fn(z_local_grad, eps_sample_iter, t_val_ref)
                grad_input_tensor_in_args = z_local_grad
            else:
                raise NotImplementedError(f"Gradient loop helper not implemented for {target_fn}")
            
            if z_local_grad.grad is not None: z_local_grad.grad.zero_()
            
            # grad_output = torch.ones_like(val) # if val is not scalar
            # grad_val_tuple = torch.autograd.grad(val, grad_input_tensor_in_args, grad_outputs=grad_output, retain_graph=True, allow_unused=True)
            grad_val_tuple = torch.autograd.grad(val, grad_input_tensor_in_args, retain_graph=True, allow_unused=True)

            grad_val = grad_val_tuple[0] if grad_val_tuple[0] is not None else torch.zeros_like(grad_input_tensor_in_args)
            grads_list.append(grad_val)
        
        z_local_grad.requires_grad_(False)
        return torch.stack(grads_list, dim=0)


    def eltwise_grad_latent_log_prob(self, gs_batch: torch.Tensor, single_z: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        # JAX: vmap(grad(self.latent_log_prob, 1), (0, None, None), 0)(gs, single_z, t)
        # grad w.r.t. single_z (arg 1 of latent_log_prob). vmap over gs (arg 0).
        return self._get_grads_for_loop(self.latent_log_prob, gs_batch, (t,), 1, single_z)

    def eltwise_log_joint_prob(self, gs_batch: torch.Tensor, single_theta: Any, generator: Optional[torch.Generator]) -> torch.Tensor:
        log_probs_list = [self.log_joint_prob_fn(gs_batch[i], single_theta, self.x, self.interv_mask, generator) for i in range(gs_batch.shape[0])]
        return torch.stack(log_probs_list)

    def log_joint_prob_soft(self, single_z: torch.Tensor, single_theta: Any, eps_noise: torch.Tensor, t: Union[int, float], generator: Optional[torch.Generator]) -> torch.Tensor:
        soft_g_sample = self.particle_to_soft_graph(single_z, eps_noise, t)
        return self.log_joint_prob_fn(soft_g_sample, single_theta, self.x, self.interv_mask, generator)

    # Gradient estimators for d/dZ log p(theta, D | Z)
    def eltwise_grad_z_likelihood(self, zs_batch: torch.Tensor, thetas_collection: Union[List[Any], Tuple[Any, ...], Any], 
                                  baselines_batch: torch.Tensor, t: Union[int, float], 
                                  generators_batch: List[Optional[torch.Generator]]) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_fn_selected = self.grad_z_likelihood_score_function if self.grad_estimator_z == 'score' else \
                           self.grad_z_likelihood_gumbel if self.grad_estimator_z == 'reparam' else \
                           (_ for _ in ()).throw(ValueError(f'Unknown Z-gradient estimator: {self.grad_estimator_z}'))

        out_grads_z, out_baselines = [], []
        is_thetas_batched_list = isinstance(thetas_collection, (list, tuple)) and len(thetas_collection) == zs_batch.shape[0]

        for i in range(zs_batch.shape[0]):
            single_z = zs_batch[i]
            single_theta = thetas_collection[i] if is_thetas_batched_list else \
                           self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) and hasattr(leaf, 'shape') and len(leaf.shape)>0 and leaf.shape[0] == zs_batch.shape[0] else leaf, thetas_collection)
            
            current_baseline = baselines_batch[i] if baselines_batch is not None else torch.tensor(0.0, device=self.device) # Add default if None
            single_gen = generators_batch[i] if generators_batch and i < len(generators_batch) else None
            
            grad_z, updated_baseline = grad_fn_selected(single_z, single_theta, current_baseline, t, single_gen)
            out_grads_z.append(grad_z)
            out_baselines.append(updated_baseline)
        
        return torch.stack(out_grads_z), torch.stack(out_baselines) if out_baselines else None

    def grad_z_likelihood_score_function(self, single_z: torch.Tensor, single_theta: Any, current_baseline_val: torch.Tensor, 
                                         t: Union[int, float], generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_for_grad = single_z.detach().requires_grad_(True)
        p_edge_probs = self.edge_probs(z_for_grad, t) # Depends on z_for_grad
        
        g_samples = self.sample_g(p_edge_probs.detach(), generator, self.n_grad_mc_samples) # Detach for REINFORCE sampling
        logprobs_f_G = self.eltwise_log_joint_prob(g_samples, single_theta, generator)
        
        f_minus_b = logprobs_f_G
        if self.score_function_baseline_factor > 0.0: # Baseline is active
            f_minus_b = logprobs_f_G - current_baseline_val
        
        grad_log_p_G_Z = self.eltwise_grad_latent_log_prob(g_samples, z_for_grad, t) # [S, d, k, 2]
        
        # REINFORCE: E_G[(f(G)-b) * grad_Z log p(G|Z)]
        term_to_sum = f_minus_b.view(-1, *((1,) * (grad_log_p_G_Z.ndim - 1))) * grad_log_p_G_Z
        estimated_grad_z = torch.mean(term_to_sum, dim=0)

        new_baseline_val = current_baseline_val
        if self.score_function_baseline_factor > 0.0:
            new_baseline_val = (self.score_function_baseline_factor * logprobs_f_G.mean() + \
                               (1.0 - self.score_function_baseline_factor) * current_baseline_val)
        
        z_for_grad.requires_grad_(False)
        return estimated_grad_z, new_baseline_val.detach()

    def grad_z_likelihood_gumbel(self, single_z: torch.Tensor, single_theta: Any, current_baseline_val: torch.Tensor, 
                                 t: Union[int, float], generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_for_grad = single_z.detach().requires_grad_(True)
        n_vars = single_z.shape[0]
        eps_noise_batch = sample_logistic_pytorch((self.n_grad_mc_samples, n_vars, n_vars), generator, self.device, single_z.dtype)

        # JAX: vmap(grad(self.log_joint_prob_soft, 0), (None, None, 0, None, None), 0)(single_z, single_theta, eps, t, subk_)
        # grad w.r.t. single_z (arg 0 of log_joint_prob_soft). vmap over eps_noise_batch (arg 2)
        static_args_for_loop = (single_theta, t, generator) # For self.log_joint_prob_soft's args other than z and eps
        mc_grads_z = self._get_grads_for_loop(self.log_joint_prob_soft, eps_noise_batch, static_args_for_loop, 0, z_for_grad)
        
        estimated_grad_z = mc_grads_z.mean(dim=0)
        z_for_grad.requires_grad_(False)
        return estimated_grad_z, current_baseline_val # Baseline not updated

    # --- Estimators for score d/dtheta log p(theta, D | Z) ---
    def _prepare_theta_for_grad(self, single_theta: Any) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, bool]]]:
        """Extracts leaf tensors from theta, makes them require grad, stores original states."""
        leaf_tensors_for_grad = []
        original_requires_grad_states = []
        def _get_leaves(pytree_node):
            if isinstance(pytree_node, torch.Tensor):
                original_requires_grad_states.append((pytree_node, pytree_node.requires_grad))
                pytree_node.requires_grad_(True)
                leaf_tensors_for_grad.append(pytree_node)
            elif isinstance(pytree_node, (list, tuple)): [ _get_leaves(item) for item in pytree_node ]
            elif isinstance(pytree_node, dict): [ _get_leaves(v) for v in pytree_node.values() ]
        _get_leaves(single_theta)
        return leaf_tensors_for_grad, original_requires_grad_states

    def _restore_theta_grad_state(self, original_states: List[Tuple[torch.Tensor, bool]]):
        for tensor, state in original_states: tensor.requires_grad_(state)

    def _reconstruct_theta_grads(self, single_theta_template: Any, final_grad_leaves: List[torch.Tensor], 
                                 leaf_tensors_original_order: List[torch.Tensor]) -> Any:
        """Reconstructs gradient PyTree from a flat list of gradient tensors."""
        final_grads_iter = iter(final_grad_leaves)
        # Create a mapping from original tensor object id to its gradient, for safety.
        # This assumes leaf_tensors_original_order and final_grad_leaves are aligned.
        grad_map = {id(orig_leaf): grad_leaf for orig_leaf, grad_leaf in zip(leaf_tensors_original_order, final_grad_leaves)}

        def _reconstruct(template_node):
            if isinstance(template_node, torch.Tensor):
                # If this tensor was differentiated, return its grad, else zeros.
                return grad_map.get(id(template_node), torch.zeros_like(template_node))
            elif isinstance(template_node, list): return [_reconstruct(item) for item in template_node]
            elif isinstance(template_node, tuple): return tuple(_reconstruct(item) for item in template_node)
            elif isinstance(template_node, dict): return {k: _reconstruct(v) for k, v in template_node.items()}
            return template_node # Non-tensor, non-collection leaves
        return _reconstruct(single_theta_template)


    def grad_theta_likelihood(self, single_z: torch.Tensor, single_theta: Any, 
                              t: Union[int, float], generator: Optional[torch.Generator]) -> Any:
        theta_leaves, orig_states = self._prepare_theta_for_grad(single_theta)
        if not theta_leaves:
            self._restore_theta_grad_state(orig_states)
            return self._tree_map(lambda p: torch.zeros_like(p) if torch.is_tensor(p) else p, single_theta)

        p_edge_probs = self.edge_probs(single_z, t)
        g_samples = self.sample_g(p_edge_probs.detach(), generator, self.n_grad_mc_samples)
        logprobs_f_G = self.eltwise_log_joint_prob(g_samples, single_theta, generator) # [S]

        per_g_grad_theta_leaves_list = [] # List of (list of grad_tensors for theta_leaves)
        for i in range(g_samples.shape[0]):
            g_sample_i = g_samples[i]
            # Clear .grad for leaf params if using .backward() style, not needed for torch.autograd.grad
            log_p_val = self.log_joint_prob_fn(g_sample_i, single_theta, self.x, self.interv_mask, generator)
            grads_for_this_g = torch.autograd.grad(log_p_val, theta_leaves, retain_graph=True, allow_unused=True)
            per_g_grad_theta_leaves_list.append([g if g is not None else torch.zeros_like(p) for g,p in zip(grads_for_this_g, theta_leaves)])
        
        grad_theta_leaves_stacked = [torch.stack([grads[i] for grads in per_g_grad_theta_leaves_list], dim=0) for i in range(len(theta_leaves))]

        final_grad_leaves = []
        log_denominator = logsumexp_pytorch(logprobs_f_G, axis=0) # Scalar, b=None

        for leaf_grad_S_shape in grad_theta_leaves_stacked: # leaf_grad_S_shape is [S, *param_shape]
            # logsumexp_pytorch_for_dibs(a_vec [S], b_matrix [D,S], axis_to_sum=1) output [D]
            # Here, a_vec is logprobs_f_G [S]. b_matrix needs to be [NumParamElements, S]
            # So, reshape leaf_grad_S_shape from [S, P1, P2..] to [S, TotalParamElements] then transpose to [TotalParamElements, S]
            original_param_shape = leaf_grad_S_shape.shape[1:]
            b_matrix_for_lse = leaf_grad_S_shape.reshape(self.n_grad_mc_samples, -1).transpose(0, 1) # [TotalParamElements, S]
            
            log_num_leaf, sign_leaf = logsumexp_pytorch_for_dibs(logprobs_f_G, b_matrix_for_lse, axis_to_sum=1, return_sign=True)
            
            # Reshape back to original param shape
            log_num_leaf = log_num_leaf.reshape(original_param_shape)
            sign_leaf = sign_leaf.reshape(original_param_shape)
            
            # JAX had log(n_mc_numerator) and log(n_mc_denominator) which are same here.
            stable_grad_leaf = sign_leaf * torch.exp(log_num_leaf - log_denominator)
            final_grad_leaves.append(stable_grad_leaf)

        self._restore_theta_grad_state(orig_states)
        return self._reconstruct_theta_grads(single_theta, final_grad_leaves, theta_leaves)

    def eltwise_grad_theta_likelihood(self, zs_batch: torch.Tensor, thetas_collection: Any, 
                                      t: Union[int, float], generators_batch: List[Optional[torch.Generator]]) -> Any:
        results_list = []
        is_thetas_batched_list = isinstance(thetas_collection, (list, tuple)) and len(thetas_collection) == zs_batch.shape[0]

        for i in range(zs_batch.shape[0]):
            single_z = zs_batch[i]
            single_theta = thetas_collection[i] if is_thetas_batched_list else \
                           self._tree_map(lambda leaf: leaf[i] if torch.is_tensor(leaf) and hasattr(leaf, 'shape') and len(leaf.shape)>0 and leaf.shape[0] == zs_batch.shape[0] else leaf, thetas_collection)
            single_gen = generators_batch[i] if generators_batch and i < len(generators_batch) else None
            results_list.append(self.grad_theta_likelihood(single_z, single_theta, t, single_gen))
        
        if not results_list: return None # Or empty structure matching theta
        return self._tree_map_multi(lambda *leaves: torch.stack(leaves, dim=0) if all(torch.is_tensor(l) for l in leaves) else leaves[0], *results_list)


    # --- Estimators for score d/dZ log p(Z) ---
    def constraint_gumbel(self, single_z: torch.Tensor, single_eps_noise: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        n_vars = single_z.shape[0]
        G_soft = self.particle_to_soft_graph(single_z, single_eps_noise, t)
        return acyclic_constr_nograd_pytorch(G_soft, n_vars)

    def grad_constraint_gumbel(self, single_z: torch.Tensor, generator: Optional[torch.Generator], t: Union[int, float]) -> torch.Tensor:
        z_for_grad = single_z.detach().requires_grad_(True)
        n_vars = single_z.shape[0]
        eps_noise_batch = sample_logistic_pytorch((self.n_acyclicity_mc_samples, n_vars, n_vars), generator, self.device, single_z.dtype)

        # JAX: vmap(grad(self.constraint_gumbel, 0), (None, 0, None), 0)(single_z, eps, t)
        # grad w.r.t single_z (arg 0). vmap over eps (arg 1).
        static_args_for_loop = (t,) # For self.constraint_gumbel's arg t
        mc_grads = self._get_grads_for_loop(self.constraint_gumbel, eps_noise_batch, static_args_for_loop, 0, z_for_grad)
        
        estimated_grad = mc_grads.mean(dim=0)
        z_for_grad.requires_grad_(False)
        return estimated_grad

    def log_graph_prior_particle(self, single_z: torch.Tensor, t: Union[int, float]) -> torch.Tensor:
        single_soft_g = self.edge_probs(single_z, t)
        return self.log_graph_prior_fn(soft_g=single_soft_g)

    def eltwise_grad_latent_prior(self, zs_batch: torch.Tensor, generators_batch: List[Optional[torch.Generator]], 
                                  t: Union[int, float]) -> torch.Tensor:
        # grad_log_f_Z_term
        grad_f_Z_list = []
        for i in range(zs_batch.shape[0]):
            z_s = zs_batch[i].detach().requires_grad_(True)
            log_f_val = self.log_graph_prior_particle(z_s, t)
            grad_s = torch.autograd.grad(log_f_val, z_s, allow_unused=True)[0]
            grad_f_Z_list.append(grad_s if grad_s is not None else torch.zeros_like(z_s))
            z_s.requires_grad_(False)
        grad_f_Z_term = torch.stack(grad_f_Z_list)

        # grad_constraint_term
        grad_h_Z_list = [self.grad_constraint_gumbel(zs_batch[i], (generators_batch[i] if generators_batch and i < len(generators_batch) else None), t) for i in range(zs_batch.shape[0])]
        grad_h_Z_term = torch.stack(grad_h_Z_list)
        
        # grad_gaussian_prior_term: - Z / std^2
        if self.latent_prior_std_val is None or self.latent_prior_std_val <= 0:
            grad_N_Z_term = torch.zeros_like(zs_batch)
            if self.verbose and self.latent_prior_std_val is None: print("Warning: latent_prior_std not set, Gaussian prior on Z has zero gradient.")
        else:
            grad_N_Z_term = -zs_batch / (self.latent_prior_std_val ** 2.0)
        
        return -self.beta_fn(t) * grad_h_Z_term + grad_N_Z_term + grad_f_Z_term

    def visualize_callback(self, ipython: bool = True, save_path: Optional[str] = None):
        # Visualization part is highly dependent on external libraries and their PyTorch compatibility
        # This is a simplified adaptation.
        try:
            from IPython import display as ipy_display # type: ignore
        except ImportError:
            ipy_display = None
        
        # Placeholder for actual plotting lib
        def _visualize_placeholder(probs_tensor, save_path_local, t_local, show):
            if self.verbose: print(f"[Viz@{t_local}] Mean edge prob sum: {probs_tensor.sum().item() if probs_tensor.numel() > 0 else 0.0 }")
            if save_path_local: print(f"[Viz@{t_local}] Would save to {save_path_local}_{t_local}.png")

        def _constraint_check_placeholder(gs_tensor_batch_or_single, n_vars_local):
            # Simplified check for demo
            num_cyclic_local = 0
            if gs_tensor_batch_or_single.numel() == 0: return 0
            
            if gs_tensor_batch_or_single.ndim == 2: # Single graph
                gs_tensor_batch_or_single = gs_tensor_batch_or_single.unsqueeze(0) # Make it a batch of 1
            
            for i in range(gs_tensor_batch_or_single.shape[0]):
                # A real check would use mat_is_dag_torch or similar
                # h_val = acyclic_constr_nograd_pytorch(gs_tensor_batch_or_single[i], n_vars_local)
                # if h_val > 1e-5: num_cyclic_local +=1 
                # Placeholder: check for self-loops as a proxy for not being a DAG (after zero_diagonal)
                # This is NOT a real acyclicity check.
                if torch.trace(gs_tensor_batch_or_single[i]) > 0 : num_cyclic_local +=1 
            return num_cyclic_local


        def callback_fn(**kwargs): # zs, dibs (self), t
            zs_p = kwargs["zs"]
            t_step = kwargs["t"]
            
            gs_lim_p = self.particle_to_g_lim(zs_p)
            probs_p = self.edge_probs(zs_p, t_step)
            
            # For visualization, usually mean over particles if zs_p is batched
            probs_to_show = probs_p.mean(dim=0) if probs_p.ndim > 2 and probs_p.shape[0] > 1 else probs_p
            
            if ipython and ipy_display: ipy_display.clear_output(wait=True)
            _visualize_placeholder(probs_to_show, save_path, t_step, True)
            
            num_particles_total = gs_lim_p.shape[0] if gs_lim_p.ndim == 3 else 1
            num_cyclic_found = _constraint_check_placeholder(gs_lim_p, self.n_vars)

            print(f'Iter {t_step:6d} | alpha {self.alpha_fn(t_step):6.2f} | beta {self.beta_fn(t_step):6.2f} | #cyclic {num_cyclic_found:3d}/{num_particles_total}')
        return callback_fn