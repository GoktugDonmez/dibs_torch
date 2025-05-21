import unittest
import torch
import functools
import collections.abc
from torch.utils._pytree import tree_map

# --- Pytree helpers (copied for self-containment) ---
def tree_map_torch(func, tree1, tree2):
    if isinstance(tree1, torch.Tensor) and isinstance(tree2, torch.Tensor):
        return func(tree1, tree2)
    elif isinstance(tree1, dict) and isinstance(tree2, dict):
        if set(tree1.keys()) != set(tree2.keys()):
            raise ValueError("Pytrees have different structures (dict keys).")
        return {k: tree_map_torch(func, tree1[k], tree2[k]) for k in tree1}
    elif isinstance(tree1, (list, tuple)) and isinstance(tree2, (list, tuple)):
        if len(tree1) != len(tree2):
            raise ValueError("Pytrees have different structures (list/tuple lengths).")
        return type(tree1)(tree_map_torch(func, t1, t2) for t1, t2 in zip(tree1, tree2))
    elif tree1 is None and tree2 is None: # Allow None nodes
        return None
    else:
        raise TypeError(f"Unsupported pytree structure or mismatched types: {type(tree1)}, {type(tree2)}")

def tree_map_single_torch(func, tree):
    if isinstance(tree, torch.Tensor):
        return func(tree)
    elif isinstance(tree, dict):
        return {k: tree_map_single_torch(func, tree[k]) for k in tree}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map_single_torch(func, item) for item in tree)
    elif tree is None: # Allow None nodes
        return None
    else:
        raise TypeError(f"Unsupported pytree structure: {type(tree)}")

def squared_norm_pytree(x, y):
    """Computes squared euclidean norm between two pytrees (PyTorch version)"""
    diff = tree_map_torch(torch.subtract, x, y)
    
    def _square_sum_leaf(leaf):
        if isinstance(leaf, torch.Tensor):
            return torch.square(leaf).sum()
        raise TypeError(f"Expected Tensor, got {type(leaf)}")

    squared_norm_ind_tree = tree_map_single_torch(_square_sum_leaf, diff)
    
    scalar_sums = []
    def _collect_leaves(subtree):
        if isinstance(subtree, torch.Tensor): 
            scalar_sums.append(subtree)
        elif isinstance(subtree, dict):
            for k in subtree:
                _collect_leaves(subtree[k])
        elif isinstance(subtree, (list, tuple)):
            for item in subtree:
                _collect_leaves(item)
        elif subtree is None: # If None was a valid leaf from tree_map
            pass
        else:
             raise TypeError(f"Unexpected type in tree reduction: {type(subtree)}")

    _collect_leaves(squared_norm_ind_tree)
    
    if not scalar_sums:
        # Determine device from inputs if possible, default to CPU
        # This part is a bit tricky if x or y are complex nested structures without any tensors.
        # For simplicity, assuming at least one tensor exists if a norm is meaningful,
        # or the structures are empty leading to a 0 norm.
        device = 'cpu'
        if isinstance(x, torch.Tensor): device = x.device
        elif isinstance(y, torch.Tensor): device = y.device
        # A more robust way would be to find the first tensor in x or y to get device.
        return torch.tensor(0.0, device=device)

    total_squared_norm = functools.reduce(torch.add, scalar_sums, torch.tensor(0.0, device=scalar_sums[0].device))
    return total_squared_norm

def zero_diagonal(g):
    """
    Returns the argument matrix with its diagonal set to zero. (PyTorch version)
    """
    if g.numel() == 0 : # Handle empty tensor
        return g.clone()
    if g.shape[-1] == 0 or g.shape[-2] == 0: # Handle matrices with zero dimension
        return g.clone()
        
    d = g.shape[-1]
    if g.shape[-2] != d:
        raise ValueError(f"Last two dimensions must be equal for diagonal, got {g.shape}")

    g_clone = g.clone()
    diag_indices = torch.arange(d, device=g.device)
    g_clone[..., diag_indices, diag_indices] = 0.0
    return g_clone

@torch.jit.script
def _slogdet_torch(m, parents_bool):
    """
    Log determinant of a submatrix. (PyTorch version)
    """
    n_vars = m.shape[-1]
    if m.shape[-2] != n_vars:
         raise ValueError(f"Matrix m must be square in its last two dimensions, got {m.shape}")
    if parents_bool.shape[-1] != n_vars:
        raise ValueError(f"Last dimension of parents_bool ({parents_bool.shape[-1]}) must match matrix dimension ({n_vars})")

    parents_float = parents_bool.float()
    mask = parents_float.unsqueeze(-1) * parents_float.unsqueeze(-2) 
    
    eye_mat = torch.eye(n_vars, device=m.device, dtype=m.dtype)
    
    submat = mask * m + (1.0 - mask) * eye_mat
    
    slogdet_result = torch.linalg.slogdet(submat)
    return slogdet_result.logabsdet

