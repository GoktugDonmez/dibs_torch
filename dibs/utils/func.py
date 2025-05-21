import unittest
import torch
import functools
import collections.abc
from torch.utils._pytree import tree_map


def expand_by(arr, n):
    """
    Expands torch.Tensor by n dimensions at the end
    Args:
        arr: shape [...]
        n (int)
    
    Returns:
        arr of shape [..., 1, ..., 1] with `n` ones
    """
    if n == 0:
        return arr
    new_shape = arr.shape + tuple(1 for _ in range(n))
    return arr.view(new_shape)

@torch.jit.script
def sel(mat, mask):
    """
    jit/vmap helper function (PyTorch version)

    Args:
        mat (torch.Tensor): Input tensor of shape [N, d].
        mask (torch.Tensor): Boolean tensor of shape [d,]. 

    Returns:
        torch.Tensor: Tensor of shape [N, d] with columns of `mat` where `mask == True` are non-zero
                      and the columns where `mask == False` are zero.

    Example:
        >>> mat = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mask = torch.tensor([True, False, True])
        >>> sel(mat, mask)
        tensor([[1, 0, 3],
                [4, 0, 6],
                [7, 0, 9]])
    """
    return torch.where(mask, mat, torch.zeros_like(mat))


@torch.jit.script
def leftsel(mat, mask, maskval: float = 0.0):
    """
    jit/vmap helper function (PyTorch version)

    Args:
        mat: [N, d]
        mask: [d, ]  boolean 

    Returns:
        [N, d] with columns of `mat` with `mask` == 1 non-zero
        and pushed leftmost; the columns with `mask` == 0 are filled with maskval
    """
    d = mask.shape[0]
    if d == 0: # Handle empty mask case
        return torch.empty_like(mat)

    arange_d = torch.arange(d, device=mat.device)
    
    valid_indices = torch.where(mask, arange_d, torch.tensor(d, device=mat.device, dtype=torch.long))
    
    maskval_tensor = torch.full((mat.shape[0], 1), maskval, device=mat.device, dtype=mat.dtype)
    padded_mat = torch.cat([mat, maskval_tensor], dim=1)
    
    sorted_indices = torch.sort(valid_indices).values
    
    padded_valid_mat = padded_mat[:, sorted_indices]
    return padded_valid_mat


@torch.jit.script
def mask_topk(x, topkk: int):
    """
    Returns indices of `topk` entries of `x` in decreasing order

    Args:
        x: [N, ]
        topkk (int)

    Returns:
        array of shape [topk, ]
    """
    
    num_elements = x.shape[0]
    if num_elements == 0: # If x is empty
        return torch.empty(0, dtype=torch.long, device=x.device)

    # Cap topkk at the number of available elements
    actual_topkk = min(topkk, num_elements)
    
    return torch.argsort(x, descending=True)[:actual_topkk]



def zero_diagonal(g: torch.Tensor) -> torch.Tensor:
    """
    Returns the argument matrix with its diagonal set to zero.

    Args:
        g (torch.Tensor): matrix of shape [..., d, d]
    """
    
    d = g.shape[-1]
    diag_mask = ~torch.eye(d, dtype=torch.bool, device=g.device)
    
    return g * diag_mask

def squared_norm_pytree(x, y):
    """Computes squared euclidean norm between two pytrees

    Args:
        x:  PyTree of torch.Tensor
        y:  PyTree of torch.Tensor

    Returns:
        shape [] (scalar tensor)
        
    Example:
        # For simple tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([2.0, 3.0, 4.0])
        norm = squared_norm_pytree(x, y)  # (1-2)² + (2-3)² + (3-4)² = 3
        
        # For nested structures
        x = {
            'a': torch.tensor([1.0, 2.0]),
            'b': torch.tensor([3.0, 4.0])
        }
        y = {
            'a': torch.tensor([2.0, 3.0]),
            'b': torch.tensor([4.0, 5.0])
        }
        norm = squared_norm_pytree(x, y)  # (1-2)² + (2-3)² + (3-4)² + (4-5)² = 4
    """
    # Compute difference between corresponding elements
    diff = tree_map(torch.sub, x, y)
    
    # Square each element and sum within each tensor
    squared_norm_ind = tree_map(lambda leaf: torch.square(leaf).sum() if isinstance(leaf, torch.Tensor) else leaf, diff)
    
    # If the result is already a tensor (simple case), return it
    if isinstance(squared_norm_ind, torch.Tensor):
        return squared_norm_ind
    
    # For nested structures, collect all tensor values and sum them
    tensor_values = []
    
    def collect_tensors(item):
        if isinstance(item, torch.Tensor):
            tensor_values.append(item)
        elif isinstance(item, dict):
            for v in item.values():
                collect_tensors(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                collect_tensors(v)
    
    collect_tensors(squared_norm_ind)
    
    if not tensor_values:
        return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
    
    return sum(tensor_values)



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

    # Handle empty mask case
    if not parents_bool.any():
        return torch.tensor(0.0, device=m.device, dtype=m.dtype)

    # Get submatrix by selecting rows and columns where mask is True
    sub_matrix = m[parents_bool][:, parents_bool]
    
    # Handle empty submatrix case
    if sub_matrix.numel() == 0:
        return torch.tensor(0.0, device=m.device, dtype=m.dtype)

    slogdet_result = torch.linalg.slogdet(sub_matrix)
    return slogdet_result.logabsdet
