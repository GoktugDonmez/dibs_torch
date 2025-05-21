import unittest
import torch
import functools
import collections.abc
from torch.utils._pytree import tree_map


def test_leftsel():
    """
            mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 3 0
        4 6 0
        7 9 0

    """
    mat = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = torch.tensor([True, False, True])
    assert torch.allclose(leftsel(mat, mask), torch.tensor([[1, 3, 0], [4, 6, 0], [7, 9, 0]]))
    print("test_leftsel passed")

def test_mask_topk():
    """Test the mask_topk function with various inputs."""
    # Test basic functionality
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    indices = mask_topk(x, 3)
    expected = torch.tensor([4, 2, 0], dtype=torch.long)
    assert torch.equal(indices, expected), f"Expected {expected}, got {indices}"

    # Test with empty tensor
    empty = torch.tensor([])
    indices_empty = mask_topk(empty, 3)
    assert indices_empty.shape == (0,), f"Expected shape (0,), got {indices_empty.shape}"
    assert indices_empty.dtype == torch.long, f"Expected dtype torch.long, got {indices_empty.dtype}"

    # Test with topkk=0
    indices_zero = mask_topk(x, 0)
    assert indices_zero.shape == (0,), f"Expected shape (0,), got {indices_zero.shape}"
    assert indices_zero.dtype == torch.long, f"Expected dtype torch.long, got {indices_zero.dtype}"

    # Test with negative topkk
    indices_neg = mask_topk(x, -2)
    expected_neg = torch.tensor([4, 2, 0], dtype=torch.long)
    assert torch.equal(indices_neg, expected_neg), f"Expected {expected_neg}, got {indices_neg}"

    # Test with topkk larger than tensor size
    indices_large = mask_topk(x, 10)
    expected_large = torch.tensor([4, 2, 0, 1, 3], dtype=torch.long)
    assert torch.equal(indices_large, expected_large), f"Expected {expected_large}, got {indices_large}"

    # Test with duplicate values
    x_dup = torch.tensor([1.0, 2.0, 2.0, 3.0])
    indices_dup = mask_topk(x_dup, 3)
    expected_dup = torch.tensor([3, 1, 2], dtype=torch.long)
    assert torch.equal(indices_dup, expected_dup), f"Expected {expected_dup}, got {indices_dup}"

    print("All mask_topk tests passed!")


def test_zero_diagonal():
    """Test the zero_diagonal function with various inputs."""
    # Test 1: Basic 2x2 matrix
    g1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected1 = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    result1 = zero_diagonal(g1)
    assert torch.equal(result1, expected1), f"Expected {expected1}, got {result1}"

    # Test 2: 3x3 matrix
    g2 = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    expected2 = torch.tensor([[0.0, 2.0, 3.0],
                            [4.0, 0.0, 6.0],
                            [7.0, 8.0, 0.0]])
    result2 = zero_diagonal(g2)
    assert torch.equal(result2, expected2), f"Expected {expected2}, got {result2}"

    # Test 3: Empty tensor
    g3 = torch.tensor([])
    result3 = zero_diagonal(g3)
    assert torch.equal(result3, g3), "Empty tensor should return unchanged"

    # Test 4: Tensor with zero dimension
    g4 = torch.zeros((0, 0))
    result4 = zero_diagonal(g4)
    assert torch.equal(result4, g4), "Zero dimension tensor should return unchanged"

    # Test 5: Non-square matrix (should raise ValueError)
    g5 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    try:
        zero_diagonal(g5)
        assert False, "Should have raised ValueError for non-square matrix"
    except ValueError as e:
        assert "Last two dimensions must be equal" in str(e)

    # Test 6: 3D tensor (batch of matrices)
    g6 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]])
    expected6 = torch.tensor([[[0.0, 2.0], [3.0, 0.0]],
                            [[0.0, 6.0], [7.0, 0.0]]])
    result6 = zero_diagonal(g6)
    assert torch.equal(result6, expected6), f"Expected {expected6}, got {result6}"

    # Test 7: Different device
    if torch.cuda.is_available():
        g7 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
        expected7 = torch.tensor([[0.0, 2.0], [3.0, 0.0]], device='cuda')
        result7 = zero_diagonal(g7)
        assert torch.equal(result7, expected7), f"Expected {expected7}, got {result7}"

    print("All zero_diagonal tests passed!")


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
    if topkk < 0:
        # Or raise ValueError, JAX argsort with negative slice might behave differently or error
        # For consistency, let's treat negative as 0 or raise. Here, cap at 0.
        topkk = 0
    if topkk == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    
    num_elements = x.shape[0]
    if num_elements == 0: # If x is empty
        return torch.empty(0, dtype=torch.long, device=x.device)

    # Cap topkk at the number of available elements
    actual_topkk = min(topkk, num_elements)
    
    return torch.argsort(x, descending=True)[:actual_topkk]

def test_squared_norm_pytree():
    """Test the squared_norm_pytree function with simple and nested inputs."""
    # Test 1: Simple tensors
    x1 = torch.tensor([1.0, 2.0, 3.0])
    y1 = torch.tensor([2.0, 3.0, 4.0])
    expected1 = torch.tensor(3.0)  # (1-2)² + (2-3)² + (3-4)² = 1 + 1 + 1 = 3
    result1 = squared_norm_pytree(x1, y1)
    assert torch.allclose(result1, expected1), f"Expected {expected1}, got {result1}"

    # Test 2: Nested structure
    x2 = {
        'a': torch.tensor([1.0, 2.0]),
        'b': torch.tensor([3.0, 4.0])
    }
    y2 = {
        'a': torch.tensor([2.0, 3.0]),
        'b': torch.tensor([4.0, 5.0])
    }
    expected2 = torch.tensor(4.0)  # (1-2)² + (2-3)² + (3-4)² + (4-5)² = 1 + 1 + 1 + 1 = 4
    result2 = squared_norm_pytree(x2, y2)
    assert torch.allclose(result2, expected2), f"Expected {expected2}, got {result2}"

    # Test 3: Empty tensors
    x3 = torch.tensor([])
    y3 = torch.tensor([])
    expected3 = torch.tensor(0.0)
    result3 = squared_norm_pytree(x3, y3)
    assert torch.allclose(result3, expected3), f"Expected {expected3}, got {result3}"

    print("All squared_norm_pytree tests passed!")


def test_slogdet_torch():
    """Test the _slogdet_torch function with various inputs"""
    # Test case 1: Simple 2x2 matrix with all parents
    m = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    parents = torch.tensor([True, True])
    result = _slogdet_torch(m, parents)
    expected = torch.log(torch.tensor(3.0))  # det([[2,1],[1,2]]) = 3
    assert torch.allclose(result, expected)

    # Test case 2: 3x3 matrix with some parents
    m = torch.tensor([[2.0, 1.0, 0.0],
                     [1.0, 2.0, 1.0],
                     [0.0, 1.0, 2.0]])
    parents = torch.tensor([True, False, True])
    result = _slogdet_torch(m, parents)
    # The submatrix should be [[2,0],[0,2]] with det = 4
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(result, expected)


    # Test case 4: Non-square matrix (should raise error)
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    parents = torch.tensor([True, True])
    try:
        _slogdet_torch(m, parents)
        assert False, "Expected ValueError for non-square matrix"
    except ValueError:
        pass

    # Test case 5: Mismatched dimensions (should raise error)
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    parents = torch.tensor([True, True, True])
    try:
        _slogdet_torch(m, parents)
        assert False, "Expected ValueError for mismatched dimensions"
    except ValueError:
        pass

    # Test case 6: Identity matrix with all parents
    n = 3
    m = torch.eye(n)
    parents = torch.ones(n, dtype=torch.bool)
    result = _slogdet_torch(m, parents)
    expected = torch.tensor(0.0)  # log(det(I)) = log(1) = 0
    assert torch.allclose(result, expected)
