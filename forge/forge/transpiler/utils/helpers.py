"""
Common utility functions shared across all frontends.
"""
import torch
from typing import List, Union


def is_constant(value: torch.Tensor) -> bool:
    """Check if tensor is a constant (scalar or single element)."""
    return value.ndim == 0 or value.shape == torch.Size([1])


def is_symmetric_padding(pads: List[int]) -> bool:
    """
    Check if padding parameters are symmetric.
    For pads [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...],
    checks if pad_before_i == pad_after_i for all i.
    """
    if len(pads) % 2 != 0:
        return False
    mid = len(pads) // 2
    return pads[:mid] == pads[mid:]


def extract_padding_for_conv(pads: List[int]) -> Union[tuple, List[int]]:
    """
    Extract padding parameters for Conv layers.
    If symmetric, returns tuple for PyTorch (e.g., (1, 1) for 2D).
    If asymmetric, returns list for padding layer.
    """
    if not pads:
        return 0
    
    if is_symmetric_padding(pads):
        # Return half (e.g., [1,1,1,1] -> (1,1))
        pad_dim = len(pads) // 2
        return tuple(pads[:pad_dim])
    else:
        # Asymmetric padding - return as-is for padding layer
        return pads


def get_selection(indices: Union[torch.Tensor, List[int]], dim: int) -> List:
    """
    Get selection indices for dynamic dimension indexing.
    Example: tensor[get_selection(indices, dim=2)] = values
    """
    if dim < 0:
        raise ValueError("Negative dimension not supported in get_selection")
    
    if isinstance(indices, list):
        indices = torch.tensor(indices)
    
    selection = [slice(None) for _ in range(dim + 1)]
    selection[dim] = indices
    return selection

