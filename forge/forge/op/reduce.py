# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Union

from forge._C.ops import OpType
from ..tensor import Tensor
from .common import ForgeOp as op


def _normalize_dims(dim: Union[int, List[int], Tuple[int, ...]], rank: int = 4) -> List[int]:
    """Normalize dim argument to a list of ints, each validated to be in [-rank, rank-1]."""
    dims: List[int] = [dim] if isinstance(dim, int) else list(dim)
    for d in dims:
        assert (d >= -rank) and (d <= rank - 1), f"dim {d} out of range for rank-{rank} tensor"
    return dims


def ReduceSum(
    name: str,
    operandA: Tensor,
    dim: Union[int, List[int], Tuple[int, ...]],
    keep_dim: bool = True,
) -> Tensor:
    """
    Reduce by summing along one or more dimensions.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int or list/tuple of ints
        Dimension(s) along which to reduce.  Each value must be in [-4, 3].

    keep_dim: bool
        If True, the reduced dimensions are kept with size 1.

    Returns
    -------
    Tensor
        Forge tensor
    """
    dims = _normalize_dims(dim)
    return op(OpType.ReduceSum, name, operandA, dim_arg=dims, keep_dim=keep_dim).get_tensor()


def ReduceAvg(
    name: str,
    operandA: Tensor,
    dim: Union[int, List[int], Tuple[int, ...]],
    keep_dim: bool = True,
) -> Tensor:
    """
    Reduce by averaging along one or more dimensions.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int or list/tuple of ints
        Dimension(s) along which to reduce.  Each value must be in [-4, 3].

    keep_dim: bool
        If True, the reduced dimensions are kept with size 1.

    Returns
    -------
    Tensor
        Forge tensor
    """
    dims = _normalize_dims(dim)
    return op(OpType.ReduceAvg, name, operandA, dim_arg=dims, keep_dim=keep_dim).get_tensor()


def ReduceMax(
    name: str,
    operandA: Tensor,
    dim: Union[int, List[int], Tuple[int, ...]],
    keep_dim: bool = True,
) -> Tensor:
    """
    Reduce by taking the maximum along one or more dimensions.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int or list/tuple of ints
        Dimension(s) along which to reduce.  Each value must be in [-4, 3].

    keep_dim: bool
        If True, the reduced dimensions are kept with size 1.

    Returns
    -------
    Tensor
        Forge tensor
    """
    dims = _normalize_dims(dim)
    return op(OpType.ReduceMax, name, operandA, dim_arg=dims, keep_dim=keep_dim).get_tensor()


def Argmax(name: str, operandA: Tensor, dim: int = None, keep_dim=False) -> Tensor:
    """
    Argmax

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        The dimension to reduce (if None, the output is the argmax of the whole tensor)

    keep_dim: bool
        If True, retains the dimension that is reduced, with size 1.
        If False (default), the dimension is removed from the output shape.

    Returns
    -------
    Tensor
        Forge tensor
    """

    kwargs = {"keep_dim": keep_dim}

    if dim is not None:
        if dim < 0:
            dim += len(operandA.shape)
        kwargs["dim_arg"] = [dim]

    return op(OpType.Argmax, name, operandA, **kwargs).get_tensor()
