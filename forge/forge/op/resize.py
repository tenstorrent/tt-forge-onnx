# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Union

from forge._C.ops import OpType
from ..tensor import Tensor
from .common import ForgeOp as op


def Resize1d(
    name: str,
    operandA: Tensor,
    size: int,
    mode: str = "nearest",
    align_corners: bool = False,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default mode 'nearest'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    size: int
        The target size to extrapolate

    mode: str
        Interpolation mode

    channel_last: bool
        Whether the input is in channel-last format (NWC)

    """

    assert isinstance(size, int)
    assert mode in ["nearest", "linear"], "Only support nearest and linear mode for now"

    result: Tensor = op(
        OpType.Resize1d,
        name,
        operandA,
        size=size,
        mode=mode,
        align_corners=align_corners,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[List[int], Tuple[int, int]],
    mode: str = "nearest",
    align_corners: bool = False,
    channel_last: bool = False,
) -> Tensor:
    """
    Resizes the spatial dimensions of a 2D input tensor using interpolation.

    The Resize2d operation resizes the height and width dimensions of a 4D
    input tensor to specified target sizes. This operation is commonly used
    in computer vision tasks for image resizing, upsampling, and downsampling.

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.
        Use empty string to auto-generate.

    operandA : Tensor
        Input tensor of shape `(N, C, H, W)` (channel-first) or
        `(N, H, W, C)` (channel-last) where:
        - `N` is the batch size
        - `C` is the number of channels
        - `H` is the input height
        - `W` is the input width

    sizes : Union[List[int], Tuple[int, int]]
        Target output spatial dimensions as `[height, width]` or
        `(height, width)`. The output tensor will have these exact
        height and width values.

    mode : str, optional
        Interpolation mode. Supported values:
        - `'nearest'`: Nearest neighbor interpolation (fast, may produce aliasing)
        - `'bilinear'`: Bilinear interpolation (smoother, better for upsampling)
        Default: `'nearest'`

    align_corners : bool, optional
        If True, align corner pixels of input and output tensors.
        Only affects bilinear mode. When False, pixels are aligned by centers.
        Default: `False`

    channel_last : bool, optional
        If True, input is in channel-last format `(N, H, W, C)`.
        If False, input is in channel-first format `(N, C, H, W)`.
        Default: `False`

    Returns
    -------
    Tensor
        Output tensor with resized spatial dimensions:
        - Shape `(N, C, H_out, W_out)` if `channel_last=False`
        - Shape `(N, H_out, W_out, C)` if `channel_last=True`
        where `H_out, W_out` are the values specified in `sizes`.

    See Also
    --------
    forge.op.Resize1d : Resize 1D tensors
    forge.op.Upsample2d : Upsample using scale factors
    forge.op.Downsample2d : Downsample operation
    """

    assert isinstance(sizes, (list, tuple)) and len(sizes) == 2
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear mode for now"

    result: Tensor = op(
        OpType.Resize2d,
        name,
        operandA,
        sizes=sizes,
        mode=mode,
        align_corners=align_corners,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Upsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[int, List[int], Tuple[int, int]],
    mode: str = "nearest",
    channel_last: bool = False,
) -> Tensor:
    """
    Upsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: Union[int, List[int], Tuple[int, int]]
        multiplier for spatial size.

    mode: str
        the upsampling algorithm

    Returns
    -------
    Tensor
        Forge tensor
    """

    assert (isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2) or isinstance(
        scale_factor, int
    ), f"Only support List/Tuple of int (or) int scale_factor but provided {type(scale_factor)}"
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear interpolation for now"

    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)

    result: Tensor = op(
        OpType.Upsample2d,
        name,
        operandA,
        scale_factor=scale_factor,
        mode=mode,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Downsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[int, List[int], Tuple[int, int]],
    mode: str = "nearest",
    channel_last: bool = False,
) -> Tensor:
    """
    Downsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: Union[int, List[int], Tuple[int, int]]
        Divider for spatial size.

    mode: str
        The downsampling algorithm

    channel_last: bool
        Whether the input is in channel-last format (NHWC)

    Returns
    -------
    Tensor
        Forge tensor
    """

    assert (isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2) or isinstance(
        scale_factor, int
    ), f"Only support List/Tuple of int (or) int scale_factor but provided {type(scale_factor)}"
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear interpolation for now"

    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)

    result: Tensor = op(
        OpType.Downsample2d,
        name,
        operandA,
        scale_factor=scale_factor,
        mode=mode,
        channel_last=channel_last,
    ).get_tensor()

    return result
