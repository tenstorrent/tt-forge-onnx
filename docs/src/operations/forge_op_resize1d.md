# forge.op.Resize1d

## Overview

Resizes the spatial dimension of a 1D input tensor using interpolation. This operation is commonly used for sequence resizing and temporal dimension manipulation.

## Function Signature

```python
forge.op.Resize1d(
    name: str,
    operandA: Tensor,
    size: int,
    mode: str,
    align_corners: bool,
    channel_last: bool
) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor of shape `(N, C, L)` (channel-first) or `(N, L, C)` (channel-last) where:
  - `N` is the batch size
  - `C` is the number of channels
  - `L` is the input length

- **size** (`int`): The target size to extrapolate
- **mode** (`str`, default: `'nearest'`): Interpolation mode. Supported values:
  - `'nearest'`: Nearest neighbor interpolation (fast, but may produce aliasing)
  - `'bilinear'`: Bilinear interpolation (smoother results, better for upsampling)
- **align_corners** (`bool`, default: `False`): If `True`, the corner pixels of the input and output tensors are aligned. This parameter only affects bilinear interpolation mode. When `False`, the input and output tensors are aligned by their corner points of the corner pixels, and the sampling points are computed based on the pixel centers.
- **channel_last** (`bool`, default: `False`): If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`.

## Returns

- **result** (`Tensor`): Output tensor with resized spatial dimension. The output shape preserves the batch and channel dimensions while modifying the spatial dimension according to the `sizes` parameter.

## Related Operations

- [forge.op.Resize2d](./forge_op_resize2d.md): Resize 2D tensors
- [forge.op.Upsample2d](./forge_op_upsample2d.md): Upsample operation

