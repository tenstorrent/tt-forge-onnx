# forge.op.Resize2d

## Overview

Resizes the spatial dimensions (height and width) of a 2D input tensor using interpolation.

The `Resize2d` operation resizes the height and width dimensions of a 4D input tensor to specified target sizes. This operation is commonly used in computer vision tasks for image resizing, upsampling, and downsampling. It supports two interpolation modes: nearest neighbor and bilinear interpolation.

## Function Signature

```python
forge.op.Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[List[int], Tuple[int, int]],
    mode: str,
    align_corners: bool,
    channel_last: bool
) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor of shape `(N, C, H, W)` (channel-first) or `(N, H, W, C)` (channel-last) where:
  - `N` is the batch size
  - `C` is the number of channels
  - `H` is the input height
  - `W` is the input width

- **sizes** (`Union[List[int], Tuple[int, int]]`): Target output spatial dimensions as `[height, width]` or `(height, width)`. The output tensor will have these exact height and width values.
- **mode** (`str`, default: `'nearest'`): Interpolation mode. Supported values:
  - `'nearest'`: Nearest neighbor interpolation (fast, but may produce aliasing)
  - `'bilinear'`: Bilinear interpolation (smoother results, better for upsampling)
- **align_corners** (`bool`, default: `False`): If `True`, the corner pixels of the input and output tensors are aligned. This parameter only affects bilinear interpolation mode. When `False`, the input and output tensors are aligned by their corner points of the corner pixels, and the sampling points are computed based on the pixel centers.
- **channel_last** (`bool`, default: `False`): If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`.

## Returns

- **result** (`Tensor`): Output tensor with resized spatial dimensions. The output shape is `(N, C, H_out, W_out)` if `channel_last=False` or `(N, H_out, W_out, C)` if `channel_last=True`, where `H_out` and `W_out` are the values specified in the `sizes` parameter. The batch size `N` and number of channels `C` remain unchanged.

## Mathematical Definition

### Nearest Neighbor Interpolation

For nearest neighbor interpolation, each output pixel value is taken from the nearest input pixel:

```
output[i, j] = input[round(i * H_in / H_out), round(j * W_in / W_out)]
```

### Bilinear Interpolation

For bilinear interpolation, each output pixel is computed as a weighted average of the four nearest input pixels:

```
output[i, j] = Σ(weight_k * input[k]) for k in {top-left, top-right, bottom-left, bottom-right}
```

The weights are computed based on the distance from the output pixel to the surrounding input pixels.

## Related Operations

- [forge.op.Resize1d](./forge_op_resize1d.md): Resize 1D tensors (e.g., sequences)
- [forge.op.Upsample2d](./forge_op_upsample2d.md): Upsample using scale factors instead of target sizes
- [forge.op.Downsample2d](./forge_op_downsample2d.md): Downsample operation
- [forge.op.Transpose](./forge_op_transpose.md): Rearrange tensor dimensions

