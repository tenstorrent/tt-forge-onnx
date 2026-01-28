# forge.op.Resize2d

## Overview

Resizes the spatial dimensions (height and width) of a 2D input tensor using interpolation. This operation is commonly used in computer vision tasks for image resizing, upsampling, and downsampling.

## Function Signature

```python
forge.op.Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[(List[int], Tuple[(int, int)])],
    mode: str = 'nearest',
    align_corners: bool = False,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Input tensor of shape `(N, C, H, W)` for channel-first or `(N, H, W, C)` for channel-last format.

- **sizes** (`Union[(List[int], Tuple[(int, int)])]`): Target output spatial dimensions as `[height, width]`. The output tensor will have these exact height and width values.

- **mode** (`str`, default: `'nearest'`): Interpolation mode: `'nearest'` for nearest neighbor (fast) or `'bilinear'` for bilinear interpolation (smoother).

- **align_corners** (`bool`, default: `False`): If `True`, corner pixels are aligned. Only affects bilinear mode.

- **channel_last** (`bool`, default: `False`): If `True`, input is `(N, H, W, C)` format; if `False`, input is `(N, C, H, W)` format.

## Returns

- **result** (`Tensor`): Tensor Output tensor with resized spatial dimensions: - Shape `(N, C, H_out, W_out)` if `channel_last=False` - Shape `(N, H_out, W_out, C)` if `channel_last=True` where `H_out, W_out` are the values specified in `sizes`.

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

- [forge.op.Resize1d](./resize1d.md): Resize 1D tensors (e.g., sequences)
- [forge.op.Upsample2d](./upsample2d.md): Upsample using scale factors instead of target sizes
- [forge.op.Downsample2d](./downsample2d.md): Downsample operation
- [forge.op.Transpose](./transpose.md): Rearrange tensor dimensions
