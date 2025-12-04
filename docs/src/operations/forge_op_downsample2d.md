# forge.op.Downsample2d

## Overview

Downsample 2D operation

## Function Signature

```python
forge.op.Downsample2d(name: str, operandA: Tensor, scale_factor: Union[int, List[int], Tuple[int, int]], mode: str, channel_last: bool) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **scale_factor** (`Union[int, List[int], Tuple[int, int]]`): Union[int, List[int], Tuple[int, int]] Divider for spatial size.
- **mode** (`str`, default: `'nearest'`): The downsampling algorithm
- **channel_last** (`bool`, default: `False`): If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`.

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

