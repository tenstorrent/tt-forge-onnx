# forge.op.Downsample2d

## Overview

Downsample 2D operation

## Function Signature

```python
forge.op.Downsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[(int, List[int], Tuple[(int, int)])],
    mode: str = 'nearest',
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **scale_factor** (`Union[(int, List[int], Tuple[(int, int)])]`): Union[int, List[int], Tuple[int, int]] Divider for spatial size.

- **mode** (`str`, default: `'nearest'`): str The downsampling algorithm

- **channel_last** (`bool`, default: `False`): bool Whether the input is in channel-last format (NHWC)

## Returns

- **result** (`Tensor`): Tensor Forge tensor
