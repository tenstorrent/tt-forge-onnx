# forge.op.Upsample2d

## Overview

Upsample 2D operation

## Function Signature

```python
forge.op.Upsample2d(
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

- **scale_factor** (`Union[(int, List[int], Tuple[(int, int)])]`): Union[int, List[int], Tuple[int, int]] multiplier for spatial size.

- **mode** (`str`, default: `'nearest'`): str the upsampling algorithm

- **channel_last** (`bool`, default: `False`): channel_last parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
