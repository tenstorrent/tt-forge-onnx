# forge.op.Resize1d

## Overview

Resize input activations, with default mode 'nearest'

## Function Signature

```python
forge.op.Resize1d(
    name: str,
    operandA: Tensor,
    size: int,
    mode: str = 'nearest',
    align_corners: bool = False,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **size** (`int`): int The target size to extrapolate

- **mode** (`str`, default: `'nearest'`): str Interpolation mode

- **align_corners** (`bool`, default: `False`): align_corners parameter

- **channel_last** (`bool`, default: `False`): bool Whether the input is in channel-last format (NWC)

## Returns

- **result** (`Tensor`): Output tensor

## Related Operations

- [forge.op.Resize2d](./resize2d.md): Resize 2D tensors
- [forge.op.Upsample2d](./upsample2d.md): Upsample operation
