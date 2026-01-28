# forge.op.PixelShuffle

## Overview

Pixel shuffle operation.

## Function Signature

```python
forge.op.PixelShuffle(
    name: str,
    operandA: Tensor,
    upscale_factor: int
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **upscale_factor** (`int`): upscale_factor parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
