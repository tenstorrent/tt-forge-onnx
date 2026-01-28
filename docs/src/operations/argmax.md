# forge.op.Argmax

## Overview

Argmax

## Function Signature

```python
forge.op.Argmax(
    name: str,
    operandA: Tensor,
    dim: int = None,
    keep_dim = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`, default: `None`): int The dimension to reduce (if None, the output is the argmax of the whole tensor)

- **keep_dim** (`Any`, default: `False`): bool If True, retains the dimension that is reduced, with size 1. If False (default), the dimension is removed from the output shape.

## Returns

- **result** (`Tensor`): Tensor Forge tensor
