# forge.op.ReduceMax

## Overview

Reduce by taking maximum along the given dimension

## Function Signature

```python
forge.op.ReduceMax(
    name: str,
    operandA: Tensor,
    dim: int,
    keep_dim: bool = True
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`): int Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

- **keep_dim** (`bool`, default: `True`): keep_dim parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
