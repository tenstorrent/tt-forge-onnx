# forge.op.Dropout

## Overview

Dropout

## Function Signature

```python
forge.op.Dropout(
    name: str,
    operandA: Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int = 0
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **p** (`float`, default: `0.5`): float Probability of an element to be zeroed.

- **training** (`bool`, default: `True`): bool Apply dropout if true

- **seed** (`int`, default: `0`): int RNG seed

## Returns

- **result** (`Tensor`): Tensor Forge tensor
