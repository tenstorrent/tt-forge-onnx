# forge.op.AdvIndex

## Overview

TM

## Function Signature

```python
forge.op.AdvIndex(
    name: str,
    operandA: Tensor,
    operandB: Tensor,
    dim: int = 0
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand B - indices

- **operandB** (`Tensor`): operandB tensor

- **dim** (`int`, default: `0`): int Dimension to fetch indices over

## Returns

- **result** (`Tensor`): Tensor Forge tensor
