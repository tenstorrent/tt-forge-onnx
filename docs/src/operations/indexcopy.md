# forge.op.IndexCopy

## Overview

Copies the elements of value into operandA at index along dim

## Function Signature

```python
forge.op.IndexCopy(
    name: str,
    operandA: Tensor,
    index: Tensor,
    value: Tensor,
    dim: int
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **index** (`Tensor`): Tensor Index at which to write into operandA

- **value** (`Tensor`): Tensor Value to write out

- **dim** (`int`): int Dimension to broadcast

## Returns

- **result** (`Tensor`): Tensor Forge tensor
