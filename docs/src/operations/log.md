# forge.op.Log

## Overview

Log operation: natural logarithm of the elements of `operandA`.

yi = log_e(xi) for all xi in operandA tensor

## Function Signature

```python
forge.op.Log(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

## Returns

- **result** (`Tensor`): Tensor Forge tensor
