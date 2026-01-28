# forge.op.Transpose

## Overview

Tranpose X and Y (i.e. rows and columns) dimensions.

## Function Signature

```python
forge.op.Transpose(name: str, operandA: Tensor, dim0: int, dim1: int) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim0** (`int`): dim0 parameter

- **dim1** (`int`): dim1 parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
