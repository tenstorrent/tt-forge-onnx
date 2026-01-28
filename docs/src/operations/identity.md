# forge.op.Identity

## Overview

Identity operation.

## Function Signature

```python
forge.op.Identity(
    name: str,
    operandA: Tensor,
    unsqueeze: str = None,
    unsqueeze_dim: int = None
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **unsqueeze** (`str`, default: `None`): str If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.

- **unsqueeze_dim** (`int`, default: `None`): int The index at where singleton dimenion can be inserted

## Returns

- **result** (`Tensor`): Tensor Forge tensor
