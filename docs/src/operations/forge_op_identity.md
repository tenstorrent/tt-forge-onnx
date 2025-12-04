# forge.op.Identity

## Overview

Identity operation.

## Function Signature

```python
forge.op.Identity(name: str, operandA: Tensor, unsqueeze: str, unsqueeze_dim: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand

- **unsqueeze** (`str`, default: `None`): If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.
- **unsqueeze_dim** (`int`, default: `None`): The index at where singleton dimenion can be inserted

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

