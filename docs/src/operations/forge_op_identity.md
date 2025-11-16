# forge.op.Identity

Identity operation.

## Function Signature

```python
forge.op.Identity(name: str, operandA: Tensor, unsqueeze: str, unsqueeze_dim: int) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand

- **unsqueeze** (str) (default: None): If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.
- **unsqueeze_dim** (int) (default: None): The index at where singleton dimenion can be inserted

## Returns

- **result** (Output tensor): Tensor

