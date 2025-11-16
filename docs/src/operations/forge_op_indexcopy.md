# forge.op.IndexCopy

Copies the elements of value into operandA at index along dim

## Function Signature

```python
forge.op.IndexCopy(name: str, operandA: Tensor, index: Tensor, value: Tensor, dim: int) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A
- **index** (Tensor): Index at which to write into operandA
- **value** (Tensor): Value to write out

- **dim** (int): Dimension to broadcast

## Returns

- **result** (Output tensor): Tensor

