# forge.op.ReduceSum

Reduce by summing along the given dimension

## Function Signature

```python
forge.op.ReduceSum(name: str, operandA: Tensor, dim: int, keep_dim: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand

- **dim** (int): Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.
- **keep_dim** (bool) (default: True): keep_dim parameter

## Returns

- **result** (Output tensor): Tensor

