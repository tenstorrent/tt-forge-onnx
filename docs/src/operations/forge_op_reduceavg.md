# forge.op.ReduceAvg

Reduce by averaging along the given dimension

## Function Signature

```python
forge.op.ReduceAvg(name: str, operandA: Tensor, dim: int, keep_dim: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand

- **dim** (int): Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.
- **keep_dim** (bool) (default: True): keep_dim parameter

## Returns

- **result** (Output tensor): Tensor

