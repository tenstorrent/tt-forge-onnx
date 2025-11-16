# forge.op.Argmax

Argmax

## Function Signature

```python
forge.op.Argmax(name: str, operandA: Tensor, dim: int, keep_dim) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand

- **dim** (int) (default: None): The dimension to reduce (if None, the output is the argmax of the whole tensor)
- **keep_dim** (Any) (default: False): If True, retains the dimension that is reduced, with size 1. If False (default), the dimension is removed from the output shape.

## Returns

- **result** (Output tensor): Tensor

