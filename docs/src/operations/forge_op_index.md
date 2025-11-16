# forge.op.Index

TM

## Function Signature

```python
forge.op.Index(name: str, operandA: Tensor, dim: int, start: int, stop: int, stride: int) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **dim** (int): Dimension to slice
- **start** (int): Starting slice index (inclusive)
- **stop** (int) (default: None): Stopping slice index (exclusive)
- **stride** (int) (default: 1): Stride amount along that dimension

## Returns

- **result** (Output tensor): Tensor

