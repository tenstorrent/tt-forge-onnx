# forge.op.Select

TM

## Function Signature

```python
forge.op.Select(name: str, operandA: Tensor, dim: int, index: Union[int, Tuple[int, int]], stride: int) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **dim** (int): Dimension to slice
- **index** (Union[int, Tuple[int, int]]): int: Index to select from that dimension [start: int, length: int]: Index range to select from that dimension
- **stride** (int) (default: 0): Stride amount along that dimension

## Returns

- **result** (Output tensor): Tensor

