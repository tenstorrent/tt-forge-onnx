# forge.op.Select

## Overview

Select tensor manipulation operation

## Function Signature

```python
forge.op.Select(name: str, operandA: Tensor, dim: int, index: Union[int, Tuple[int, int]], stride: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **dim** (`int`): Dimension to slice
- **index** (`Union[int, Tuple[int, int]]`): int: Index to select from that dimension [start: int, length: int]: Index range to select from that dimension
- **stride** (`int`, default: `0`): Stride amount along that dimension

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

