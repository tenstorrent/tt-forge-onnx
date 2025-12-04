# forge.op.Index

## Overview

Index tensor manipulation operation

## Function Signature

```python
forge.op.Index(name: str, operandA: Tensor, dim: int, start: int, stop: int, stride: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **dim** (`int`): Dimension to slice
- **start** (`int`): Starting slice index (inclusive)
- **stop** (`int`, default: `None`): Stopping slice index (exclusive)
- **stride** (`int`, default: `1`): Stride amount along that dimension

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

