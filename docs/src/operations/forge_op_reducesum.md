# forge.op.ReduceSum

## Overview

Reduce by summing along the given dimension

## Function Signature

```python
forge.op.ReduceSum(name: str, operandA: Tensor, dim: int, keep_dim: bool) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand

- **dim** (`int`): Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.
- **keep_dim** (`bool`, default: `True`): keep_dim parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

