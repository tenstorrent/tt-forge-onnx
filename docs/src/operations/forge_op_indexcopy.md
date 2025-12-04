# forge.op.IndexCopy

## Overview

Copies the elements of value into operandA at index along dim

## Function Signature

```python
forge.op.IndexCopy(name: str, operandA: Tensor, index: Tensor, value: Tensor, dim: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.
- **index** (`Tensor`): Index at which to write into operandA
- **value** (`Tensor`): Value to write out

- **dim** (`int`): Dimension to broadcast

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

