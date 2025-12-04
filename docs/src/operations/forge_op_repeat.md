# forge.op.Repeat

## Overview

Repeats this tensor along the specified dimensions. >>> x = torch.tensor([1, 2, 3])

>>> x.repeat(4, 2)

tensor([[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3]])

NOTE:

This Forge.Repeat is equivalent to torch.repeat, numpy.tile, tvm.tile, and ttnn.repeat

## Function Signature

```python
forge.op.Repeat(name: str, operandA: Tensor, repeats: List[int]) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **repeats** (`List[int]`): repeats parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

