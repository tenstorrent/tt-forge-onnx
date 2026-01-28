# forge.op.Repeat

## Overview

Repeats this tensor along the specified dimensions.

>>> x = torch.tensor([1, 2, 3])

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

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **repeats** (`List[int]`): repeats parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
