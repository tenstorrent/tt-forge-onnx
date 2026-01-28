# forge.op.RepeatInterleave

## Overview

Repeat elements of a tensor.

>>> x = torch.tensor([1, 2, 3])

>>> x.repeat_interleave(2)

tensor([1, 1, 2, 2, 3, 3])

NOTE:

This Forge.RepeatInterleave is equivalent to torch.repeat_interleave, numpy.repeat, tvm.repeat, and ttnn.repeat_interleave

## Function Signature

```python
forge.op.RepeatInterleave(
    name: str,
    operandA: Tensor,
    repeats: int,
    dim: int
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **repeats** (`int`): int The number of repetitions for each element.

- **dim** (`int`): int The dimension along which to repeat values.

## Returns

- **result** (`Tensor`): Tensor Forge tensor
