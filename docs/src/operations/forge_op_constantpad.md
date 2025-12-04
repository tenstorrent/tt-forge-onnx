# forge.op.ConstantPad

## Overview

Applies constant padding to the input tensor. This is a low-level padding operation that directly specifies padding values for each dimension in Forge format.

## Function Signature

```python
forge.op.ConstantPad(name: str, operandA: Tensor, padding: List[int], value: float) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input operand A to which padding will be applied.

- **padding** (`List[int]`): Padding values in Forge format: `[dim0_low, dim0_high, dim1_low, dim1_high, ...]`. Length must be 2 * rank of input tensor. Each dimension has a low and high padding value.
- **value** (`float`, default: `0.0`): float, optional The constant value to use for padding. Default is 0.0.

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

