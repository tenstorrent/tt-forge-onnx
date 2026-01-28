# forge.op.Abs

## Overview

Computes the elementwise absolute value of the input tensor.

The Abs operation returns the magnitude of each element without regard

to its sign. For real numbers, it returns the non-negative value.

This operation is idempotent: abs(abs(x)) = abs(x).

## Function Signature

```python
forge.op.Abs(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Tensor Input tensor of any shape. All elements will have absolute value computed independently.

## Returns

- **result** (`Tensor`): Tensor Output tensor with same shape as input. Each element is the absolute value of the corresponding input element.

## Mathematical Definition

```
abs(x) = |x| = { x if x ≥ 0, -x if x < 0 }
```

## Related Operations

- [forge.op.Relu](./relu.md): ReLU activation (sets negatives to zero)
- [forge.op.Sigmoid](./sigmoid.md): Sigmoid activation function
