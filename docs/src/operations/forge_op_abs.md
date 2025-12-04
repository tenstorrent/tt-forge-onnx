# forge.op.Abs

## Overview

Computes the elementwise absolute value of the input tensor. Each output element is the absolute value of the corresponding input element.

## Function Signature

```python
forge.op.Abs(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
## Returns

- **result** (`Tensor`): Output tensor with the same shape as the input. Each element is the result of applying the abs function to the corresponding input element.

## Related Operations

- [forge.op.Relu](./forge_op_relu.md): Relu activation function
- [forge.op.Sigmoid](./forge_op_sigmoid.md): Sigmoid activation function
- [forge.op.Tanh](./forge_op_tanh.md): Tanh activation function
- [forge.op.Gelu](./forge_op_gelu.md): Gelu activation function
- [forge.op.Leakyrelu](./forge_op_leakyrelu.md): Leakyrelu activation function

