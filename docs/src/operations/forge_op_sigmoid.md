# forge.op.Sigmoid

## Overview

Applies the sigmoid activation function elementwise. The sigmoid function maps input values to the range (0, 1), making it useful for binary classification and probability outputs.

## Function Signature

```python
forge.op.Sigmoid(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
## Returns

- **result** (`Tensor`): Output tensor with the same shape as the input. Each element is the result of applying the sigmoid function to the corresponding input element.

## Related Operations

- [forge.op.Relu](./forge_op_relu.md): Relu activation function
- [forge.op.Tanh](./forge_op_tanh.md): Tanh activation function
- [forge.op.Gelu](./forge_op_gelu.md): Gelu activation function
- [forge.op.Leakyrelu](./forge_op_leakyrelu.md): Leakyrelu activation function

