# forge.op.Relu

## Overview

Applies the Rectified Linear Unit (ReLU) activation function elementwise. ReLU sets all negative values to zero while keeping positive values unchanged, introducing non-linearity to neural networks.

## Function Signature

```python
forge.op.Relu(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
## Returns

- **result** (`Tensor`): Output tensor with the same shape as the input. Each element is the result of applying the relu function to the corresponding input element.

## Related Operations

- [forge.op.Sigmoid](./forge_op_sigmoid.md): Sigmoid activation function
- [forge.op.Tanh](./forge_op_tanh.md): Tanh activation function
- [forge.op.Gelu](./forge_op_gelu.md): Gelu activation function
- [forge.op.Leakyrelu](./forge_op_leakyrelu.md): Leakyrelu activation function

