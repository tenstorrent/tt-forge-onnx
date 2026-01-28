# forge.op.Relu

## Overview

Applies the Rectified Linear Unit (ReLU) activation function elementwise.

ReLU sets all negative values to zero while keeping positive values

unchanged. This introduces non-linearity to neural networks and is one

of the most commonly used activation functions due to its simplicity

and effectiveness.

## Function Signature

```python
forge.op.Relu(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Tensor Input tensor of any shape. The ReLU function is applied independently to each element.

## Returns

- **result** (`Tensor`): Tensor Output tensor with same shape as input. Each element is max(0, x) where x is the corresponding input element.

## Mathematical Definition

```
relu(x) = max(0, x) = { x if x > 0, 0 if x ≤ 0 }
```

## Related Operations

- [forge.op.LeakyRelu](./leakyrelu.md): Leaky ReLU with non-zero negative slope
- [forge.op.Gelu](./gelu.md): Gaussian Error Linear Unit
- [forge.op.Sigmoid](./sigmoid.md): Sigmoid activation function
- [forge.op.Tanh](./tanh.md): Hyperbolic tangent activation
