# forge.op.Sigmoid

## Overview

Sigmoid

## Function Signature

```python
forge.op.Sigmoid(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

## Returns

- **result** (`Tensor`): Tensor Forge tensor

## Mathematical Definition

```
sigmoid(x) = 1 / (1 + exp(-x))
```

The output is always in the range (0, 1).

## Related Operations

- [forge.op.Relu](./relu.md): ReLU activation function
- [forge.op.Tanh](./tanh.md): Hyperbolic tangent activation
- [forge.op.Gelu](./gelu.md): Gaussian Error Linear Unit
