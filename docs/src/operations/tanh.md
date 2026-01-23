# forge.op.Tanh

## Overview

Tanh operation.

## Function Signature

```python
forge.op.Tanh(name: str, operandA: Tensor) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

## Returns

- **result** (`Tensor`): Tensor Forge tensor

## Mathematical Definition

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

The output is always in the range (-1, 1).

## Related Operations

- [forge.op.Sigmoid](./sigmoid.md): Sigmoid activation function
- [forge.op.Relu](./relu.md): ReLU activation function
