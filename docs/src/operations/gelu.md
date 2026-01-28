# forge.op.Gelu

## Overview

GeLU

## Function Signature

```python
forge.op.Gelu(name: str, operandA: Tensor, approximate = 'none') -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **approximate** (`Any`, default: `'none'`): str The gelu approximation algorithm to use: 'none' | 'tanh'. Default: 'none'

## Returns

- **result** (`Tensor`): Tensor Forge tensor

## Mathematical Definition

```
gelu(x) = x * Φ(x)
```

Where Φ(x) is the cumulative distribution function of the standard normal distribution.

For 'tanh' approximation:
```
gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

## Related Operations

- [forge.op.Relu](./relu.md): ReLU activation function
- [forge.op.Sigmoid](./sigmoid.md): Sigmoid activation function
