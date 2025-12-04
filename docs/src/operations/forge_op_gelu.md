# forge.op.Gelu

## Overview

GeLU

## Function Signature

```python
forge.op.Gelu(name: str, operandA: Tensor, approximate) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand

- **approximate** (`Any`, default: `'none'`): The gelu approximation algorithm to use: 'none' | 'tanh'. Default: 'none'

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

- [forge.op.Relu](./forge_op_relu.md): Relu activation function
- [forge.op.Sigmoid](./forge_op_sigmoid.md): Sigmoid activation function
- [forge.op.Tanh](./forge_op_tanh.md): Tanh activation function
- [forge.op.Leakyrelu](./forge_op_leakyrelu.md): Leakyrelu activation function

