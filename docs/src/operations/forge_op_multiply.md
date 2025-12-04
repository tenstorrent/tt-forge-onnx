# forge.op.Multiply

## Overview

Performs elementwise multiply operation on two input tensors. The operation is applied element-by-element, requiring both tensors to be broadcastable to the same shape.

## Function Signature

```python
forge.op.Multiply(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
- **operandB** (`Union[Tensor, Parameter]`): Second input tensor. Must be broadcastable with operandA.
## Returns

- **result** (`Tensor`): Output tensor with the same shape as the broadcasted input tensors. Each element is the result of the elementwise operation.

## Related Operations

- [forge.op.Add](./forge_op_add.md): Elementwise add operation
- [forge.op.Subtract](./forge_op_subtract.md): Elementwise subtract operation
- [forge.op.Divide](./forge_op_divide.md): Elementwise divide operation
- [forge.op.Max](./forge_op_max.md): Elementwise max operation
- [forge.op.Min](./forge_op_min.md): Elementwise min operation

