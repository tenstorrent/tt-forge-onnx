# forge.op.Reshape

## Overview

Reshapes a tensor to new dimensions while preserving the total number of elements. The operation changes the tensor's shape without modifying its data.

## Function Signature

```python
forge.op.Reshape(name: str, operandA: Tensor, shape: Tuple[int, ...]) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **shape** (`Tuple[int, ...]`): shape parameter

## Returns

- **result** (`Tensor`): Output tensor with the new shape. The total number of elements remains the same as the input.

## Related Operations

- [forge.op.Transpose](./forge_op_transpose.md): Transpose tensor manipulation operation
- [forge.op.Squeeze](./forge_op_squeeze.md): Squeeze tensor manipulation operation
- [forge.op.Unsqueeze](./forge_op_unsqueeze.md): Unsqueeze tensor manipulation operation
- [forge.op.Select](./forge_op_select.md): Select tensor manipulation operation
- [forge.op.Index](./forge_op_index.md): Index tensor manipulation operation

