# forge.op.Squeeze

## Overview

Squeeze tensor manipulation operation

## Function Signature

```python
forge.op.Squeeze(name: str, operandA: Tensor, dim: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.

- **dim** (`int`): Dimension to broadcast

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

- [forge.op.Reshape](./forge_op_reshape.md): Reshape tensor manipulation operation
- [forge.op.Transpose](./forge_op_transpose.md): Transpose tensor manipulation operation
- [forge.op.Unsqueeze](./forge_op_unsqueeze.md): Unsqueeze tensor manipulation operation
- [forge.op.Select](./forge_op_select.md): Select tensor manipulation operation
- [forge.op.Index](./forge_op_index.md): Index tensor manipulation operation

