# forge.op.Transpose

## Overview

Tranpose X and Y (i.e. rows and columns) dimensions.

## Function Signature

```python
forge.op.Transpose(name: str, operandA: Tensor, dim0: int, dim1: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand

- **dim0** (`int`): dim0 parameter
- **dim1** (`int`): dim1 parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

- [forge.op.Reshape](./forge_op_reshape.md): Reshape tensor manipulation operation
- [forge.op.Squeeze](./forge_op_squeeze.md): Squeeze tensor manipulation operation
- [forge.op.Unsqueeze](./forge_op_unsqueeze.md): Unsqueeze tensor manipulation operation
- [forge.op.Select](./forge_op_select.md): Select tensor manipulation operation
- [forge.op.Index](./forge_op_index.md): Index tensor manipulation operation

