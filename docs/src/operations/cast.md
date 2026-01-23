# forge.op.Cast

## Overview

Cast

## Function Signature

```python
forge.op.Cast(
    name: str,
    operandA: Tensor,
    dtype: Union[(torch.dtype, DataFormat)]
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dtype** (`Union[(torch.dtype, DataFormat)]`): Union[torch.dtype, DataFormat] Specify Torch datatype / Forge DataFormat to convert operandA

## Returns

- **result** (`Tensor`): Tensor Forge tensor
