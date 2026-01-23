# forge.op.GreaterEqual

## Overview

Elementwise greater or equal of two tensors

## Function Signature

```python
forge.op.GreaterEqual(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

## Returns

- **result** (`Tensor`): Tensor Forge tensor
