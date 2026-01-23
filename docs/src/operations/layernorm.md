# forge.op.Layernorm

## Overview

Layer normalization.

## Function Signature

```python
forge.op.Layernorm(
    name: str,
    operandA: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Union[(Tensor, Parameter)],
    dim: int = -1,
    epsilon: float = 1e-05
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **weights** (`Union[(Tensor, Parameter)]`): weights tensor

- **bias** (`Union[(Tensor, Parameter)]`): bias tensor

- **dim** (`int`, default: `-1`): dim parameter

- **epsilon** (`float`, default: `1e-05`): epsilon parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
