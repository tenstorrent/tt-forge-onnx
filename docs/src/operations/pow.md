# forge.op.Pow

## Overview

Pow operation: `operandA` to the power of `exponent`.

yi = pow(xi, exponent) for all xi in operandA tensor

## Function Signature

```python
forge.op.Pow(
    name: str,
    operandA: Tensor,
    exponent: Union[(int, float)]
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **exponent** (`Union[(int, float)]`): exponent parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
