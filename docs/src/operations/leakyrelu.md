# forge.op.LeakyRelu

## Overview

Leaky ReLU

## Function Signature

```python
forge.op.LeakyRelu(name: str, operandA: Tensor, alpha: float) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **alpha** (`float`): float Controls the angle of the negative slope

## Returns

- **result** (`Tensor`): Tensor Forge tensor
