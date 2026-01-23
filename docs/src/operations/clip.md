# forge.op.Clip

## Overview

Clips tensor values between min and max

## Function Signature

```python
forge.op.Clip(name: str, operandA: Tensor, min: float, max: float) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **min** (`float`): float Minimum value

- **max** (`float`): float Maximum value

## Returns

- **result** (`Tensor`): Tensor Forge tensor
