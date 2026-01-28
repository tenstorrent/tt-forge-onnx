# forge.op.Broadcast

## Overview

TM

## Function Signature

```python
forge.op.Broadcast(name: str, operandA: Tensor, dim: int, shape: int) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to broadcast

- **shape** (`int`): int Output length of dim

## Returns

- **result** (`Tensor`): Tensor Forge tensor
