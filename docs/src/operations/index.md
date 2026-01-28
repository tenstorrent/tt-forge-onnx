# forge.op.Index

## Overview

TM

## Function Signature

```python
forge.op.Index(
    name: str,
    operandA: Tensor,
    dim: int,
    start: int,
    stop: int = None,
    stride: int = 1
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to slice

- **start** (`int`): int Starting slice index (inclusive)

- **stop** (`int`, default: `None`): int Stopping slice index (exclusive)

- **stride** (`int`, default: `1`): int Stride amount along that dimension

## Returns

- **result** (`Tensor`): Tensor Forge tensor
