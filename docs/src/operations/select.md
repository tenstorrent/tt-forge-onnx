# forge.op.Select

## Overview

TM

## Function Signature

```python
forge.op.Select(
    name: str,
    operandA: Tensor,
    dim: int,
    index: Union[(int, Tuple[(int, int)])],
    stride: int = 0
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to slice

- **index** (`Union[(int, Tuple[(int, int)])]`): int int: Index to select from that dimension [start: int, length: int]: Index range to select from that dimension

- **stride** (`int`, default: `0`): int Stride amount along that dimension

## Returns

- **result** (`Tensor`): Tensor Forge tensor
