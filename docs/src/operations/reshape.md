# forge.op.Reshape

## Overview

TM

## Function Signature

```python
forge.op.Reshape(
    name: str,
    operandA: Tensor,
    shape: Tuple[(int, Ellipsis)]
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **shape** (`Tuple[(int, Ellipsis)]`): shape parameter

## Returns

- **result** (`Tensor`): Tensor Forge tensor
