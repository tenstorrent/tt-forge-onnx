# forge.op.Pad

## Overview

TM

## Function Signature

```python
forge.op.Pad(
    name: str,
    operandA: Tensor,
    pad: Tuple[(int, Ellipsis)],
    mode: str = 'constant',
    value: float = 0.0,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A to which padding will be applied.

- **pad** (`Tuple[(int, Ellipsis)]`): Tuple[int, ...] A tuple of padding values. The tuple should correspond to padding values for the tensor, such as [left, right, top, bottom].

- **mode** (`str`, default: `'constant'`): str, optional The padding mode. Default is "constant". Other modes can be supported depending on the implementation (e.g., "reflect", "replicate").

- **value** (`float`, default: `0.0`): float, optional The value to use for padding when the mode is "constant". Default is 0.

- **channel_last** (`bool`, default: `False`): bool, optional Whether the channel dimension is the last dimension of the tensor. Default is False.

## Returns

- **result** (`Tensor`): Tensor A tensor with the specified padding applied to the input tensor.
