# forge.op.Pad

## Overview

Pad tensor manipulation operation

## Function Signature

```python
forge.op.Pad(name: str, operandA: Tensor, pad: Tuple[int, ...], mode: str, value: float, channel_last: bool) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input operand A to which padding will be applied.

- **pad** (`Tuple[int, ...]`): Tuple[int, ...] A tuple of padding values. The tuple should correspond to padding values for the tensor, such as [left, right, top, bottom].
- **mode** (`str`, default: `'constant'`): str, optional The padding mode. Default is "constant". Other modes can be supported depending on the implementation (e.g., "reflect", "replicate").
- **value** (`float`, default: `0.0`): float, optional The value to use for padding when the mode is "constant". Default is 0.
- **channel_last** (`bool`, default: `False`): If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`.

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

