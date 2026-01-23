# forge.op.AvgPool2d

## Overview

Avgpool2d transformation on input activations

## Function Signature

```python
forge.op.AvgPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 'same',
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: float = None,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `'same'`): padding parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **count_include_pad** (`bool`, default: `True`): count_include_pad parameter

- **divisor_override** (`float`, default: `None`): divisor_override parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

## Returns

- **result** (`Tensor`): Output tensor
