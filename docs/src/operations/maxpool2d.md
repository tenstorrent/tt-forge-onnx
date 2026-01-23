# forge.op.MaxPool2d

## Overview

Maxpool2d transformation on input activations

## Function Signature

```python
forge.op.MaxPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 'same',
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    max_pool_add_sub_surround: bool = False,
    max_pool_add_sub_surround_value: float = 1.0,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `'same'`): padding parameter

- **dilation** (`int`, default: `1`): dilation parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **return_indices** (`bool`, default: `False`): return_indices parameter

- **max_pool_add_sub_surround** (`bool`, default: `False`): max_pool_add_sub_surround parameter

- **max_pool_add_sub_surround_value** (`float`, default: `1.0`): max_pool_add_sub_surround_value parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

## Returns

- **result** (`Tensor`): Output tensor
