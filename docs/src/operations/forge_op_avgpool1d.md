# forge.op.AvgPool1d

## Overview

Avgpool1d transformation on input activations

## Function Signature

```python
forge.op.AvgPool1d(name: str, activations: Tensor, kernel_size: Union[int, Tuple[int, int]], stride: int=1, padding: Union[int, str], ceil_mode: bool, count_include_pad: bool) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **activations** (`Tensor`): Input activations of shape (N, Cin, iW)

- **kernel_size** (`Union[int, Tuple[int, int]]`): Size of pooling region
- **stride** (`int`, default: `1`): stride parameter
- **padding** (`Union[int, str]`, default: `'same'`): padding parameter
- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter
- **count_include_pad** (`bool`, default: `True`): count_include_pad parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

