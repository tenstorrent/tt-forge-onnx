# forge.op.MaxPool1d

## Overview

MaxPool1d transformation on input activations

## Function Signature

```python
forge.op.MaxPool1d(name: str, activations: Tensor, kernel_size: Union[int, Tuple[int, int]], stride: int=1, padding: Union[int, str]=0, dilation: int, ceil_mode: bool, return_indices: bool) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **activations** (`Tensor`): Input activations of shape (N, Cin, iW)

- **kernel_size** (`Union[int, Tuple[int, int]]`): Size of pooling region
- **stride** (`int`, default: `1`): stride parameter
- **padding** (`Union[int, str]`, default: `0`): padding parameter
- **dilation** (`int`, default: `1`): dilation parameter
- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter
- **return_indices** (`bool`, default: `False`): return_indices parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

