# forge.op.Conv2d

## Overview

Conv2d transformation on input activations, with optional bias.

## Function Signature

```python
forge.op.Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]]=None,
    stride: Union[int, List[int]]=1,
    padding: Union[int, str, List[int]]='same',
    dilation: Union[int, List[int]],
    groups: int,
    channel_last: bool
) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **activations** (`Tensor`): Input activations of shape (N, Cin, iH, iW)
- **weights** (`Union[Tensor, Parameter]`): Input weights of shape (Cout, Cin / groups, kH, kW) [Tensor] Internal Use pre-split Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)] of length: (K*K // weight_grouping)
- **bias** (`Optional[Union[Tensor, Parameter]]`): Optional bias tensor of shape (Cout)

- **stride** (`Union[int, List[int]]`, default: `1`): stride parameter
- **padding** (`Union[int, str, List[int]]`, default: `'same'`): padding parameter
- **dilation** (`Union[int, List[int]]`, default: `1`): dilation parameter
- **groups** (`int`, default: `1`): groups parameter
- **channel_last** (`bool`, default: `False`): If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`.

## Returns

- **result** (`Tensor`): Output tensor containing the convolution result. The output shape depends on the input shape, kernel size, padding, and stride parameters.

## Related Operations

*Related operations will be automatically linked here in future updates.*

