# forge.op.Conv2d

## Overview

Conv2d transformation on input activations, with optional bias.

## Function Signature

```python
forge.op.Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Optional[Union[(Tensor, Parameter)]] = None,
    stride: Union[(int, List[int])] = 1,
    padding: Union[(int, str, List[int])] = 'same',
    dilation: Union[(int, List[int])] = 1,
    groups: int = 1,
    channel_last: bool = False
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **weights** (`Union[(Tensor, Parameter)]`): Tensor Input weights of shape (Cout, Cin / groups, kH, kW) [Tensor] Internal Use pre-split Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)] of length: (K*K // weight_grouping)

- **bias** (`Optional[Union[(Tensor, Parameter)]]`): Optional bias tensor of shape `(C_out,)`. Added to each output channel.

- **stride** (`Union[(int, List[int])]`, default: `1`): stride parameter

- **padding** (`Union[(int, str, List[int])]`, default: `'same'`): padding parameter

- **dilation** (`Union[(int, List[int])]`, default: `1`): dilation parameter

- **groups** (`int`, default: `1`): groups parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

## Returns

- **result** (`Tensor`): Output tensor

## Mathematical Definition

For input `x` of shape `(N, C_in, H, W)` and kernel `k` of shape `(C_out, C_in, K_H, K_W)`:

```
output[n, c_out, h, w] = Σ_{c_in} Σ_{kh} Σ_{kw} x[n, c_in, h*s + kh*d, w*s + kw*d] * k[c_out, c_in, kh, kw] + bias[c_out]
```

Where `s` is stride and `d` is dilation.

## Related Operations

- [forge.op.Conv2dTranspose](./conv2dtranspose.md): Transposed 2D convolution
- [forge.op.AvgPool2d](./avgpool2d.md): 2D average pooling
- [forge.op.MaxPool2d](./maxpool2d.md): 2D max pooling
