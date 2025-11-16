# forge.op.Conv2dTranspose

Conv2dTranspose transformation on input activations, with optional bias.

## Function Signature

```python
forge.op.Conv2dTranspose(name: str, activations: Tensor, weights: Union[Tensor, Parameter], bias: Optional[Union[Tensor, Parameter]]=None, stride: int=1, padding: Union[int, str, Tuple[int, int, int, int]]='same', dilation: int=1, groups: int, channel_last: bool, output_padding: Union[int, Tuple[int, int]]) -> Tensor
```

## Parameters

- **activations** (Tensor): Input activations of shape (N, Cin, iH, iW)
- **weights** (Union[Tensor, Parameter]): Input weights of shape (Cout, Cin / groups, kH, kW) [Tensor] Internal Use pre-split Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)] of length: (K*K // weight_grouping)
- **bias** (Optional[Union[Tensor, Parameter]]): Optional bias tensor of shape (Cout)

- **stride** (int) (default: 1): stride parameter
- **padding** (Union[int, str, Tuple[int, int, int, int]]) (default: 'same'): padding parameter
- **dilation** (int) (default: 1): dilation parameter
- **groups** (int) (default: 1): groups parameter
- **channel_last** (bool) (default: False): channel_last parameter
- **output_padding** (Union[int, Tuple[int, int]]) (default: 0): output_padding parameter

## Returns

- **result** (Output tensor): Tensor

