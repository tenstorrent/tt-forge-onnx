# forge.op.Downsample2d

Downsample 2D operation

## Function Signature

```python
forge.op.Downsample2d(name: str, operandA: Tensor, scale_factor: Union[int, List[int], Tuple[int, int]], mode: str, channel_last: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **scale_factor** (Union[int, List[int], Tuple[int, int]]): Union[int, List[int], Tuple[int, int]] Divider for spatial size.
- **mode** (str) (default: 'nearest'): The downsampling algorithm
- **channel_last** (bool) (default: False): Whether the input is in channel-last format (NHWC)

## Returns

- **result** (Output tensor): Tensor

