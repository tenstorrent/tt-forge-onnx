# forge.op.Resize2d

Resize input activations, with default mode 'nearest'

## Function Signature

```python
forge.op.Resize2d(name: str, operandA: Tensor, sizes: Union[List[int], Tuple[int, int]], mode: str, align_corners: bool, channel_last: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **sizes** (Union[List[int], Tuple[int, int]]): Union[List[int], Tuple[int, int]] The target 2D sizes to extrapolate to
- **mode** (str) (default: 'nearest'): Interpolation mode
- **align_corners** (bool) (default: False): align_corners parameter
- **channel_last** (bool) (default: False): Whether the input is in channel-last format (NHWC)

## Returns

- **result** (Output tensor): Tensor

