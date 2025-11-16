# forge.op.Resize1d

Resize input activations, with default mode 'nearest'

## Function Signature

```python
forge.op.Resize1d(name: str, operandA: Tensor, size: int, mode: str, align_corners: bool, channel_last: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **size** (int): The target size to extrapolate
- **mode** (str) (default: 'nearest'): Interpolation mode
- **align_corners** (bool) (default: False): align_corners parameter
- **channel_last** (bool) (default: False): Whether the input is in channel-last format (NWC)

## Returns

- **result** (Output tensor): Tensor

