# forge.op.Upsample2d

Upsample 2D operation

## Function Signature

```python
forge.op.Upsample2d(name: str, operandA: Tensor, scale_factor: Union[int, List[int], Tuple[int, int]], mode: str, channel_last: bool) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **scale_factor** (Union[int, List[int], Tuple[int, int]]): Union[int, List[int], Tuple[int, int]] multiplier for spatial size.
- **mode** (str) (default: 'nearest'): the upsampling algorithm
- **channel_last** (bool) (default: False): channel_last parameter

## Returns

- **result** (Output tensor): Tensor

