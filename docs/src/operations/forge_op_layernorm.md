# forge.op.Layernorm

Layer normalization.

## Function Signature

```python
forge.op.Layernorm(name: str, operandA: Tensor, weights: Union[Tensor, Parameter], bias: Union[Tensor, Parameter], dim: int, epsilon: float) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand
- **weights** (Union[Tensor, Parameter]): weights tensor
- **bias** (Union[Tensor, Parameter]): bias tensor

- **dim** (int) (default: -1): dim parameter
- **epsilon** (float) (default: 1e-05): epsilon parameter

## Returns

- **result** (Output tensor): Tensor

