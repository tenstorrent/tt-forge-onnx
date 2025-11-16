# forge.op.Batchnorm

Batch normalization.

## Function Signature

```python
forge.op.Batchnorm(name: str, operandA: Tensor, weights: Union[Tensor, Parameter], bias: Union[Tensor, Parameter], running_mean: Union[Tensor, Parameter], running_var: Union[Tensor, Parameter], epsilon: float) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand
- **weights** (Union[Tensor, Parameter]): weights tensor
- **bias** (Union[Tensor, Parameter]): bias tensor
- **running_mean** (Union[Tensor, Parameter]): running_mean tensor
- **running_var** (Union[Tensor, Parameter]): running_var tensor

- **epsilon** (float) (default: 1e-05): epsilon parameter

## Returns

- **result** (Output tensor): Tensor

