# forge.op.Matmul

Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

## Function Signature

```python
forge.op.Matmul(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter], bias: Optional[Union[Tensor, Parameter]]) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A
- **operandB** (Union[Tensor, Parameter]): Input operand B
- **bias** (Optional[Union[Tensor, Parameter]]): Optional bias tensor
## Returns

- **result** (Output tensor): Tensor

