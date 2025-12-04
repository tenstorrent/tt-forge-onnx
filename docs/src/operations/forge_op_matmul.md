# forge.op.Matmul

## Overview

Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

## Function Signature

```python
forge.op.Matmul(
    name: str,
    operandA: Tensor,
    operandB: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]]
) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): Input tensor. Shape and data type depend on the specific operation requirements.
- **operandB** (`Union[Tensor, Parameter]`): Second input tensor. Must be broadcastable with operandA.
- **bias** (`Optional[Union[Tensor, Parameter]]`): Optional bias tensor
## Returns

- **result** (`Tensor`): Output tensor containing the matrix multiplication result. The output shape is determined by the input tensor shapes following matrix multiplication rules.

## Related Operations

*Related operations will be automatically linked here in future updates.*

