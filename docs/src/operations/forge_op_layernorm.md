# forge.op.Layernorm

## Overview

Layer normalization.

## Function Signature

```python
forge.op.Layernorm(name: str, operandA: Tensor, weights: Union[Tensor, Parameter], bias: Union[Tensor, Parameter], dim: int, epsilon: float) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
- **weights** (`Union[Tensor, Parameter]`): weights tensor
- **bias** (`Union[Tensor, Parameter]`): bias tensor

- **dim** (`int`, default: `-1`): dim parameter
- **epsilon** (`float`, default: `1e-05`): epsilon parameter

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

