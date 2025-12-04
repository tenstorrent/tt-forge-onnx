# forge.op.Power

## Overview

OperandA to the power of OperandB

## Function Signature

```python
forge.op.Power(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand
- **operandB** (`Union[Tensor, Parameter]`): Second input tensor. Must be broadcastable with operandA.
## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

