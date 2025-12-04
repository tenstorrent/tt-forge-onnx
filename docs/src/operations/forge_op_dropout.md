# forge.op.Dropout

## Overview

Dropout

## Function Signature

```python
forge.op.Dropout(name: str, operandA: Tensor, p: float=0.5, training: bool, seed: int) -> Tensor
```

## Parameters

- **name** (`str`): Name identifier for this operation in the computation graph.

- **operandA** (`Tensor`): First operand

- **p** (`float`, default: `0.5`): Probability of an element to be zeroed.
- **training** (`bool`, default: `True`): Apply dropout if true
- **seed** (`int`, default: `0`): RNG seed

## Returns

- **result** (`Tensor`): Tensor

## Related Operations

*Related operations will be automatically linked here in future updates.*

