# forge.op.Dropout

Dropout

## Function Signature

```python
forge.op.Dropout(name: str, operandA: Tensor, p: float=0.5, training: bool, seed: int) -> Tensor
```

## Parameters

- **operandA** (Tensor): First operand

- **p** (float) (default: 0.5): Probability of an element to be zeroed.
- **training** (bool) (default: True): Apply dropout if true
- **seed** (int) (default: 0): RNG seed

## Returns

- **result** (Output tensor): Tensor

