# forge.op.ConstantPad

TM - Direct TTIR constant padding operation.

## Function Signature

```python
forge.op.ConstantPad(name: str, operandA: Tensor, padding: List[int], value: float) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A to which padding will be applied.

- **padding** (List[int]): List[int] Padding values in TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...] Length must be 2 * rank of input tensor.
- **value** (float) (default: 0.0): float, optional The constant value to use for padding. Default is 0.0.

## Returns

- **result** (Output tensor): Tensor

