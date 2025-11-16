# forge.op.Repeat

Repeats this tensor along the specified dimensions.

>>> x = torch.tensor([1, 2, 3])

>>> x.repeat(4, 2)

tensor([[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3]])

NOTE:

This Forge.Repeat is equivalent to torch.repeat, numpy.tile, tvm.tile, and ttnn.repeat

## Function Signature

```python
forge.op.Repeat(name: str, operandA: Tensor, repeats: List[int]) -> Tensor
```

## Parameters

- **operandA** (Tensor): Input operand A

- **repeats** (List[int]): repeats parameter

## Returns

- **result** (Output tensor): Tensor

