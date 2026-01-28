# forge.op.Matmul

## Overview

Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

## Function Signature

```python
forge.op.Matmul(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)],
    bias: Optional[Union[(Tensor, Parameter)]] = None
) -> Tensor
```

## Parameters

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Input operand B

- **bias** (`Optional[Union[(Tensor, Parameter)]]`): Tenor, optional Optional bias tensor

## Returns

- **result** (`Tensor`): Output tensor

## Mathematical Definition

For matrices `A` of shape `(M, K)` and `B` of shape `(K, N)`:

```
output[i, j] = Σ_k A[i, k] * B[k, j]
```

For batched inputs, the operation is applied to the last two dimensions.

## Related Operations

- [forge.op.Add](./add.md): Elementwise addition (for bias)
- [forge.op.Transpose](./transpose.md): Transpose dimensions before matmul
