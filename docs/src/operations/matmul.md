# ttir.matmul

Matrix multiplication operation.

The matmul operation computes the matrix multiplication of two tensors.

This operation performs matrix multiplication between tensors a and b. It supports optional transposition of either input tensor before multiplication. For 2D tensors, this computes the standard matrix product. For tensors with more dimensions, it applies batched matrix multiplication.

## Function Signature

```python
ttir.matmul(a, b, transpose_a=false, transpose_b=false, output)
```

## Parameters

- **a** (ranked tensor of any type values): The first input tensor
- **b** (ranked tensor of any type values): The second input tensor

- **transpose_a** (bool) (default: false): Whether to transpose tensor a before multiplication.
- **transpose_b** (bool) (default: false): Whether to transpose tensor b before multiplication.

## Returns

- **result** (ranked tensor of any type values): The result of the matrix multiplication

## Examples

```python
# Basic matrix multiplication of 2D tensors
%result = ttir.matmul(%a, %b, %output) : tensor<3x4xf32>, tensor<4x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
```

## Notes

The inner dimensions of the input tensors must be compatible for matrix multiplication. If a has shape [..., m, k] and b has shape [..., k, n], then the result will have shape [..., m, n].

