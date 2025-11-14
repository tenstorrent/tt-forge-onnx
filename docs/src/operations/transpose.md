# ttir.transpose

Tensor transpose operation.

The transpose operation swaps two dimensions of a tensor.

This operation exchanges the positions of two specified dimensions in the input tensor, effectively transposing those dimensions.

## Function Signature

```python
ttir.transpose(input, dim0, dim1, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

- **dim0** (i32): The first dimension to swap.
- **dim1** (i32): The second dimension to swap.

## Returns

- **result** (ranked tensor of any type values): The transposed tensor

## Examples

```python
# Transpose dimensions 0 and 1
%result = ttir.transpose(%input, %output) {dim0 = 0 : i32, dim1 = 1 : i32} : tensor<2x3x4xf32>, tensor<3x2x4xf32> -> tensor<3x2x4xf32>
```

