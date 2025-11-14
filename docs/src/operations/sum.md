# ttir.sum

Sum reduction operation.

The sum operation computes the sum of elements along specified dimensions of the input tensor.

This operation reduces the input tensor by computing the sum of all elements along the dimensions specified in dim_arg. If dim_arg is not provided, the sum is computed over all dimensions, resulting in a scalar value. If keep_dim is set to true, the reduced dimensions are retained with a size of 1.

## Function Signature

```python
ttir.sum(input, keep_dim=false, dim_arg, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

- **keep_dim** (bool) (default: false): Whether to keep the reduced dimensions or not.
- **dim_arg** (array<i32>): Dimensions to reduce along. If not provided, reduces over all dimensions.

## Returns

- **result** (ranked tensor of any type values): The result tensor after applying the reduction

## Mathematical Definition

sum(x, dim) = ∑ x[i] for all i in dimension dim

## Examples

```python
# Sum along dimension 1
%result = ttir.sum(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
```

