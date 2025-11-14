# ttir.max

Maximum reduction operation.

The max operation computes the maximum value of elements along specified dimensions of the input tensor.

This operation reduces the input tensor by finding the maximum value of all elements along the dimensions specified in dim_arg. If dim_arg is not provided, the maximum is computed over all dimensions, resulting in a scalar value.

## Function Signature

```python
ttir.max(input, keep_dim=false, dim_arg, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

- **keep_dim** (bool) (default: false): Whether to keep the reduced dimensions or not.
- **dim_arg** (array<i32>): Dimensions to reduce along.

## Returns

- **result** (ranked tensor of any type values): The result tensor after applying the reduction

## Mathematical Definition

max(x, dim) = max(x[i]) for all i in dimension dim

## Notes

When comparing with NaN values, NaN is typically not selected as the maximum value.

