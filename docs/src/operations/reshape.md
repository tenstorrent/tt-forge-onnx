# ttir.reshape

Tensor reshape operation.

The reshape operation changes the shape of a tensor without changing the data or number of elements.

This operation takes an input tensor and reshapes it to a new shape specified by the shape attribute. The total number of elements in the tensor must remain the same after reshaping.

## Function Signature

```python
ttir.reshape(input, shape, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor to reshape

- **shape** (array<i32>): The new shape for the tensor as an array of integers.

## Returns

- **result** (ranked tensor of any type values): The reshaped tensor

## Examples

```python
# Reshape a 2x3 tensor to a 1x6 tensor
%result = ttir.reshape(%input, %output) {shape = [1, 6]} : tensor<2x3xf32>, tensor<1x6xf32> -> tensor<1x6xf32>
```

## Notes

The total number of elements in the input tensor must equal the total number of elements in the output tensor.

