# ttir.add

Elementwise addition operation.

The add operation performs an elementwise addition between two tensors.

For each pair of corresponding elements, it adds the elements and places the result in the output tensor.

## Function Signature

```python
ttir.add(lhs, rhs, output)
```

## Parameters

- **lhs** (ranked tensor of any type values): Left-hand side tensor
- **rhs** (ranked tensor of any type values): Right-hand side tensor

## Returns

- **result** (ranked tensor of any type values): The result tensor

## Mathematical Definition

add(x, y) = x + y

## Examples

```python
# Addition operation
%result = ttir.add(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
```

