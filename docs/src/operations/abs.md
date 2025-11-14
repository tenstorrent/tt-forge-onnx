# ttir.abs

Elementwise absolute value operation.

The abs operation computes the absolute value of each element in the input tensor.

For each element, it returns the magnitude of the value without regard to its sign:
- For real numbers, it returns |x| (the non-negative value without sign)

This operation has the idempotence property, meaning that applying it multiple times produces the same result as applying it once: abs(abs(x)) = abs(x). The operation preserves the data type of the input.

## Function Signature

```python
ttir.abs(input, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

## Returns

- **result** (ranked tensor of any type values): The result tensor

## Mathematical Definition

abs(x) = |x| = { x if x ≥ 0, -x if x < 0 }

## Examples

```python
# Compute absolute values
%result = ttir.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
```

## Implementation Details

**Traits:** AlwaysSpeculatableImplTrait, TTIR_Broadcastable, TTIR_Idempotence, TwoOperands

