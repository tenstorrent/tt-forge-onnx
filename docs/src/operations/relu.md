# ttir.relu

Eltwise ReLU.

The relu operation computes the rectified linear unit (ReLU) of each element in the input tensor.

For each element, it returns the maximum of 0 and the value. The operation preserves the data type of the input.

## Function Signature

```python
ttir.relu(input, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

## Returns

- **result** (ranked tensor of any type values): The result tensor

## Mathematical Definition

relu(x) = max(0, x)

## Implementation Details

**Traits:** AlwaysSpeculatableImplTrait, TTIR_Broadcastable, TTIR_Idempotence, TwoOperands

