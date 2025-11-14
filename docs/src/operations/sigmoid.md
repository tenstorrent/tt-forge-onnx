# ttir.sigmoid

Eltwise sigmoid.

The sigmoid operation computes the sigmoid of each element in the input tensor.

For each element, it returns the sigmoid of the value.

## Function Signature

```python
ttir.sigmoid(input, output)
```

## Parameters

- **input** (ranked tensor of any type values): The input tensor

## Returns

- **result** (ranked tensor of any type values): The result tensor

## Mathematical Definition

sigmoid(x) = 1 / (1 + exp(-x))

