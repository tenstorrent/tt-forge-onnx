# ttir.max_pool2d

2D maximum pooling operation.

The max_pool2d operation applies a 2D maximum pooling over an input tensor composed of several input planes.

This operation performs downsampling by dividing the input into local regions and computing the maximum value of each region. It reduces the spatial dimensions (height and width) of an input tensor while preserving the batch and channel dimensions.

## Function Signature

```python
ttir.max_pool2d(input, kernel, stride=1, dilation=1, padding=0, ceil_mode=false, output)
```

## Parameters

- **input** (ranked tensor of any type values): Input tensor in NHWC format (batch, height, width, channels)

- **kernel** (i32 | array<2xi32>): Kernel size for height and width dimensions.
- **stride** (i32 | array<2xi32>) (default: 1): Stride for height and width dimensions.
- **dilation** (i32 | array<2xi32>) (default: 1): Dilation for height and width dimensions.
- **padding** (i32 | array<2xi32> | array<4xi32>) (default: 0): Padding applied to the input.
- **ceil_mode** (bool) (default: false): When true, uses ceil instead of floor for output shape calculation.

## Returns

- **result** (ranked tensor of any type values): Output tensor after maximum pooling

