# ttir.avg_pool2d

2D average pooling operation.

The avg_pool2d operation applies a 2D average pooling over an input tensor composed of several input planes.

This operation performs downsampling by dividing the input into local regions and computing the average value of each region. It reduces the spatial dimensions (height and width) of an input tensor while preserving the batch and channel dimensions.

## Function Signature

```python
ttir.avg_pool2d(input, kernel, stride=1, dilation=1, padding=0, ceil_mode=false, count_include_pad=true, output)
```

## Parameters

- **input** (ranked tensor of any type values): Input tensor in NHWC format (batch, height, width, channels)

- **kernel** (i32 | array<2xi32>): Kernel size for height and width dimensions. Can be a single number or a tuple [kH, kW].
- **stride** (i32 | array<2xi32>) (default: 1): Stride for height and width dimensions. Can be a single number or a tuple [sH, sW].
- **dilation** (i32 | array<2xi32>) (default: 1): Dilation for height and width dimensions. Can be a single number or a tuple [dH, dW].
- **padding** (i32 | array<2xi32> | array<4xi32>) (default: 0): Padding applied to the input. Can be a single number, tuple [pH, pW], or tuple [pT, pL, pB, pR].
- **ceil_mode** (bool) (default: false): When true, uses ceil instead of floor for output shape calculation.
- **count_include_pad** (bool) (default: true): When true, include padding in the average calculation.

## Returns

- **result** (ranked tensor of any type values): Output tensor after average pooling

## Examples

```python
# Basic 2D average pooling with a 2x2 kernel and stride 1
%result = ttir.avg_pool2d(%input, %output) {
    kernel = [2, 2],
    stride = [1, 1],
    dilation = [1, 1],
    padding = [0, 0, 0, 0],
    ceil_mode = false
} : tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32> -> tensor<1x2x2x1xf32>
```

