# ttir.conv2d

Conv2d operation.

Applies a 2D convolution over an input image composed of several input planes.

This operation performs a 2D convolution on the input tensor using the provided weight tensor and optional bias. It supports configurable stride, padding, dilation, and grouping parameters to control the convolution behavior.

## Function Signature

```python
ttir.conv2d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, output)
```

## Parameters

- **input** (ranked tensor of any type values): Input tensor in NHWC format (batch, height, width, channels)
- **weight** (ranked tensor of any type values): Weight tensor in format (O, C/G, K_H, K_W)
- **bias** (ranked tensor of any type values): Optional bias tensor

- **stride** (i32 | array<2xi32>) (default: 1): Stride of the convolving kernel. Can be a single number or a tuple (sH, sW).
- **padding** (i32 | array<2xi32> | array<4xi32>) (default: 0): Padding applied to the input. Can be a single number, tuple (pH, pW), or tuple (pT, pL, pB, pR).
- **dilation** (i32 | array<2xi32>) (default: 1): Spacing between kernel elements. Can be a single number or a tuple (dH, dW).
- **groups** (i32) (default: 1): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

## Returns

- **result** (ranked tensor of any type values): Output tensor after convolution

## Examples

```python
# Basic 2D convolution
%result = ttir.conv2d(%input, %weight, %bias, %output) {
    stride = [1, 1],
    padding = [0, 0, 0, 0],
    dilation = [1, 1],
    groups = 1
} : tensor<1x28x28x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>, tensor<1x26x26x16xf32> -> tensor<1x26x26x16xf32>
```

## Notes

The input tensor is expected in NHWC format: (N, H_in, W_in, C) where N is batch size, H_in is height, W_in is width, and C is number of channels.

The weight tensor format is (O, C/G, K_H, K_W) where O is output channels, C is input channels, G is number of groups, K_H is kernel height, and K_W is kernel width.

The output shape is calculated as: H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1

