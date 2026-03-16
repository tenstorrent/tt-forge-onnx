# Forge Operations Reference

Welcome to the Forge Operations Reference. This page provides a comprehensive guide to all supported operations in the Forge framework.

## Overview

Forge operations are organized into logical categories based on their functionality. Each operation is documented with detailed information including function signatures, parameters, examples, and usage notes.

## Quick Navigation

- [Elementwise Operations](#elementwise-operations) - Mathematical operations applied element-wise
- [Convolution Operations](#convolution-operations) - Convolution and related transformations
- [Pooling Operations](#pooling-operations) - Pooling and downsampling operations
- [Normalization Operations](#normalization-operations) - Batch and layer normalization
- [Tensor Manipulation](#tensor-manipulation) - Reshaping, slicing, and tensor operations
- [Reduction Operations](#reduction-operations) - Aggregation and reduction operations
- [Linear Operations](#linear-operations) - Matrix multiplication and linear transformations
- [Activation Functions](#activation-functions) - Non-linear activation functions
- [Memory Operations](#memory-operations) - Cache and memory management operations
- [Other Operations](#other-operations) - Miscellaneous operations

---

## Elementwise Operations

Mathematical operations applied element-wise.

| Operation | Description | Link |
|-----------|-------------|------|
| **Abs** | Computes the elementwise absolute value of the input tensor. | [forge.op.Abs](#abs) |
| **Add** | Elementwise add of two tensors | [forge.op.Add](#add) |
| **Atan** | Elementwise arctangent (atan) | [forge.op.Atan](#atan) |
| **BitwiseAnd** | Bitwise and operation. | [forge.op.BitwiseAnd](#bitwiseand) |
| **Cast** | Cast | [forge.op.Cast](#cast) |
| **Clip** | Clips tensor values between min and max | [forge.op.Clip](#clip) |
| **Concatenate** | Concatenate tensors along axis | [forge.op.Concatenate](#concatenate) |
| **Cosine** | Elementwise cosine | [forge.op.Cosine](#cosine) |
| **Divide** | Elementwise divide of two tensors | [forge.op.Divide](#divide) |
| **Equal** | Elementwise equal of two tensors | [forge.op.Equal](#equal) |
| **Erf** | Error function (erf) | [forge.op.Erf](#erf) |
| **Exp** | Exponent operation. | [forge.op.Exp](#exp) |
| **Greater** | Elementwise greater of two tensors | [forge.op.Greater](#greater) |
| **GreaterEqual** | Elementwise greater or equal of two tensors | [forge.op.GreaterEqual](#greaterequal) |
| **Heaviside** | Elementwise max of two tensors | [forge.op.Heaviside](#heaviside) |
| **Identity** | Identity operation. | [forge.op.Identity](#identity) |
| **IndexCopy** | Copies the elements of value into operandA at index along dim | [forge.op.IndexCopy](#indexcopy) |
| **Less** | Elementwise less of two tensors | [forge.op.Less](#less) |
| **LessEqual** | Elementwise less or equal of two tensors | [forge.op.LessEqual](#lessequal) |
| **Log** | Log operation: natural logarithm of the elements of `operandA` | [forge.op.Log](#log) |
| **LogicalAnd** | Logical and operation. | [forge.op.LogicalAnd](#logicaland) |
| **LogicalNot** | Logical not operation. | [forge.op.LogicalNot](#logicalnot) |
| **Max** | Elementwise max of two tensors | [forge.op.Max](#max) |
| **Min** | Elementwise min of two tensors | [forge.op.Min](#min) |
| **Multiply** | Elementwise multiply of two tensors | [forge.op.Multiply](#multiply) |
| **NotEqual** | Elementwise equal of two tensors | [forge.op.NotEqual](#notequal) |
| **Pow** | Pow operation: `operandA` to the power of `exponent` | [forge.op.Pow](#pow) |
| **Power** | OperandA to the power of OperandB | [forge.op.Power](#power) |
| **Reciprocal** | Reciprocal operation. | [forge.op.Reciprocal](#reciprocal) |
| **Remainder** |  | [forge.op.Remainder](#remainder) |
| **Sine** | Elementwise sine | [forge.op.Sine](#sine) |
| **Sqrt** | Square root. | [forge.op.Sqrt](#sqrt) |
| **Stack** | Stack tensors along new axis | [forge.op.Stack](#stack) |
| **Subtract** | Elementwise subtraction of two tensors | [forge.op.Subtract](#subtract) |
| **Where** |  | [forge.op.Where](#where) |

## Convolution Operations

Convolution and related transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Conv2d** | Conv2d transformation on input activations, with optional bias. | [forge.op.Conv2d](#conv2d) |
| **Conv2dTranspose** | Conv2dTranspose transformation on input activations, with optional bias. | [forge.op.Conv2dTranspose](#conv2dtranspose) |

## Pooling Operations

Pooling and downsampling operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **AvgPool1d** | Avgpool1d transformation on input activations | [forge.op.AvgPool1d](#avgpool1d) |
| **AvgPool2d** | Avgpool2d transformation on input activations | [forge.op.AvgPool2d](#avgpool2d) |
| **MaxPool1d** | MaxPool1d transformation on input activations | [forge.op.MaxPool1d](#maxpool1d) |
| **MaxPool2d** | Maxpool2d transformation on input activations | [forge.op.MaxPool2d](#maxpool2d) |

## Normalization Operations

Batch and layer normalization.

| Operation | Description | Link |
|-----------|-------------|------|
| **Batchnorm** | Batch normalization. | [forge.op.Batchnorm](#batchnorm) |
| **Dropout** | Dropout | [forge.op.Dropout](#dropout) |
| **Layernorm** | Layer normalization. | [forge.op.Layernorm](#layernorm) |
| **LogSoftmax** | LogSoftmax operation. | [forge.op.LogSoftmax](#logsoftmax) |
| **Softmax** | Softmax operation. | [forge.op.Softmax](#softmax) |

## Tensor Manipulation

Reshaping, slicing, and tensor operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **AdvIndex** | TM | [forge.op.AdvIndex](#advindex) |
| **Broadcast** | TM | [forge.op.Broadcast](#broadcast) |
| **ConstantPad** | TM - Direct TTIR constant padding operation. | [forge.op.ConstantPad](#constantpad) |
| **Downsample2d** | Downsample 2D operation | [forge.op.Downsample2d](#downsample2d) |
| **Index** | TM | [forge.op.Index](#index) |
| **Pad** | TM | [forge.op.Pad](#pad) |
| **PixelShuffle** | Pixel shuffle operation. | [forge.op.PixelShuffle](#pixelshuffle) |
| **Repeat** | Repeats this tensor along the specified dimensions. | [forge.op.Repeat](#repeat) |
| **RepeatInterleave** | Repeat elements of a tensor. | [forge.op.RepeatInterleave](#repeatinterleave) |
| **Reshape** | TM | [forge.op.Reshape](#reshape) |
| **Resize1d** | Resize input activations, with default mode 'nearest' | [forge.op.Resize1d](#resize1d) |
| **Resize2d** | Resizes the spatial dimensions of a 2D input tensor using interpolation. | [forge.op.Resize2d](#resize2d) |
| **Select** | TM | [forge.op.Select](#select) |
| **Squeeze** | TM | [forge.op.Squeeze](#squeeze) |
| **Transpose** | Tranpose X and Y (i.e. rows and columns) dimensions. | [forge.op.Transpose](#transpose) |
| **Unsqueeze** | TM | [forge.op.Unsqueeze](#unsqueeze) |
| **Upsample2d** | Upsample 2D operation | [forge.op.Upsample2d](#upsample2d) |

## Reduction Operations

Aggregation and reduction operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Argmax** | Argmax | [forge.op.Argmax](#argmax) |
| **ReduceAvg** | Reduce by averaging along the given dimension | [forge.op.ReduceAvg](#reduceavg) |
| **ReduceMax** | Reduce by taking maximum along the given dimension | [forge.op.ReduceMax](#reducemax) |
| **ReduceSum** | Reduce by summing along the given dimension | [forge.op.ReduceSum](#reducesum) |

## Linear Operations

Matrix multiplication and linear transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Matmul** | Matrix multiplication transformation on input activations, with optional bias. y... | [forge.op.Matmul](#matmul) |

## Activation Functions

Non-linear activation functions.

| Operation | Description | Link |
|-----------|-------------|------|
| **Gelu** | GeLU | [forge.op.Gelu](#gelu) |
| **LeakyRelu** | Leaky ReLU | [forge.op.LeakyRelu](#leakyrelu) |
| **Relu** | Applies the Rectified Linear Unit (ReLU) activation function elementwise. | [forge.op.Relu](#relu) |
| **Sigmoid** | Sigmoid | [forge.op.Sigmoid](#sigmoid) |
| **Tanh** | Tanh operation. | [forge.op.Tanh](#tanh) |

## Memory Operations

Cache and memory management operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **FillCache** | FillCache op writes the input into the cache tensor starting at the specified up... | [forge.op.FillCache](#fillcache) |
| **UpdateCache** | UpdateCache writes a single token (S=1) slice into the cache tensor on specified... | [forge.op.UpdateCache](#updatecache) |

## Other Operations

Miscellaneous operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Constant** | Op representing user-defined constant | [forge.op.Constant](#constant) |
| **CumSum** | Cumulative sum operation. | [forge.op.CumSum](#cumsum) |
| **Embedding** | Embedding lookup | [forge.op.Embedding](#embedding) |

---

## Operation Details

### Abs

Computes the elementwise absolute value of the input tensor.

The Abs operation returns the magnitude of each element without regard

to its sign. For real numbers, it returns the non-negative value.

This operation is idempotent: abs(abs(x)) = abs(x).

**Function Signature**

```python
forge.op.Abs(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Tensor Input tensor of any shape. All elements will have absolute value computed independently.

**Returns**

- **result** (`Tensor`): Tensor Output tensor with same shape as input. Each element is the absolute value of the corresponding input element.

**Mathematical Definition**

```
abs(x) = |x| = { x if x ≥ 0, -x if x < 0 }
```

---
### Add

Elementwise add of two tensors

**Function Signature**

```python
forge.op.Add(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### AdvIndex

TM

**Function Signature**

```python
forge.op.AdvIndex(
    name: str,
    operandA: Tensor,
    operandB: Tensor,
    dim: int = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand B - indices

- **operandB** (`Tensor`): operandB tensor

- **dim** (`int`, default: `0`): int Dimension to fetch indices over

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Argmax

Argmax

**Function Signature**

```python
forge.op.Argmax(
    name: str,
    operandA: Tensor,
    dim: int = None,
    keep_dim = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`, default: `None`): int The dimension to reduce (if None, the output is the argmax of the whole tensor)

- **keep_dim** (`Any`, default: `False`): bool If True, retains the dimension that is reduced, with size 1. If False (default), the dimension is removed from the output shape.

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Atan

Elementwise arctangent (atan)

**Function Signature**

```python
forge.op.Atan(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### AvgPool1d

Avgpool1d transformation on input activations

**Function Signature**

```python
forge.op.AvgPool1d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 'same',
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `'same'`): padding parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **count_include_pad** (`bool`, default: `True`): count_include_pad parameter

**Returns**

- **result** (`Tensor`): Output tensor

---
### AvgPool2d

Avgpool2d transformation on input activations

**Function Signature**

```python
forge.op.AvgPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 'same',
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: float = None,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `'same'`): padding parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **count_include_pad** (`bool`, default: `True`): count_include_pad parameter

- **divisor_override** (`float`, default: `None`): divisor_override parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

**Returns**

- **result** (`Tensor`): Output tensor

---
### Batchnorm

Batch normalization.

**Function Signature**

```python
forge.op.Batchnorm(
    name: str,
    operandA: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Union[(Tensor, Parameter)],
    running_mean: Union[(Tensor, Parameter)],
    running_var: Union[(Tensor, Parameter)],
    epsilon: float = 1e-05
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **weights** (`Union[(Tensor, Parameter)]`): weights tensor

- **bias** (`Union[(Tensor, Parameter)]`): bias tensor

- **running_mean** (`Union[(Tensor, Parameter)]`): running_mean tensor

- **running_var** (`Union[(Tensor, Parameter)]`): running_var tensor

- **epsilon** (`float`, default: `1e-05`): epsilon parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### BitwiseAnd

Bitwise and operation.

**Function Signature**

```python
forge.op.BitwiseAnd(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Broadcast

TM

**Function Signature**

```python
forge.op.Broadcast(name: str, operandA: Tensor, dim: int, shape: int) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to broadcast

- **shape** (`int`): int Output length of dim

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Cast

Cast

**Function Signature**

```python
forge.op.Cast(
    name: str,
    operandA: Tensor,
    dtype: Union[(torch.dtype, DataFormat)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dtype** (`Union[(torch.dtype, DataFormat)]`): Union[torch.dtype, DataFormat] Specify Torch datatype / Forge DataFormat to convert operandA

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Clip

Clips tensor values between min and max

**Function Signature**

```python
forge.op.Clip(name: str, operandA: Tensor, min: float, max: float) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **min** (`float`): float Minimum value

- **max** (`float`): float Maximum value

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Concatenate

Concatenate tensors along axis

**Function Signature**

```python
forge.op.Concatenate(name: str) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Constant

Op representing user-defined constant

**Function Signature**

```python
forge.op.Constant(name: str) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### ConstantPad

TM - Direct TTIR constant padding operation.

**Function Signature**

```python
forge.op.ConstantPad(
    name: str,
    operandA: Tensor,
    padding: List[int],
    value: float = 0.0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A to which padding will be applied.

- **padding** (`List[int]`): List[int] Padding values in TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...] Length must be 2 * rank of input tensor.

- **value** (`float`, default: `0.0`): float, optional The constant value to use for padding. Default is 0.0.

**Returns**

- **result** (`Tensor`): Tensor A tensor with the specified constant padding applied to the input tensor.

---
### Conv2d

Conv2d transformation on input activations, with optional bias.

**Function Signature**

```python
forge.op.Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Optional[Union[(Tensor, Parameter)]] = None,
    stride: Union[(int, List[int])] = 1,
    padding: Union[(int, str, List[int])] = 'same',
    dilation: Union[(int, List[int])] = 1,
    groups: int = 1,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **weights** (`Union[(Tensor, Parameter)]`): Tensor Input weights of shape (Cout, Cin / groups, kH, kW) [Tensor] Internal Use pre-split Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)] of length: (K*K // weight_grouping)

- **bias** (`Optional[Union[(Tensor, Parameter)]]`): Optional bias tensor of shape `(C_out,)`. Added to each output channel.

- **stride** (`Union[(int, List[int])]`, default: `1`): stride parameter

- **padding** (`Union[(int, str, List[int])]`, default: `'same'`): padding parameter

- **dilation** (`Union[(int, List[int])]`, default: `1`): dilation parameter

- **groups** (`int`, default: `1`): groups parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

**Returns**

- **result** (`Tensor`): Output tensor

**Mathematical Definition**

For input `x` of shape `(N, C_in, H, W)` and kernel `k` of shape `(C_out, C_in, K_H, K_W)`:

```
output[n, c_out, h, w] = Σ_{c_in} Σ_{kh} Σ_{kw} x[n, c_in, h*s + kh*d, w*s + kw*d] * k[c_out, c_in, kh, kw] + bias[c_out]
```

Where `s` is stride and `d` is dilation.

---
### Conv2dTranspose

Conv2dTranspose transformation on input activations, with optional bias.

**Function Signature**

```python
forge.op.Conv2dTranspose(
    name: str,
    activations: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Optional[Union[(Tensor, Parameter)]] = None,
    stride: int = 1,
    padding: Union[(int, str, Tuple[(int, int, int, int)])] = 'same',
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
    output_padding: Union[(int, Tuple[(int, int)])] = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **weights** (`Union[(Tensor, Parameter)]`): Tensor Input weights of shape (Cout, Cin / groups, kH, kW) [Tensor] Internal Use pre-split Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)] of length: (K*K // weight_grouping)

- **bias** (`Optional[Union[(Tensor, Parameter)]]`): Tenor, optional Optional bias tensor of shape (Cout)

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str, Tuple[(int, int, int, int)])]`, default: `'same'`): padding parameter

- **dilation** (`int`, default: `1`): dilation parameter

- **groups** (`int`, default: `1`): groups parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

- **output_padding** (`Union[(int, Tuple[(int, int)])]`, default: `0`): output_padding parameter

**Returns**

- **result** (`Tensor`): Output tensor

---
### Cosine

Elementwise cosine

**Function Signature**

```python
forge.op.Cosine(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### CumSum

Cumulative sum operation.

**Function Signature**

```python
forge.op.CumSum(name: str, operandA: Tensor, dim: int) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`): dim parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Divide

Elementwise divide of two tensors

**Function Signature**

```python
forge.op.Divide(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Downsample2d

Downsample 2D operation

**Function Signature**

```python
forge.op.Downsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[(int, List[int], Tuple[(int, int)])],
    mode: str = 'nearest',
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **scale_factor** (`Union[(int, List[int], Tuple[(int, int)])]`): Union[int, List[int], Tuple[int, int]] Divider for spatial size.

- **mode** (`str`, default: `'nearest'`): str The downsampling algorithm

- **channel_last** (`bool`, default: `False`): bool Whether the input is in channel-last format (NHWC)

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Dropout

Dropout

**Function Signature**

```python
forge.op.Dropout(
    name: str,
    operandA: Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **p** (`float`, default: `0.5`): float Probability of an element to be zeroed.

- **training** (`bool`, default: `True`): bool Apply dropout if true

- **seed** (`int`, default: `0`): int RNG seed

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Embedding

Embedding lookup

**Function Signature**

```python
forge.op.Embedding(
    name: str,
    indices: Tensor,
    embedding_table: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **indices** (`Tensor`): Tensor Integer tensor, the elements of which are used to index into the embedding table

- **embedding_table** (`Union[(Tensor, Parameter)]`): Tensor Dictionary of embeddings

**Returns**

- **result** (`Tensor`): Output tensor

---
### Equal

Elementwise equal of two tensors

**Function Signature**

```python
forge.op.Equal(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Erf

Error function (erf)

**Function Signature**

```python
forge.op.Erf(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Exp

Exponent operation.

**Function Signature**

```python
forge.op.Exp(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### FillCache

FillCache op writes the input into the cache tensor starting at the specified update index.

**Function Signature**

```python
forge.op.FillCache(
    name: str,
    cache: Tensor,
    input: Tensor,
    batch_offset: int = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Unique op name.

- **cache** (`Tensor`): Tensor 4D cache tensor of shape [B, H, S_total, D]

- **input** (`Tensor`): Tensor 4D input tensor of shape [B, H, S_input, D]

- **batch_offset** (`int`, default: `0`): int Offset in the batch dimension.

**Returns**

- **result** (`Tensor`): Output tensor

---
### Gelu

GeLU

**Function Signature**

```python
forge.op.Gelu(name: str, operandA: Tensor, approximate = 'none') -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **approximate** (`Any`, default: `'none'`): str The gelu approximation algorithm to use: 'none' | 'tanh'. Default: 'none'

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

**Mathematical Definition**

```
gelu(x) = x * Φ(x)
```

Where Φ(x) is the cumulative distribution function of the standard normal distribution.

For 'tanh' approximation:
```
gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

---
### Greater

Elementwise greater of two tensors

**Function Signature**

```python
forge.op.Greater(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### GreaterEqual

Elementwise greater or equal of two tensors

**Function Signature**

```python
forge.op.GreaterEqual(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Heaviside

Elementwise max of two tensors

**Function Signature**

```python
forge.op.Heaviside(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Identity

Identity operation.

**Function Signature**

```python
forge.op.Identity(
    name: str,
    operandA: Tensor,
    unsqueeze: str = None,
    unsqueeze_dim: int = None
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **unsqueeze** (`str`, default: `None`): str If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.

- **unsqueeze_dim** (`int`, default: `None`): int The index at where singleton dimenion can be inserted

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Index

TM

**Function Signature**

```python
forge.op.Index(
    name: str,
    operandA: Tensor,
    dim: int,
    start: int,
    stop: int = None,
    stride: int = 1
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to slice

- **start** (`int`): int Starting slice index (inclusive)

- **stop** (`int`, default: `None`): int Stopping slice index (exclusive)

- **stride** (`int`, default: `1`): int Stride amount along that dimension

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### IndexCopy

Copies the elements of value into operandA at index along dim

**Function Signature**

```python
forge.op.IndexCopy(
    name: str,
    operandA: Tensor,
    index: Tensor,
    value: Tensor,
    dim: int
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **index** (`Tensor`): Tensor Index at which to write into operandA

- **value** (`Tensor`): Tensor Value to write out

- **dim** (`int`): int Dimension to broadcast

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Layernorm

Layer normalization.

**Function Signature**

```python
forge.op.Layernorm(
    name: str,
    operandA: Tensor,
    weights: Union[(Tensor, Parameter)],
    bias: Union[(Tensor, Parameter)],
    dim: int = -1,
    epsilon: float = 1e-05
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **weights** (`Union[(Tensor, Parameter)]`): weights tensor

- **bias** (`Union[(Tensor, Parameter)]`): bias tensor

- **dim** (`int`, default: `-1`): dim parameter

- **epsilon** (`float`, default: `1e-05`): epsilon parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### LeakyRelu

Leaky ReLU

**Function Signature**

```python
forge.op.LeakyRelu(name: str, operandA: Tensor, alpha: float) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **alpha** (`float`): float Controls the angle of the negative slope

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Less

Elementwise less of two tensors

**Function Signature**

```python
forge.op.Less(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### LessEqual

Elementwise less or equal of two tensors

**Function Signature**

```python
forge.op.LessEqual(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Log

Log operation: natural logarithm of the elements of `operandA`.

yi = log_e(xi) for all xi in operandA tensor

**Function Signature**

```python
forge.op.Log(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### LogicalAnd

Logical and operation.

**Function Signature**

```python
forge.op.LogicalAnd(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### LogicalNot

Logical not operation.

**Function Signature**

```python
forge.op.LogicalNot(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### LogSoftmax

LogSoftmax operation.

**Function Signature**

```python
forge.op.LogSoftmax(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Matmul

Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

**Function Signature**

```python
forge.op.Matmul(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)],
    bias: Optional[Union[(Tensor, Parameter)]] = None
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Input operand B

- **bias** (`Optional[Union[(Tensor, Parameter)]]`): Tenor, optional Optional bias tensor

**Returns**

- **result** (`Tensor`): Output tensor

**Mathematical Definition**

For matrices `A` of shape `(M, K)` and `B` of shape `(K, N)`:

```
output[i, j] = Σ_k A[i, k] * B[k, j]
```

For batched inputs, the operation is applied to the last two dimensions.

---
### Max

Elementwise max of two tensors

**Function Signature**

```python
forge.op.Max(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### MaxPool1d

MaxPool1d transformation on input activations

**Function Signature**

```python
forge.op.MaxPool1d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `0`): padding parameter

- **dilation** (`int`, default: `1`): dilation parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **return_indices** (`bool`, default: `False`): return_indices parameter

**Returns**

- **result** (`Tensor`): Output tensor

---
### MaxPool2d

Maxpool2d transformation on input activations

**Function Signature**

```python
forge.op.MaxPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[(int, Tuple[(int, int)])],
    stride: int = 1,
    padding: Union[(int, str)] = 'same',
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    max_pool_add_sub_surround: bool = False,
    max_pool_add_sub_surround_value: float = 1.0,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **activations** (`Tensor`): Tensor Input activations of shape (N, Cin, iH, iW)

- **kernel_size** (`Union[(int, Tuple[(int, int)])]`): Size of pooling region

- **stride** (`int`, default: `1`): stride parameter

- **padding** (`Union[(int, str)]`, default: `'same'`): padding parameter

- **dilation** (`int`, default: `1`): dilation parameter

- **ceil_mode** (`bool`, default: `False`): ceil_mode parameter

- **return_indices** (`bool`, default: `False`): return_indices parameter

- **max_pool_add_sub_surround** (`bool`, default: `False`): max_pool_add_sub_surround parameter

- **max_pool_add_sub_surround_value** (`float`, default: `1.0`): max_pool_add_sub_surround_value parameter

- **channel_last** (`bool`, default: `False`): channel_last parameter

**Returns**

- **result** (`Tensor`): Output tensor

---
### Min

Elementwise min of two tensors

**Function Signature**

```python
forge.op.Min(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Multiply

Elementwise multiply of two tensors

**Function Signature**

```python
forge.op.Multiply(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### NotEqual

Elementwise equal of two tensors

**Function Signature**

```python
forge.op.NotEqual(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Pad

TM

**Function Signature**

```python
forge.op.Pad(
    name: str,
    operandA: Tensor,
    pad: Tuple[(int, Ellipsis)],
    mode: str = 'constant',
    value: float = 0.0,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A to which padding will be applied.

- **pad** (`Tuple[(int, Ellipsis)]`): Tuple[int, ...] A tuple of padding values. The tuple should correspond to padding values for the tensor, such as [left, right, top, bottom].

- **mode** (`str`, default: `'constant'`): str, optional The padding mode. Default is "constant". Other modes can be supported depending on the implementation (e.g., "reflect", "replicate").

- **value** (`float`, default: `0.0`): float, optional The value to use for padding when the mode is "constant". Default is 0.

- **channel_last** (`bool`, default: `False`): bool, optional Whether the channel dimension is the last dimension of the tensor. Default is False.

**Returns**

- **result** (`Tensor`): Tensor A tensor with the specified padding applied to the input tensor.

---
### PixelShuffle

Pixel shuffle operation.

**Function Signature**

```python
forge.op.PixelShuffle(
    name: str,
    operandA: Tensor,
    upscale_factor: int
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **upscale_factor** (`int`): upscale_factor parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Pow

Pow operation: `operandA` to the power of `exponent`.

yi = pow(xi, exponent) for all xi in operandA tensor

**Function Signature**

```python
forge.op.Pow(
    name: str,
    operandA: Tensor,
    exponent: Union[(int, float)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **exponent** (`Union[(int, float)]`): exponent parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Power

OperandA to the power of OperandB

**Function Signature**

```python
forge.op.Power(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Union[(Tensor, Parameter)]`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Reciprocal

Reciprocal operation.

**Function Signature**

```python
forge.op.Reciprocal(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### ReduceAvg

Reduce by averaging along the given dimension

**Function Signature**

```python
forge.op.ReduceAvg(
    name: str,
    operandA: Tensor,
    dim: int,
    keep_dim: bool = True
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`): int Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

- **keep_dim** (`bool`, default: `True`): keep_dim parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### ReduceMax

Reduce by taking maximum along the given dimension

**Function Signature**

```python
forge.op.ReduceMax(
    name: str,
    operandA: Tensor,
    dim: int,
    keep_dim: bool = True
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`): int Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

- **keep_dim** (`bool`, default: `True`): keep_dim parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### ReduceSum

Reduce by summing along the given dimension

**Function Signature**

```python
forge.op.ReduceSum(
    name: str,
    operandA: Tensor,
    dim: int,
    keep_dim: bool = True
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim** (`int`): int Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

- **keep_dim** (`bool`, default: `True`): keep_dim parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Relu

Applies the Rectified Linear Unit (ReLU) activation function elementwise.

ReLU sets all negative values to zero while keeping positive values

unchanged. This introduces non-linearity to neural networks and is one

of the most commonly used activation functions due to its simplicity

and effectiveness.

**Function Signature**

```python
forge.op.Relu(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Tensor Input tensor of any shape. The ReLU function is applied independently to each element.

**Returns**

- **result** (`Tensor`): Tensor Output tensor with same shape as input. Each element is max(0, x) where x is the corresponding input element.

**Mathematical Definition**

```
relu(x) = max(0, x) = { x if x > 0, 0 if x ≤ 0 }
```

---
### Remainder



**Function Signature**

```python
forge.op.Remainder(
    name: str,
    operandA: Tensor,
    operandB: Union[(Tensor, Parameter)]
) -> Tensor
```

**Parameters**

- **name** (`str`): name parameter

- **operandA** (`Tensor`): operandA tensor

- **operandB** (`Union[(Tensor, Parameter)]`): operandB tensor

**Returns**

- **result** (`Tensor`): Output tensor

---
### Repeat

Repeats this tensor along the specified dimensions.

>>> x = torch.tensor([1, 2, 3])

>>> x.repeat(4, 2)

tensor([[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3],

[ 1,  2,  3,  1,  2,  3]])

NOTE:

This Forge.Repeat is equivalent to torch.repeat, numpy.tile, tvm.tile, and ttnn.repeat

**Function Signature**

```python
forge.op.Repeat(name: str, operandA: Tensor, repeats: List[int]) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **repeats** (`List[int]`): repeats parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### RepeatInterleave

Repeat elements of a tensor.

>>> x = torch.tensor([1, 2, 3])

>>> x.repeat_interleave(2)

tensor([1, 1, 2, 2, 3, 3])

NOTE:

This Forge.RepeatInterleave is equivalent to torch.repeat_interleave, numpy.repeat, tvm.repeat, and ttnn.repeat_interleave

**Function Signature**

```python
forge.op.RepeatInterleave(
    name: str,
    operandA: Tensor,
    repeats: int,
    dim: int
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **repeats** (`int`): int The number of repetitions for each element.

- **dim** (`int`): int The dimension along which to repeat values.

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Reshape

TM

**Function Signature**

```python
forge.op.Reshape(
    name: str,
    operandA: Tensor,
    shape: Tuple[(int, Ellipsis)]
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **shape** (`Tuple[(int, Ellipsis)]`): shape parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Resize1d

Resize input activations, with default mode 'nearest'

**Function Signature**

```python
forge.op.Resize1d(
    name: str,
    operandA: Tensor,
    size: int,
    mode: str = 'nearest',
    align_corners: bool = False,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **size** (`int`): int The target size to extrapolate

- **mode** (`str`, default: `'nearest'`): str Interpolation mode

- **align_corners** (`bool`, default: `False`): align_corners parameter

- **channel_last** (`bool`, default: `False`): bool Whether the input is in channel-last format (NWC)

**Returns**

- **result** (`Tensor`): Output tensor

---
### Resize2d

Resizes the spatial dimensions (height and width) of a 2D input tensor using interpolation. This operation is commonly used in computer vision tasks for image resizing, upsampling, and downsampling.

**Function Signature**

```python
forge.op.Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[(List[int], Tuple[(int, int)])],
    mode: str = 'nearest',
    align_corners: bool = False,
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Name identifier for this operation in the computation graph. Use empty string to auto-generate.

- **operandA** (`Tensor`): Input tensor of shape `(N, C, H, W)` for channel-first or `(N, H, W, C)` for channel-last format.

- **sizes** (`Union[(List[int], Tuple[(int, int)])]`): Target output spatial dimensions as `[height, width]`. The output tensor will have these exact height and width values.

- **mode** (`str`, default: `'nearest'`): Interpolation mode: `'nearest'` for nearest neighbor (fast) or `'bilinear'` for bilinear interpolation (smoother).

- **align_corners** (`bool`, default: `False`): If `True`, corner pixels are aligned. Only affects bilinear mode.

- **channel_last** (`bool`, default: `False`): If `True`, input is `(N, H, W, C)` format; if `False`, input is `(N, C, H, W)` format.

**Returns**

- **result** (`Tensor`): Tensor Output tensor with resized spatial dimensions: - Shape `(N, C, H_out, W_out)` if `channel_last=False` - Shape `(N, H_out, W_out, C)` if `channel_last=True` where `H_out, W_out` are the values specified in `sizes`.

**Mathematical Definition**

### Nearest Neighbor Interpolation

For nearest neighbor interpolation, each output pixel value is taken from the nearest input pixel:

```
output[i, j] = input[round(i * H_in / H_out), round(j * W_in / W_out)]
```

### Bilinear Interpolation

For bilinear interpolation, each output pixel is computed as a weighted average of the four nearest input pixels:

```
output[i, j] = Σ(weight_k * input[k]) for k in {top-left, top-right, bottom-left, bottom-right}
```

The weights are computed based on the distance from the output pixel to the surrounding input pixels.

---
### Select

TM

**Function Signature**

```python
forge.op.Select(
    name: str,
    operandA: Tensor,
    dim: int,
    index: Union[(int, Tuple[(int, int)])],
    stride: int = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to slice

- **index** (`Union[(int, Tuple[(int, int)])]`): int int: Index to select from that dimension [start: int, length: int]: Index range to select from that dimension

- **stride** (`int`, default: `0`): int Stride amount along that dimension

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Sigmoid

Sigmoid

**Function Signature**

```python
forge.op.Sigmoid(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

**Mathematical Definition**

```
sigmoid(x) = 1 / (1 + exp(-x))
```

The output is always in the range (0, 1).

---
### Sine

Elementwise sine

**Function Signature**

```python
forge.op.Sine(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Softmax

Softmax operation.

**Function Signature**

```python
forge.op.Softmax(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Sqrt

Square root.

**Function Signature**

```python
forge.op.Sqrt(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Squeeze

TM

**Function Signature**

```python
forge.op.Squeeze(name: str, operandA: Tensor, dim: int) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to broadcast

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Stack

Stack tensors along new axis

**Function Signature**

```python
forge.op.Stack(name: str) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Subtract

Elementwise subtraction of two tensors

**Function Signature**

```python
forge.op.Subtract(name: str, operandA: Tensor, operandB: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **operandB** (`Tensor`): Tensor Second operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Tanh

Tanh operation.

**Function Signature**

```python
forge.op.Tanh(name: str, operandA: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

**Mathematical Definition**

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

The output is always in the range (-1, 1).

---
### Transpose

Tranpose X and Y (i.e. rows and columns) dimensions.

**Function Signature**

```python
forge.op.Transpose(name: str, operandA: Tensor, dim0: int, dim1: int) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor First operand

- **dim0** (`int`): dim0 parameter

- **dim1** (`int`): dim1 parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Unsqueeze

TM

**Function Signature**

```python
forge.op.Unsqueeze(name: str, operandA: Tensor, dim: int) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **dim** (`int`): int Dimension to broadcast

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### UpdateCache

UpdateCache writes a single token (S=1) slice into the cache tensor on specified index.

**Function Signature**

```python
forge.op.UpdateCache(
    name: str,
    cache: Tensor,
    input: Tensor,
    update_index: int,
    batch_offset: int = 0
) -> Tensor
```

**Parameters**

- **name** (`str`): str Unique op name.

- **cache** (`Tensor`): Tensor 4D cache tensor of shape [B, H, S_total, D]

- **input** (`Tensor`): Tensor 4D input tensor of shape [B, H, 1, D]

- **update_index** (`int`): update_index parameter

- **batch_offset** (`int`, default: `0`): int Offset in the batch dimension.

**Returns**

- **result** (`Tensor`): Output tensor

---
### Upsample2d

Upsample 2D operation

**Function Signature**

```python
forge.op.Upsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[(int, List[int], Tuple[(int, int)])],
    mode: str = 'nearest',
    channel_last: bool = False
) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **operandA** (`Tensor`): Tensor Input operand A

- **scale_factor** (`Union[(int, List[int], Tuple[(int, int)])]`): Union[int, List[int], Tuple[int, int]] multiplier for spatial size.

- **mode** (`str`, default: `'nearest'`): str the upsampling algorithm

- **channel_last** (`bool`, default: `False`): channel_last parameter

**Returns**

- **result** (`Tensor`): Tensor Forge tensor

---
### Where



**Function Signature**

```python
forge.op.Where(name: str, condition: Tensor, x: Tensor, y: Tensor) -> Tensor
```

**Parameters**

- **name** (`str`): str Op name, unique to the module, or leave blank to autoset

- **condition** (`Tensor`): Tensor When True (nonzero), yield x, else y

- **x** (`Tensor`): Tensor value(s) if true

- **y** (`Tensor`): Tensor value(s) if false

**Returns**

- **result** (`Tensor`): Parameters name: str Op name, unique to the module, or leave blank to autoset condition: Tensor When True (nonzero), yield x, else y x: Tensor value(s) if true y: Tensor value(s) if false Tensor Forge tensor

---
*This documentation is automatically generated from operation definitions in `forge/forge/op/*.py`. For the most up-to-date information, refer to the source code.*
