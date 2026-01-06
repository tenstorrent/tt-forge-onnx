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
| **Abs** | Computes the elementwise absolute value of the input tensor. | [forge.op.Abs](./operations/abs.md) |
| **Add** | Elementwise add of two tensors | [forge.op.Add](./operations/add.md) |
| **Atan** | Elementwise arctangent (atan) | [forge.op.Atan](./operations/atan.md) |
| **BitwiseAnd** | Bitwise and operation. | [forge.op.BitwiseAnd](./operations/bitwiseand.md) |
| **Cast** | Cast | [forge.op.Cast](./operations/cast.md) |
| **Clip** | Clips tensor values between min and max | [forge.op.Clip](./operations/clip.md) |
| **Concatenate** | Concatenate tensors along axis | [forge.op.Concatenate](./operations/concatenate.md) |
| **Cosine** | Elementwise cosine | [forge.op.Cosine](./operations/cosine.md) |
| **Divide** | Elementwise divide of two tensors | [forge.op.Divide](./operations/divide.md) |
| **Equal** | Elementwise equal of two tensors | [forge.op.Equal](./operations/equal.md) |
| **Erf** | Error function (erf) | [forge.op.Erf](./operations/erf.md) |
| **Exp** | Exponent operation. | [forge.op.Exp](./operations/exp.md) |
| **Greater** | Elementwise greater of two tensors | [forge.op.Greater](./operations/greater.md) |
| **GreaterEqual** | Elementwise greater or equal of two tensors | [forge.op.GreaterEqual](./operations/greaterequal.md) |
| **Heaviside** | Elementwise max of two tensors | [forge.op.Heaviside](./operations/heaviside.md) |
| **Identity** | Identity operation. | [forge.op.Identity](./operations/identity.md) |
| **IndexCopy** | Copies the elements of value into operandA at index along dim | [forge.op.IndexCopy](./operations/indexcopy.md) |
| **Less** | Elementwise less of two tensors | [forge.op.Less](./operations/less.md) |
| **LessEqual** | Elementwise less or equal of two tensors | [forge.op.LessEqual](./operations/lessequal.md) |
| **Log** | Log operation: natural logarithm of the elements of `operandA` | [forge.op.Log](./operations/log.md) |
| **LogicalAnd** | Logical and operation. | [forge.op.LogicalAnd](./operations/logicaland.md) |
| **LogicalNot** | Logical not operation. | [forge.op.LogicalNot](./operations/logicalnot.md) |
| **Max** | Elementwise max of two tensors | [forge.op.Max](./operations/max.md) |
| **Min** | Elementwise min of two tensors | [forge.op.Min](./operations/min.md) |
| **Multiply** | Elementwise multiply of two tensors | [forge.op.Multiply](./operations/multiply.md) |
| **NotEqual** | Elementwise equal of two tensors | [forge.op.NotEqual](./operations/notequal.md) |
| **Pow** | Pow operation: `operandA` to the power of `exponent` | [forge.op.Pow](./operations/pow.md) |
| **Power** | OperandA to the power of OperandB | [forge.op.Power](./operations/power.md) |
| **Reciprocal** | Reciprocal operation. | [forge.op.Reciprocal](./operations/reciprocal.md) |
| **Remainder** |  | [forge.op.Remainder](./operations/remainder.md) |
| **Sine** | Elementwise sine | [forge.op.Sine](./operations/sine.md) |
| **Sqrt** | Square root. | [forge.op.Sqrt](./operations/sqrt.md) |
| **Stack** | Stack tensors along new axis | [forge.op.Stack](./operations/stack.md) |
| **Subtract** | Elementwise subtraction of two tensors | [forge.op.Subtract](./operations/subtract.md) |
| **Where** |  | [forge.op.Where](./operations/where.md) |

## Convolution Operations

Convolution and related transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Conv2d** | Conv2d transformation on input activations, with optional bias. | [forge.op.Conv2d](./operations/conv2d.md) |
| **Conv2dTranspose** | Conv2dTranspose transformation on input activations, with optional bias. | [forge.op.Conv2dTranspose](./operations/conv2dtranspose.md) |

## Pooling Operations

Pooling and downsampling operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **AvgPool1d** | Avgpool1d transformation on input activations | [forge.op.AvgPool1d](./operations/avgpool1d.md) |
| **AvgPool2d** | Avgpool2d transformation on input activations | [forge.op.AvgPool2d](./operations/avgpool2d.md) |
| **MaxPool1d** | MaxPool1d transformation on input activations | [forge.op.MaxPool1d](./operations/maxpool1d.md) |
| **MaxPool2d** | Maxpool2d transformation on input activations | [forge.op.MaxPool2d](./operations/maxpool2d.md) |

## Normalization Operations

Batch and layer normalization.

| Operation | Description | Link |
|-----------|-------------|------|
| **Batchnorm** | Batch normalization. | [forge.op.Batchnorm](./operations/batchnorm.md) |
| **Dropout** | Dropout | [forge.op.Dropout](./operations/dropout.md) |
| **Layernorm** | Layer normalization. | [forge.op.Layernorm](./operations/layernorm.md) |
| **LogSoftmax** | LogSoftmax operation. | [forge.op.LogSoftmax](./operations/logsoftmax.md) |
| **Softmax** | Softmax operation. | [forge.op.Softmax](./operations/softmax.md) |

## Tensor Manipulation

Reshaping, slicing, and tensor operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **AdvIndex** | TM | [forge.op.AdvIndex](./operations/advindex.md) |
| **Broadcast** | TM | [forge.op.Broadcast](./operations/broadcast.md) |
| **ConstantPad** | TM - Direct TTIR constant padding operation. | [forge.op.ConstantPad](./operations/constantpad.md) |
| **Downsample2d** | Downsample 2D operation | [forge.op.Downsample2d](./operations/downsample2d.md) |
| **Index** | TM | [forge.op.Index](./operations/index.md) |
| **Pad** | TM | [forge.op.Pad](./operations/pad.md) |
| **PixelShuffle** | Pixel shuffle operation. | [forge.op.PixelShuffle](./operations/pixelshuffle.md) |
| **Repeat** | Repeats this tensor along the specified dimensions. | [forge.op.Repeat](./operations/repeat.md) |
| **RepeatInterleave** | Repeat elements of a tensor. | [forge.op.RepeatInterleave](./operations/repeatinterleave.md) |
| **Reshape** | TM | [forge.op.Reshape](./operations/reshape.md) |
| **Resize1d** | Resize input activations, with default mode 'nearest' | [forge.op.Resize1d](./operations/resize1d.md) |
| **Resize2d** | Resizes the spatial dimensions of a 2D input tensor using interpolation. | [forge.op.Resize2d](./operations/resize2d.md) |
| **Select** | TM | [forge.op.Select](./operations/select.md) |
| **Squeeze** | TM | [forge.op.Squeeze](./operations/squeeze.md) |
| **Transpose** | Tranpose X and Y (i.e. rows and columns) dimensions. | [forge.op.Transpose](./operations/transpose.md) |
| **Unsqueeze** | TM | [forge.op.Unsqueeze](./operations/unsqueeze.md) |
| **Upsample2d** | Upsample 2D operation | [forge.op.Upsample2d](./operations/upsample2d.md) |

## Reduction Operations

Aggregation and reduction operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Argmax** | Argmax | [forge.op.Argmax](./operations/argmax.md) |
| **ReduceAvg** | Reduce by averaging along the given dimension | [forge.op.ReduceAvg](./operations/reduceavg.md) |
| **ReduceMax** | Reduce by taking maximum along the given dimension | [forge.op.ReduceMax](./operations/reducemax.md) |
| **ReduceSum** | Reduce by summing along the given dimension | [forge.op.ReduceSum](./operations/reducesum.md) |

## Linear Operations

Matrix multiplication and linear transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Matmul** | Matrix multiplication transformation on input activations, with optional bias. y... | [forge.op.Matmul](./operations/matmul.md) |

## Activation Functions

Non-linear activation functions.

| Operation | Description | Link |
|-----------|-------------|------|
| **Gelu** | GeLU | [forge.op.Gelu](./operations/gelu.md) |
| **LeakyRelu** | Leaky ReLU | [forge.op.LeakyRelu](./operations/leakyrelu.md) |
| **Relu** | Applies the Rectified Linear Unit (ReLU) activation function elementwise. | [forge.op.Relu](./operations/relu.md) |
| **Sigmoid** | Sigmoid | [forge.op.Sigmoid](./operations/sigmoid.md) |
| **Tanh** | Tanh operation. | [forge.op.Tanh](./operations/tanh.md) |

## Memory Operations

Cache and memory management operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **FillCache** | FillCache op writes the input into the cache tensor starting at the specified up... | [forge.op.FillCache](./operations/fillcache.md) |
| **UpdateCache** | UpdateCache writes a single token (S=1) slice into the cache tensor on specified... | [forge.op.UpdateCache](./operations/updatecache.md) |

## Other Operations

Miscellaneous operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Constant** | Op representing user-defined constant | [forge.op.Constant](./operations/constant.md) |
| **CumSum** | Cumulative sum operation. | [forge.op.CumSum](./operations/cumsum.md) |
| **Embedding** | Embedding lookup | [forge.op.Embedding](./operations/embedding.md) |

---

## Documentation Structure

Each operation documentation page includes:

- **Overview**: Brief description of what the operation does
- **Function Signature**: Python API signature with type hints
- **Parameters**: Detailed parameter descriptions with types and defaults
- **Returns**: Return value description
- **Mathematical Definition**: Mathematical formula (where applicable)
- **Related Operations**: Links to related operations

---

*This documentation is automatically generated from operation definitions in `forge/forge/op/*.py`. For the most up-to-date information, refer to the source code.*
