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
- [Creation Operations](#creation-operations) - Operations in this category
- [Embedding Functions](#embedding-functions) - Operations in this category
- [Resize Operations](#resize-operations) - Operations in this category

---

## Elementwise Operations

Elementwise operations apply a function to each element of the input tensor(s) independently.

| Operation | Description | Link |
|-----------|-------------|------|
| **Abs** | Elementwise absolute value operation | [forge.op.Abs](./operations/forge_op_abs.md) |
| **Add** | Elementwise add of two tensors | [forge.op.Add](./operations/forge_op_add.md) |
| **Atan** | Elementwise arctangent (atan) | [forge.op.Atan](./operations/forge_op_atan.md) |
| **BitwiseAnd** | Bitwise and operation. | [forge.op.BitwiseAnd](./operations/forge_op_bitwiseand.md) |
| **Cast** | Cast | [forge.op.Cast](./operations/forge_op_cast.md) |
| **Clip** | Clips tensor values between min and max | [forge.op.Clip](./operations/forge_op_clip.md) |
| **Concatenate** | Concatenate tensors along axis | [forge.op.Concatenate](./operations/forge_op_concatenate.md) |
| **Cosine** | Elementwise cosine | [forge.op.Cosine](./operations/forge_op_cosine.md) |
| **Divide** | Elementwise divide of two tensors | [forge.op.Divide](./operations/forge_op_divide.md) |
| **Equal** | Elementwise equal of two tensors | [forge.op.Equal](./operations/forge_op_equal.md) |
| **Erf** | Error function (erf) | [forge.op.Erf](./operations/forge_op_erf.md) |
| **Exp** | Exponent operation. | [forge.op.Exp](./operations/forge_op_exp.md) |
| **Greater** | Elementwise greater of two tensors | [forge.op.Greater](./operations/forge_op_greater.md) |
| **GreaterEqual** | Elementwise greater or equal of two tensors | [forge.op.GreaterEqual](./operations/forge_op_greaterequal.md) |
| **Heaviside** | Elementwise max of two tensors | [forge.op.Heaviside](./operations/forge_op_heaviside.md) |
| **Identity** | Identity operation. | [forge.op.Identity](./operations/forge_op_identity.md) |
| **IndexCopy** | Copies the elements of value into operandA at index along dim | [forge.op.IndexCopy](./operations/forge_op_indexcopy.md) |
| **Less** | Elementwise less of two tensors | [forge.op.Less](./operations/forge_op_less.md) |
| **LessEqual** | Elementwise less or equal of two tensors | [forge.op.LessEqual](./operations/forge_op_lessequal.md) |
| **Log** | Log operation: natural logarithm of the elements of `operandA` | [forge.op.Log](./operations/forge_op_log.md) |
| **LogicalAnd** | Logical and operation. | [forge.op.LogicalAnd](./operations/forge_op_logicaland.md) |
| **LogicalNot** | Logical not operation. | [forge.op.LogicalNot](./operations/forge_op_logicalnot.md) |
| **Max** | Elementwise max of two tensors | [forge.op.Max](./operations/forge_op_max.md) |
| **Min** | Elementwise min of two tensors | [forge.op.Min](./operations/forge_op_min.md) |
| **Multiply** | Elementwise multiply of two tensors | [forge.op.Multiply](./operations/forge_op_multiply.md) |
| **NotEqual** | Elementwise equal of two tensors | [forge.op.NotEqual](./operations/forge_op_notequal.md) |
| **Pow** | Pow operation: `operandA` to the power of `exponent` | [forge.op.Pow](./operations/forge_op_pow.md) |
| **Power** | OperandA to the power of OperandB | [forge.op.Power](./operations/forge_op_power.md) |
| **Reciprocal** | Reciprocal operation. | [forge.op.Reciprocal](./operations/forge_op_reciprocal.md) |
| **Remainder** | Remainder | [forge.op.Remainder](./operations/forge_op_remainder.md) |
| **Sine** | Elementwise sine | [forge.op.Sine](./operations/forge_op_sine.md) |
| **Sqrt** | Square root. | [forge.op.Sqrt](./operations/forge_op_sqrt.md) |
| **Stack** | Stack tensors along new axis | [forge.op.Stack](./operations/forge_op_stack.md) |
| **Subtract** | Elementwise subtraction of two tensors | [forge.op.Subtract](./operations/forge_op_subtract.md) |
| **Where** | Where | [forge.op.Where](./operations/forge_op_where.md) |

## Convolution Operations

Convolution operations perform spatial filtering and feature extraction.

| Operation | Description | Link |
|-----------|-------------|------|
| **Conv2d** | Conv2d transformation on input activations, with optional bias. | [forge.op.Conv2d](./operations/forge_op_conv2d.md) |
| **Conv2dTranspose** | Conv2dTranspose transformation on input activations, with optional bias. | [forge.op.Conv2dTranspose](./operations/forge_op_conv2dtranspose.md) |

## Pooling Operations

Pooling operations reduce spatial dimensions and provide translation invariance.

| Operation | Description | Link |
|-----------|-------------|------|
| **AvgPool1d** | Avgpool1d transformation on input activations | [forge.op.AvgPool1d](./operations/forge_op_avgpool1d.md) |
| **AvgPool2d** | Avgpool2d transformation on input activations | [forge.op.AvgPool2d](./operations/forge_op_avgpool2d.md) |
| **MaxPool1d** | MaxPool1d transformation on input activations | [forge.op.MaxPool1d](./operations/forge_op_maxpool1d.md) |
| **MaxPool2d** | Maxpool2d transformation on input activations | [forge.op.MaxPool2d](./operations/forge_op_maxpool2d.md) |

## Normalization Operations

Normalization operations stabilize training and improve convergence.

| Operation | Description | Link |
|-----------|-------------|------|
| **Batchnorm** | Batch normalization. | [forge.op.Batchnorm](./operations/forge_op_batchnorm.md) |
| **Dropout** | Dropout | [forge.op.Dropout](./operations/forge_op_dropout.md) |
| **Layernorm** | Layer normalization. | [forge.op.Layernorm](./operations/forge_op_layernorm.md) |
| **LogSoftmax** | LogSoftmax operation. | [forge.op.LogSoftmax](./operations/forge_op_logsoftmax.md) |
| **Softmax** | Softmax operation. | [forge.op.Softmax](./operations/forge_op_softmax.md) |

## Tensor Manipulation

Operations for reshaping, slicing, and manipulating tensor structure.

| Operation | Description | Link |
|-----------|-------------|------|
| **AdvIndex** | TM | [forge.op.AdvIndex](./operations/forge_op_advindex.md) |
| **Broadcast** | Broadcast tensor manipulation operation | [forge.op.Broadcast](./operations/forge_op_broadcast.md) |
| **ConstantPad** | TM - Direct Forge constant padding operation. | [forge.op.ConstantPad](./operations/forge_op_constantpad.md) |
| **Index** | Index tensor manipulation operation | [forge.op.Index](./operations/forge_op_index.md) |
| **Pad** | Pad tensor manipulation operation | [forge.op.Pad](./operations/forge_op_pad.md) |
| **PixelShuffle** | Pixel shuffle operation. | [forge.op.PixelShuffle](./operations/forge_op_pixelshuffle.md) |
| **Repeat** | Repeats this tensor along the specified dimensions. | [forge.op.Repeat](./operations/forge_op_repeat.md) |
| **RepeatInterleave** | Repeat elements of a tensor. | [forge.op.RepeatInterleave](./operations/forge_op_repeatinterleave.md) |
| **Reshape** | Reshape tensor manipulation operation | [forge.op.Reshape](./operations/forge_op_reshape.md) |
| **Select** | Select tensor manipulation operation | [forge.op.Select](./operations/forge_op_select.md) |
| **Squeeze** | Squeeze tensor manipulation operation | [forge.op.Squeeze](./operations/forge_op_squeeze.md) |
| **Transpose** | Tranpose X and Y (i.e. rows and columns) dimensions. | [forge.op.Transpose](./operations/forge_op_transpose.md) |
| **Unsqueeze** | Unsqueeze tensor manipulation operation | [forge.op.Unsqueeze](./operations/forge_op_unsqueeze.md) |

## Reduction Operations

Operations that reduce tensor dimensions by aggregation.

| Operation | Description | Link |
|-----------|-------------|------|
| **Argmax** | Argmax | [forge.op.Argmax](./operations/forge_op_argmax.md) |
| **ReduceAvg** | Reduce by averaging along the given dimension | [forge.op.ReduceAvg](./operations/forge_op_reduceavg.md) |
| **ReduceMax** | Reduce by taking maximum along the given dimension | [forge.op.ReduceMax](./operations/forge_op_reducemax.md) |
| **ReduceSum** | Reduce by summing along the given dimension | [forge.op.ReduceSum](./operations/forge_op_reducesum.md) |

## Linear Operations

Matrix multiplication and linear transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Matmul** | Matrix multiplication transformation on input activations, with optional bias. y = ab + bias | [forge.op.Matmul](./operations/forge_op_matmul.md) |

## Activation Functions

Non-linear activation functions for neural networks.

| Operation | Description | Link |
|-----------|-------------|------|
| **Gelu** | GeLU | [forge.op.Gelu](./operations/forge_op_gelu.md) |
| **LeakyRelu** | Leaky ReLU | [forge.op.LeakyRelu](./operations/forge_op_leakyrelu.md) |
| **Relu** | ReLU | [forge.op.Relu](./operations/forge_op_relu.md) |
| **Sigmoid** | Sigmoid activation function | [forge.op.Sigmoid](./operations/forge_op_sigmoid.md) |
| **Tanh** | Tanh operation. | [forge.op.Tanh](./operations/forge_op_tanh.md) |

## Memory Operations

Operations for cache and memory management.

| Operation | Description | Link |
|-----------|-------------|------|
| **FillCache** | FillCache op writes the input into the cache tensor starting at the specified update index. | [forge.op.FillCache](./operations/forge_op_fillcache.md) |
| **UpdateCache** | UpdateCache writes a single token (S=1) slice into the cache tensor on specified index. | [forge.op.UpdateCache](./operations/forge_op_updatecache.md) |

## Other Operations

Miscellaneous operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **CumSum** | Cumulative sum operation. | [forge.op.CumSum](./operations/forge_op_cumsum.md) |

## Creation Operations

| Operation | Description | Link |
|-----------|-------------|------|
| **Constant** | Op representing user-defined constant | [forge.op.Constant](./operations/forge_op_constant.md) |

## Embedding Functions

| Operation | Description | Link |
|-----------|-------------|------|
| **Embedding** | Embedding lookup | [forge.op.Embedding](./operations/forge_op_embedding.md) |

## Resize Operations

| Operation | Description | Link |
|-----------|-------------|------|
| **Downsample2d** | Downsample 2D operation | [forge.op.Downsample2d](./operations/forge_op_downsample2d.md) |
| **Resize1d** | Resize input activations, with default mode 'nearest' | [forge.op.Resize1d](./operations/forge_op_resize1d.md) |
| **Resize2d** | Resize input activations, with default mode 'nearest' | [forge.op.Resize2d](./operations/forge_op_resize2d.md) |
| **Upsample2d** | Upsample 2D operation | [forge.op.Upsample2d](./operations/forge_op_upsample2d.md) |

---

## Documentation Structure

Each operation documentation page includes:

- **Overview**: Brief description of what the operation does

- **Function Signature**: Python API signature with type hints

- **Parameters**: Detailed parameter descriptions with types and defaults

- **Returns**: Return value description

- **Mathematical Definition**: Mathematical formula (where applicable)

- **Examples**: Code examples demonstrating usage

- **Notes**: Important implementation details and warnings

- **Related Operations**: Links to related operations


---

*This documentation is automatically generated from operation definitions. For the most up-to-date information, refer to the source code.*
