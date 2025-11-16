# Operations Reference

This page provides a comprehensive reference for all supported operations.

Operations are organized by category. Click on an operation name to view detailed documentation.

## Other Operations

- [forge.op.Abs](./operations/forge_op_abs.md)
  Elementwise absolute value operation

- [forge.op.Add](./operations/forge_op_add.md)
  Elementwise add of two tensors

- [forge.op.AdvIndex](./operations/forge_op_advindex.md)
  TM

- [forge.op.Argmax](./operations/forge_op_argmax.md)
  Argmax

- [forge.op.Atan](./operations/forge_op_atan.md)
  Elementwise arctangent (atan)

- [forge.op.AvgPool1d](./operations/forge_op_avgpool1d.md)
  Avgpool1d transformation on input activations

- [forge.op.AvgPool2d](./operations/forge_op_avgpool2d.md)
  Avgpool2d transformation on input activations

- [forge.op.Batchnorm](./operations/forge_op_batchnorm.md)
  Batch normalization.

- [forge.op.BitwiseAnd](./operations/forge_op_bitwiseand.md)
  Bitwise and operation.

- [forge.op.Broadcast](./operations/forge_op_broadcast.md)
  TM

- [forge.op.Cast](./operations/forge_op_cast.md)
  Cast

- [forge.op.Clip](./operations/forge_op_clip.md)
  Clips tensor values between min and max

- [forge.op.Concatenate](./operations/forge_op_concatenate.md)
  Concatenate tensors along axis

- [forge.op.Constant](./operations/forge_op_constant.md)
  Op representing user-defined constant

- [forge.op.ConstantPad](./operations/forge_op_constantpad.md)
  TM - Direct TTIR constant padding operation.

- [forge.op.Conv2d](./operations/forge_op_conv2d.md)
  Conv2d transformation on input activations, with optional bias.

- [forge.op.Conv2dTranspose](./operations/forge_op_conv2dtranspose.md)
  Conv2dTranspose transformation on input activations, with optional bias.

- [forge.op.Cosine](./operations/forge_op_cosine.md)
  Elementwise cosine

- [forge.op.CumSum](./operations/forge_op_cumsum.md)
  Cumulative sum operation.

- [forge.op.Divide](./operations/forge_op_divide.md)
  Elementwise divide of two tensors

- [forge.op.Downsample2d](./operations/forge_op_downsample2d.md)
  Downsample 2D operation

- [forge.op.Dropout](./operations/forge_op_dropout.md)
  Dropout

- [forge.op.Embedding](./operations/forge_op_embedding.md)
  Embedding lookup

- [forge.op.Equal](./operations/forge_op_equal.md)
  Elementwise equal of two tensors

- [forge.op.Erf](./operations/forge_op_erf.md)
  Error function (erf)

- [forge.op.Exp](./operations/forge_op_exp.md)
  Exponent operation.

- [forge.op.FillCache](./operations/forge_op_fillcache.md)
  FillCache op writes the input into the cache tensor starting at the specified update index.

- [forge.op.Gelu](./operations/forge_op_gelu.md)
  GeLU

- [forge.op.Greater](./operations/forge_op_greater.md)
  Elementwise greater of two tensors

- [forge.op.GreaterEqual](./operations/forge_op_greaterequal.md)
  Elementwise greater or equal of two tensors

- [forge.op.Heaviside](./operations/forge_op_heaviside.md)
  Elementwise max of two tensors

- [forge.op.Identity](./operations/forge_op_identity.md)
  Identity operation.

- [forge.op.Index](./operations/forge_op_index.md)
  TM

- [forge.op.IndexCopy](./operations/forge_op_indexcopy.md)
  Copies the elements of value into operandA at index along dim

- [forge.op.Layernorm](./operations/forge_op_layernorm.md)
  Layer normalization.

- [forge.op.LeakyRelu](./operations/forge_op_leakyrelu.md)
  Leaky ReLU

- [forge.op.Less](./operations/forge_op_less.md)
  Elementwise less of two tensors

- [forge.op.LessEqual](./operations/forge_op_lessequal.md)
  Elementwise less or equal of two tensors

- [forge.op.Log](./operations/forge_op_log.md)
  Log operation: natural logarithm of the elements of `operandA`

- [forge.op.LogicalAnd](./operations/forge_op_logicaland.md)
  Logical and operation.

- [forge.op.LogicalNot](./operations/forge_op_logicalnot.md)
  Logical not operation.

- [forge.op.LogSoftmax](./operations/forge_op_logsoftmax.md)
  LogSoftmax operation.

- [forge.op.Matmul](./operations/forge_op_matmul.md)
  Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

- [forge.op.Max](./operations/forge_op_max.md)
  Elementwise max of two tensors

- [forge.op.MaxPool1d](./operations/forge_op_maxpool1d.md)
  MaxPool1d transformation on input activations

- [forge.op.MaxPool2d](./operations/forge_op_maxpool2d.md)
  Maxpool2d transformation on input activations

- [forge.op.Min](./operations/forge_op_min.md)
  Elementwise min of two tensors

- [forge.op.Multiply](./operations/forge_op_multiply.md)
  Elementwise multiply of two tensors

- [forge.op.NotEqual](./operations/forge_op_notequal.md)
  Elementwise equal of two tensors

- [forge.op.Pad](./operations/forge_op_pad.md)
  TM

- [forge.op.PixelShuffle](./operations/forge_op_pixelshuffle.md)
  Pixel shuffle operation.

- [forge.op.Pow](./operations/forge_op_pow.md)
  Pow operation: `operandA` to the power of `exponent`

- [forge.op.Power](./operations/forge_op_power.md)
  OperandA to the power of OperandB

- [forge.op.Reciprocal](./operations/forge_op_reciprocal.md)
  Reciprocal operation.

- [forge.op.ReduceAvg](./operations/forge_op_reduceavg.md)
  Reduce by averaging along the given dimension

- [forge.op.ReduceMax](./operations/forge_op_reducemax.md)
  Reduce by taking maximum along the given dimension

- [forge.op.ReduceSum](./operations/forge_op_reducesum.md)
  Reduce by summing along the given dimension

- [forge.op.Relu](./operations/forge_op_relu.md)
  ReLU

- [forge.op.Remainder](./operations/forge_op_remainder.md)
  Remainder

- [forge.op.Repeat](./operations/forge_op_repeat.md)
  Repeats this tensor along the specified dimensions.

- [forge.op.RepeatInterleave](./operations/forge_op_repeatinterleave.md)
  Repeat elements of a tensor.

- [forge.op.Reshape](./operations/forge_op_reshape.md)
  TM

- [forge.op.Resize1d](./operations/forge_op_resize1d.md)
  Resize input activations, with default mode 'nearest'

- [forge.op.Resize2d](./operations/forge_op_resize2d.md)
  Resize input activations, with default mode 'nearest'

- [forge.op.Select](./operations/forge_op_select.md)
  TM

- [forge.op.Sigmoid](./operations/forge_op_sigmoid.md)
  Sigmoid activation function

- [forge.op.Sine](./operations/forge_op_sine.md)
  Elementwise sine

- [forge.op.Softmax](./operations/forge_op_softmax.md)
  Softmax operation.

- [forge.op.Sqrt](./operations/forge_op_sqrt.md)
  Square root.

- [forge.op.Squeeze](./operations/forge_op_squeeze.md)
  TM

- [forge.op.Stack](./operations/forge_op_stack.md)
  Stack tensors along new axis

- [forge.op.Subtract](./operations/forge_op_subtract.md)
  Elementwise subtraction of two tensors

- [forge.op.Tanh](./operations/forge_op_tanh.md)
  Tanh operation.

- [forge.op.Transpose](./operations/forge_op_transpose.md)
  Tranpose X and Y (i.e. rows and columns) dimensions.

- [forge.op.Unsqueeze](./operations/forge_op_unsqueeze.md)
  TM

- [forge.op.UpdateCache](./operations/forge_op_updatecache.md)
  UpdateCache writes a single token (S=1) slice into the cache tensor on specified index.

- [forge.op.Upsample2d](./operations/forge_op_upsample2d.md)
  Upsample 2D operation

- [forge.op.Where](./operations/forge_op_where.md)
  Where


---

*This documentation is automatically generated from operation definitions.*
