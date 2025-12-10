# Transpiler Documentation

This directory contains comprehensive documentation for the ONNX transpiler implementation.

## Documentation Files

### Operator Documentation
Complete summaries and implementation guides for ONNX operators:

- **ONNX_ADD_COMPLETE_SUMMARY.md** - Addition operator documentation
- **ONNX_AVGPOOL_COMPLETE_SUMMARY.md** - Average pooling operator documentation
- **ONNX_AVGPOOL2D_DOCUMENTATION.md** - 2D Average pooling specific documentation
- **ONNX_CLIP_COMPLETE_SUMMARY.md** - Clip operator documentation
- **ONNX_CONCAT_COMPLETE_SUMMARY.md** - Concatenation operator documentation
- **ONNX_CONV_COMPLETE_SUMMARY.md** - Convolution operator complete summary
- **ONNX_CONV_IMPLEMENTATION_GUIDE.md** - Convolution implementation guide
- **ONNX_DIV_COMPLETE_SUMMARY.md** - Division operator documentation
- **ONNX_DROPOUT_COMPLETE_SUMMARY.md** - Dropout operator documentation
- **ONNX_FLATTEN_COMPLETE_SUMMARY.md** - Flatten operator documentation
- **ONNX_GEMM_COMPLETE_SUMMARY.md** - General Matrix Multiplication operator documentation
- **ONNX_LOGSOFTMAX_COMPLETE_SUMMARY.md** - LogSoftmax operator documentation
- **ONNX_MAXPOOL_COMPLETE_SUMMARY.md** - Max pooling operator documentation
- **ONNX_MUL_COMPLETE_SUMMARY.md** - Multiplication operator documentation
- **ONNX_PAD_COMPLETE_SUMMARY.md** - Padding operator documentation
- **ONNX_REDUCEMAX_COMPLETE_SUMMARY.md** - ReduceMax operator documentation
- **ONNX_REDUCEMEAN_COMPLETE_SUMMARY.md** - ReduceMean operator documentation
- **ONNX_RELU_COMPLETE_SUMMARY.md** - ReLU activation operator documentation
- **ONNX_RESHAPE_COMPLETE_SUMMARY.md** - Reshape operator documentation
- **ONNX_SOFTMAX_COMPLETE_SUMMARY.md** - Softmax operator documentation
- **ONNX_SQUEEZE_COMPLETE_SUMMARY.md** - Squeeze operator documentation
- **ONNX_SUB_COMPLETE_SUMMARY.md** - Subtraction operator documentation
- **ONNX_TRANSPOSE_COMPLETE_SUMMARY.md** - Transpose operator documentation
- **ONNX_UNSQUEEZE_COMPLETE_SUMMARY.md** - Unsqueeze operator documentation

### Implementation Guides and Notes
- **ONNX_CONVERTER_AUDIT.md** - Converter implementation audit
- **MULTI_OUTPUT_APPROACH.md** - Approach for handling multi-output operations
- **ONNX_MULTI_OUTPUT_OPS.md** - Multi-output operations documentation
- **SPLIT_MAPPING_SUMMARY.md** - Split operation mapping summary
- **SQUEEZE_IMPLEMENTATION_EXAMPLES.md** - Squeeze implementation examples
- **TORCH_PAD_EXPLANATION.md** - PyTorch padding explanation
- **TRANSPOSE_TEST_ISSUES_AND_FIXES.md** - Transpose test issues and fixes

## Purpose

This documentation serves as a reference for:
- Understanding ONNX operator specifications across different opset versions
- Implementation details and design decisions
- Testing strategies and known issues
- Future development and maintenance

## Structure

Each operator documentation typically includes:
- Operator specification across opset versions
- Attribute and input/output descriptions
- Type constraints
- PyTorch mapping information
- Implementation examples
- Version-specific changes and comparisons

