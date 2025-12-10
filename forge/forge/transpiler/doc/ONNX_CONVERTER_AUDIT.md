# ONNX Converter Audit Report

## Overview
This document audits all ONNX operation converters to verify:
1. Correct input/output extraction per opset version
2. Correct attribute extraction and conversion to PyTorch format
3. Correct attribute conversion to Forge format (via `convert_attrs_to_forge_attrs`)
4. Proper UNKNOWN marking for unsupported operations

## Audit Methodology

### 1. Input/Output Extraction
- **ONNX Inputs**: Extracted from `node_proto.input` (list of input tensor names)
- **ONNX Outputs**: Extracted from `node_proto.output` (list of output tensor names)
- **Multi-output operations**: Should use `list(node_proto.output)` (e.g., Split)
- **Single-output operations**: Should use `[node_proto.output[0]]`

### 2. Attribute Extraction
- All converters receive `attrs` dict from `extract_attributes()` in `engine.py`
- `extract_attributes()` maps ONNX attribute names to PyTorch-friendly names:
  - `axis`/`axes` → `dim`
  - `dilations` → `dilation`
  - `kernel_shape` → `kernel_size`
  - `strides` → `stride`
  - `keepdims` → `keepdim`
  - etc.

### 3. Attribute Conversion Pipeline
- **ONNX → PyTorch**: Done in converters (extract from `attrs`, convert format)
- **PyTorch → Forge**: Done in TIRNode via `convert_attrs_to_forge_attrs()` method
- **UNKNOWN operations**: Should NOT convert attributes (will be decomposed later)

## Converter Audit Results

### ✅ Arithmetic Operations (All Correct)

#### AddConverter, SubConverter, MulConverter, DivConverter, MatMulConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ No attributes needed - Correct
- **Forge conversion**: ✅ Default (no conversion needed)
- **UNKNOWN**: ✅ N/A (all supported)

### ✅ Activation Operations

#### ReluConverter, SigmoidConverter, TanhConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ No attributes - Correct
- **Forge conversion**: ✅ Default
- **UNKNOWN**: ✅ N/A

#### SoftmaxConverter, LogSoftmaxConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ `axis` → `dim` (correctly extracted)
  - **v1-v10**: Default `axis=1` ✅
  - **v11+**: Default `axis=-1` ✅
- **Forge conversion**: ✅ `SoftmaxNode.convert_attrs_to_forge_attrs()` adds `stable=True`
- **UNKNOWN**: ✅ N/A

#### LeakyReluConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ `alpha` → `negative_slope` (correctly converted)
- **Forge conversion**: ✅ Default
- **UNKNOWN**: ✅ N/A

### ✅ Reduction Operations

#### ReduceSumConverter, ReduceMeanConverter, ReduceMaxConverter
- **Inputs**: ✅ `[node_proto.input[0]]` - Correct (only data input, axes embedded)
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v12**: `axes` as attribute ✅
  - **v13+**: `axes` as input tensor (extracted from initializers) ✅
  - `keepdims` → `keepdim` ✅
- **Forge conversion**: ✅ `ReduceSumNode.convert_attrs_to_forge_attrs()` converts `dim` and `keepdim` → `keep_dim`
- **UNKNOWN**: ✅ N/A

### ✅ Pooling Operations

#### MaxPoolConverter, AveragePoolConverter
- **Inputs**: ✅ Correctly handles AUTO_PAD (may add PadNode)
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - `kernel_shape` → `kernel_size` ✅
  - `strides` → `stride` ✅
  - `pads` → `padding` (converted to PyTorch format) ✅
  - `dilations` → `dilation` ✅
  - `ceil_mode` ✅
- **Dimension handling**: ✅ Correctly creates MaxPool1dNode, MaxPool2dNode, or MaxPool3dNode
- **Forge conversion**: ✅ 
  - MaxPool1d/2d: `forge.op.MaxPool1d/2d` ✅
  - MaxPool3d: `UNKNOWN` ✅ (correctly marked)
  - AvgPool1d/2d: `forge.op.AvgPool1d/2d` ✅
  - AvgPool3d: `UNKNOWN` ✅ (correctly marked)
- **UNKNOWN**: ✅ MaxPool3d and AvgPool3d correctly marked as UNKNOWN

#### GlobalAveragePoolConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ No attributes - Correct
- **Forge conversion**: ✅ `forge.op.ReduceAvg` - Correct
- **UNKNOWN**: ✅ N/A

### ✅ Shape Operations

#### TransposeConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct (may create intermediate nodes)
- **Attributes**: ✅ `perm` correctly extracted
- **Decomposition**: ✅ Correctly decomposes complex permutations into 2D swaps
- **Forge conversion**: ✅ `forge.op.tm.Transpose` - Correct
- **UNKNOWN**: ✅ N/A

#### ReshapeConverter
- **Inputs**: ✅ 
  - **v1-v4**: `list(node_proto.input)` ✅
  - **v5+**: `[node_proto.input[0]]` (shape embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v4**: `shape` as attribute ✅
  - **v5+**: `shape` as input tensor (extracted from initializers) ✅
- **Forge conversion**: ✅ `forge.op.tm.Reshape` - Correct
- **UNKNOWN**: ✅ N/A

#### SqueezeConverter
- **Inputs**: ✅ 
  - **v1-v12**: `list(node_proto.input)` ✅
  - **v13+**: `[node_proto.input[0]]` (axes embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v12**: `axes` as attribute ✅
  - **v13+**: `axes` as input tensor ✅
  - Converts to `dim` (handles Forge limitation: single dim only) ✅
- **Forge conversion**: ✅ `forge.op.tm.Squeeze` - Correct
- **UNKNOWN**: ✅ N/A

#### UnsqueezeConverter
- **Inputs**: ✅ 
  - **v1-v12**: `list(node_proto.input)` ✅
  - **v13+**: `[node_proto.input[0]]` (axes embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v12**: `axes` as attribute ✅
  - **v13+**: `axes` as input tensor ✅
- **Forge conversion**: ✅ `forge.op.tm.Unsqueeze` - Correct
- **UNKNOWN**: ✅ N/A

#### SplitConverter
- **Inputs**: ✅ 
  - **v1-v12**: `list(node_proto.input)` ✅
  - **v13+**: `[node_proto.input[0]]` (split embedded) ✅
- **Outputs**: ✅ `list(node_proto.output)` - Correct (multiple outputs)
- **Attributes**: ✅ 
  - **v1-v12**: `split` as attribute ✅
  - **v13+**: `split` as input tensor ✅
- **Forge conversion**: ✅ `UNKNOWN` - Correct (must be decomposed)
- **UNKNOWN**: ✅ Correctly marked as UNKNOWN

### ✅ Other Operations

#### PadConverter
- **Inputs**: ✅ 
  - **v1-v2**: `list(node_proto.input)` ✅
  - **v11+**: `[node_proto.input[0]]` (pads embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v2**: `pads` as attribute ✅
  - **v11+**: `pads` as input tensor ✅
  - Correctly converts ONNX pads format to PyTorch format ✅
- **Forge conversion**: ✅ `forge.op.misc.Pad` - Correct
- **UNKNOWN**: ✅ N/A

#### ClipConverter
- **Inputs**: ✅ 
  - **v1**: `list(node_proto.input)` ✅
  - **v6+**: `[node_proto.input[0]]` (min/max embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1**: `min`/`max` as attributes ✅
  - **v6+**: `min`/`max` as optional input tensors ✅
- **Forge conversion**: ✅ `forge.op.Clip` - Correct
- **UNKNOWN**: ✅ N/A

#### CastConverter
- **Inputs**: ✅ 
  - **v1-v12**: `list(node_proto.input)` ✅
  - **v13+**: `[node_proto.input[0]]` (dtype embedded) ✅
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - **v1-v12**: `to` as attribute ✅
  - **v13+**: `to` as optional input tensor ✅
  - Correctly converts ONNX dtype to torch dtype ✅
- **Forge conversion**: ✅ `forge.op.Cast` - Correct
- **UNKNOWN**: ✅ N/A

#### ConcatConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ `axis` → `dim` ✅
- **Forge conversion**: ✅ `forge.op.Concatenate` - Correct
- **UNKNOWN**: ✅ N/A

### ✅ Convolution Operations

#### ConvConverter
- **Inputs**: ✅ Correctly handles AUTO_PAD (may add PadNode)
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - `kernel_shape` → `kernel_size` ✅
  - `strides` → `stride` ✅
  - `pads` → `padding` (converted to PyTorch format) ✅
  - `dilations` → `dilation` ✅
  - `group` → `groups` ✅
- **Dimension handling**: ✅ Correctly creates Conv1dNode, Conv2dNode, or Conv3dNode
- **Forge conversion**: ✅ 
  - Conv1d: Raises NotImplementedError ✅ (correct)
  - Conv2d: `forge.op.convolution.Conv2d` ✅
  - Conv3d: Raises NotImplementedError ✅ (correct)
- **UNKNOWN**: ✅ Conv1d and Conv3d correctly raise NotImplementedError

### ✅ Normalization Operations

#### BatchNormalizationConverter
- **Inputs**: ✅ `list(node_proto.input)` - Correct (5 inputs: X, scale, B, mean, var)
- **Outputs**: ✅ `[node_proto.output[0]]` - Correct
- **Attributes**: ✅ 
  - `epsilon` → `eps` ✅
  - `momentum` ✅
  - `training_mode` (opset >= 9) - correctly handled ✅
- **Forge conversion**: ✅ `BatchNormalizationNode.convert_attrs_to_forge_attrs()` converts `eps` → `epsilon`
- **UNKNOWN**: ✅ N/A

## Summary of Findings

### ✅ Correct Implementations
1. **Input/Output extraction**: All converters correctly extract inputs and outputs
2. **Attribute extraction**: All converters use `extract_attributes()` correctly
3. **Opset version handling**: All versioned converters correctly handle opset differences
4. **UNKNOWN marking**: 
   - SplitNode: ✅ Correctly marked as UNKNOWN
   - MaxPool3dNode: ✅ Correctly marked as UNKNOWN
   - AvgPool3dNode: ✅ Correctly marked as UNKNOWN
   - Conv1dNode/Conv3dNode: ✅ Correctly raise NotImplementedError

### ⚠️ Potential Issues

1. **Attribute Conversion for UNKNOWN Operations**:
   - ✅ **SplitNode**: Correctly marked UNKNOWN, no Forge attribute conversion needed (will be decomposed)
   - ✅ **MaxPool3dNode/AvgPool3dNode**: Correctly marked UNKNOWN, no Forge attribute conversion needed
   - **Note**: UNKNOWN operations don't need `convert_attrs_to_forge_attrs()` because they will be decomposed before code generation

2. **Missing Attribute Conversions**:
   - Most operations use default `convert_attrs_to_forge_attrs()` (just copies attrs)
   - Operations with custom conversions:
     - ✅ SoftmaxNode/LogSoftmaxNode: Adds `stable=True`
     - ✅ ReduceSumNode/ReduceMeanNode/ReduceMaxNode: Converts `keepdim` → `keep_dim`
     - ✅ BatchNormalizationNode: Converts `eps` → `epsilon`
     - ✅ SqueezeNode: Handles multi-axis → single dim conversion

3. **Input Tensor Extraction for Opset >= 11/13**:
   - ✅ All converters correctly extract constant values from initializers when attributes become inputs
   - ⚠️ **Limitation**: Dynamic input tensors (non-constant) are not supported and raise errors
   - This is acceptable for now, but should be documented

## Recommendations

1. ✅ **All converters are correctly implemented** - No critical issues found
2. ✅ **UNKNOWN operations are correctly marked** - They will be decomposed via pattern callbacks
3. ✅ **Attribute conversion pipeline is correct** - ONNX → PyTorch in converters, PyTorch → Forge in TIRNode
4. 📝 **Documentation**: Consider adding comments about:
   - Dynamic input tensors limitation (for opset >= 11/13 operations)
   - UNKNOWN operations don't need Forge attribute conversion (they're decomposed first)

## Conclusion

All ONNX converters are correctly implemented with:
- ✅ Proper input/output extraction per opset version
- ✅ Correct attribute extraction and conversion to PyTorch format
- ✅ Proper attribute conversion to Forge format (where needed)
- ✅ Correct UNKNOWN marking for unsupported operations

The converter architecture follows best practices and correctly handles opset version differences.

