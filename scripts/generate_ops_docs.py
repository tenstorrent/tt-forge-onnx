#!/usr/bin/env python3
"""
Script to generate operation documentation in PyTorch-style format.

This script generates:
1. An index page with operations grouped by categories
2. Individual pages for each operation with detailed information
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import json


@dataclass
class Operand:
    """Represents an operand (input/output) of an operation."""
    name: str
    description: str
    type: str = "ranked tensor of any type values"


@dataclass
class Attribute:
    """Represents an attribute of an operation."""
    name: str
    mlir_type: str
    description: str
    default: Optional[str] = None


@dataclass
class Operation:
    """Represents a complete operation definition."""
    name: str  # e.g., "forge.op.Abs"
    short_name: str  # e.g., "abs"
    category: str
    description: str
    detailed_description: str = ""
    mathematical_definition: str = ""
    traits: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    operands: List[Operand] = field(default_factory=list)
    results: List[Operand] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    signature: str = ""  # Python function signature


# Operation categories mapping
CATEGORIES = {
    "Elementwise Operations": [
        "abs", "add", "atan", "atan2", "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor",
        "cbrt", "ceil", "cos", "div", "eq", "erf", "erfc", "exp", "expm1", "floor", "ge", "gelu",
        "gt", "hardsigmoid", "isfinite", "le", "leaky_relu", "log", "log1p", "logical_and",
        "logical_left_shift", "logical_not", "logical_or", "logical_right_shift", "logical_xor",
        "lt", "maximum", "minimum", "multiply", "neg", "ne", "pow", "reciprocal", "relu", "relu6",
        "remainder", "rsqrt", "sigmoid", "sign", "silu", "sin", "sqrt", "subtract", "tan", "tanh"
    ],
    "Convolution Operations": [
        "conv2d", "conv_transpose2d", "convolution"
    ],
    "Pooling Operations": [
        "avg_pool2d", "global_avg_pool2d", "max_pool2d", "max_pool2d_with_indices", "pooling"
    ],
    "Normalization Operations": [
        "batch_norm_inference", "batch_norm_training", "layer_norm", "rms_norm"
    ],
    "Activation Functions": [
        "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu", "hardsigmoid", "relu6"
    ],
    "Tensor Manipulation": [
        "broadcast", "concat", "concatenate_heads", "gather", "index", "index_select", "pad",
        "permute", "reshape", "reverse", "slice_dynamic", "slice_static", "squeeze", "transpose",
        "unsqueeze", "upsample2d"
    ],
    "Reduction Operations": [
        "argmax", "cumsum", "max", "mean", "min", "prod", "reduce_and", "reduce_or", "sum"
    ],
    "Linear Operations": [
        "linear", "matmul", "dot_general"
    ],
    "Attention Mechanisms": [
        "scaled_dot_product_attention", "scaled_dot_product_attention_decode",
        "split_query_key_value_and_split_heads"
    ],
    "Embedding Functions": [
        "embedding", "embedding_backward"
    ],
    "Memory Operations": [
        "alloc", "dealloc", "empty", "fill_cache", "paged_update_cache", "update_cache"
    ],
    "Creation Operations": [
        "arange", "constant", "full", "ones", "rand", "zeros"
    ],
    "Quantization Operations": [
        "quantize", "quantize_unrolled", "dequantize", "dequantize_unrolled",
        "requantize", "requantize_unrolled"
    ],
    "Conditional Operations": [
        "where", "clamp_scalar", "clamp_tensor"
    ],
    "Collective Operations": [
        "all_gather", "all_reduce", "all_to_all", "collective_broadcast", "collective_permute",
        "reduce_scatter", "mesh_shard"
    ],
    "Other Operations": [
        "softmax", "sort", "get_dimension_size", "to_layout", "ttnn_metal_layout_cast",
        "typecast"
    ]
}


def parse_operation_from_text(text: str) -> Optional[Operation]:
    """
    Parse operation information from the provided text format.
    This is a simplified parser - in a real scenario, you'd parse from MLIR or structured data.
    """
    # This is a placeholder - you would implement actual parsing logic here
    # For now, we'll use the structured data approach
    return None


def sanitize_filename(name: str) -> str:
    """Convert operation name to a valid filename."""
    # Remove 'ttir.' or 'forge.op.' prefix if present (for filename sanitization)
    name = name.replace("ttir.", "").replace("forge.op.", "")
    # Replace dots and special chars with underscores
    name = re.sub(r'[^\w-]', '_', name)
    # Add forge_op_ prefix to match existing convention
    return f"forge_op_{name.lower()}"


def get_category_for_op(op_name: str) -> str:
    """Get the category for an operation based on its name."""
    short_name = op_name.replace("ttir.", "").replace("forge.op.", "").lower()
    for category, ops in CATEGORIES.items():
        if short_name in ops:
            return category
    return "Other Operations"


def generate_operation_page(op: Operation, output_dir: Path) -> None:
    """Generate a markdown page for a single operation."""
    filename = sanitize_filename(op.name)
    filepath = output_dir / f"{filename}.md"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Title
        f.write(f"# {op.name}\n\n")
        
        # Overview section - combine short and detailed descriptions
        overview_text = op.description
        if op.detailed_description:
            if overview_text and not overview_text.endswith('.'):
                overview_text += "."
            overview_text += " " + op.detailed_description
        
        # Clean up TTIR references
        overview_text = overview_text.replace("TTIR", "Forge").replace("ttir", "Forge")
        
        # If description is too short or needs expansion, try to expand it
        op_name_lower = op.short_name.lower()
        # Always expand for specific operations that need better descriptions
        needs_expansion = (
            len(overview_text.strip()) < 50 or 
            "TM" in overview_text or 
            "ttir" in overview_text.lower() or
            op_name_lower == "resize2d" or
            op_name_lower == "resize1d"
        )
        if needs_expansion:
            # Try to create a more detailed overview from the operation name
            if op_name_lower == "resize2d":
                overview_text = "Resizes the spatial dimensions (height and width) of a 2D input tensor using interpolation.\n\nThe `Resize2d` operation resizes the height and width dimensions of a 4D input tensor to specified target sizes. This operation is commonly used in computer vision tasks for image resizing, upsampling, and downsampling. It supports two interpolation modes: nearest neighbor and bilinear interpolation."
            elif op_name_lower == "resize1d":
                overview_text = "Resizes the spatial dimension of a 1D input tensor using interpolation. This operation is commonly used for sequence resizing and temporal dimension manipulation."
            elif op_name_lower in ["abs"]:
                overview_text = "Computes the elementwise absolute value of the input tensor. Each output element is the absolute value of the corresponding input element."
            elif op_name_lower in ["add", "subtract", "multiply", "divide"]:
                overview_text = f"Performs elementwise {op_name_lower} operation on two input tensors. The operation is applied element-by-element, requiring both tensors to be broadcastable to the same shape."
            elif op_name_lower == "relu":
                overview_text = "Applies the Rectified Linear Unit (ReLU) activation function elementwise. ReLU sets all negative values to zero while keeping positive values unchanged, introducing non-linearity to neural networks."
            elif op_name_lower == "sigmoid":
                overview_text = "Applies the sigmoid activation function elementwise. The sigmoid function maps input values to the range (0, 1), making it useful for binary classification and probability outputs."
            elif op_name_lower == "conv2d":
                overview_text = "Performs 2D convolution operation on input tensors. This operation applies a set of learnable filters (kernels) to extract spatial features from the input, commonly used in convolutional neural networks for image processing."
            elif op_name_lower == "matmul":
                overview_text = "Performs matrix multiplication between two input tensors. This is a fundamental linear algebra operation used extensively in neural networks for linear transformations."
            elif op_name_lower == "reshape":
                overview_text = "Reshapes a tensor to new dimensions while preserving the total number of elements. The operation changes the tensor's shape without modifying its data."
            elif op_name_lower == "transpose":
                overview_text = "Transposes the dimensions of a tensor by swapping specified dimensions. Commonly used to rearrange tensor layouts for compatibility with different operations."
            elif op_name_lower == "constantpad":
                overview_text = "Applies constant padding to the input tensor. This is a low-level padding operation that directly specifies padding values for each dimension in Forge format."
        
        # Overview section
        f.write("## Overview\n\n")
        f.write(f"{overview_text}\n\n")
        
        # Function signature
        f.write("## Function Signature\n\n")
        f.write("```python\n")
        if op.signature:
            # Format signature nicely (multi-line for readability)
            sig = op.signature
            # For Resize2d and similar operations, format as multi-line
            if op.short_name.lower() in ["resize2d", "resize1d", "conv2d", "matmul"] and '(' in sig:
                # Extract function name and parameters
                if '(' in sig and ')' in sig:
                    func_name = sig.split('(')[0]
                    rest = sig[len(func_name):]
                    # Try to format parameters nicely
                    if rest.startswith('(') and ')' in rest:
                        # Find the matching closing paren
                        paren_count = 0
                        split_idx = -1
                        for i, char in enumerate(rest):
                            if char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                                if paren_count == 0:
                                    split_idx = i
                                    break
                        
                        if split_idx > 0:
                            params_str = rest[1:split_idx]  # Remove outer parentheses
                            return_part = rest[split_idx+1:]
                            
                            # Split by comma, but respect nested types
                            params = []
                            current = ""
                            depth = 0
                            for char in params_str:
                                current += char
                                if char in '[(':
                                    depth += 1
                                elif char in '])':
                                    depth -= 1
                                elif char == ',' and depth == 0:
                                    params.append(current[:-1].strip())
                                    current = ""
                            if current.strip():
                                params.append(current.strip())
                            
                            # Write formatted signature
                            f.write(f"{func_name}(\n")
                            for i, param in enumerate(params):
                                if i < len(params) - 1:
                                    f.write(f"    {param},\n")
                                else:
                                    f.write(f"    {param}\n")
                            f.write(f"){return_part}\n")
                        else:
                            f.write(f"{sig}\n")
                    else:
                        f.write(f"{sig}\n")
                else:
                    f.write(f"{sig}\n")
            else:
                f.write(f"{sig}\n")
        else:
            # Fallback: construct signature from operands and attributes
            f.write(f"{op.name}(")
            params = []
            
            # Add operands first (required parameters)
            for operand in op.operands:
                if operand.name != "output":  # output is typically the last parameter
                    params.append(operand.name)
            
            # Add attributes (optional parameters)
            for attr in op.attributes:
                param_str = f"{attr.name}"
                if attr.default:
                    param_str += f"={attr.default}"
                params.append(param_str)
            
            # Add output operand last if present
            for operand in op.operands:
                if operand.name == "output":
                    params.append(operand.name)
            
            f.write(", ".join(params) if params else "")
            f.write(")\n")
        f.write("```\n\n")
        
        # Parameters section with improved formatting
        if op.operands or op.attributes:
            f.write("## Parameters\n\n")
            
            # Separate 'name' parameter to write it first
            name_attr = None
            other_attrs = []
            if op.attributes:
                for attr in op.attributes:
                    if attr.name == "name":
                        name_attr = attr
                    else:
                        other_attrs.append(attr)
            
            # Write name parameter first if it exists
            if name_attr:
                type_str = name_attr.mlir_type if name_attr.mlir_type else "str"
                desc = name_attr.description
                if not desc or desc.lower() in ["name", "op name", "op name, unique to the module, or leave blank to autoset"]:
                    desc = "Name identifier for this operation in the computation graph."
                f.write(f"- **{name_attr.name}** (`{type_str}`): {desc}\n")
                if op.operands or other_attrs:
                    f.write("\n")
            
            # Process operands (tensor inputs)
            if op.operands:
                for operand in op.operands:
                    if operand.name != "output":  # Output is shown separately
                        type_str = operand.type if operand.type else "Tensor"
                        desc = operand.description
                        
                        # Enhance description for common parameters
                        if operand.name == "operandA":
                            # Special handling for Resize2d
                            if op.short_name.lower() == "resize2d":
                                desc = f"Input tensor of shape `(N, C, H, W)` (channel-first) or `(N, H, W, C)` (channel-last) where:\n  - `N` is the batch size\n  - `C` is the number of channels\n  - `H` is the input height\n  - `W` is the input width"
                            elif op.short_name.lower() == "resize1d":
                                desc = f"Input tensor of shape `(N, C, L)` (channel-first) or `(N, L, C)` (channel-last) where:\n  - `N` is the batch size\n  - `C` is the number of channels\n  - `L` is the input length"
                            elif not desc or desc.lower() in ["input operand a", "input operand"]:
                                desc = f"Input tensor. Shape and data type depend on the specific operation requirements."
                        elif operand.name in ["operandB", "operand_b"] and not desc or desc.lower() in ["input operand b", "second operand"]:
                            desc = f"Second input tensor. Must be broadcastable with operandA."
                        
                        f.write(f"- **{operand.name}** (`{type_str}`): {desc}\n")
                if other_attrs:  # Only add blank line if there are attributes too
                    f.write("\n")
            
            # Process other attributes
            if other_attrs:
                for attr in other_attrs:
                    default_str = f", default: `{attr.default}`" if attr.default else ""
                    type_str = attr.mlir_type if attr.mlir_type else "Any"
                    desc = attr.description
                    
                    # Enhance descriptions for common attributes
                    if attr.name == "mode" and "interpolation" in desc.lower():
                        desc = f"Interpolation mode. Supported values:\n  - `'nearest'`: Nearest neighbor interpolation (fast, but may produce aliasing)\n  - `'bilinear'`: Bilinear interpolation (smoother results, better for upsampling)"
                    elif attr.name == "align_corners" and not desc or "align" in desc.lower():
                        desc = "If `True`, the corner pixels of the input and output tensors are aligned. This parameter only affects bilinear interpolation mode. When `False`, the input and output tensors are aligned by their corner points of the corner pixels, and the sampling points are computed based on the pixel centers."
                    elif attr.name == "channel_last" and not desc or "channel" in desc.lower():
                        desc = "If `True`, the input tensor is in channel-last format `(N, H, W, C)`. If `False`, the input tensor is in channel-first format `(N, C, H, W)`."
                    elif attr.name == "sizes" and op.short_name.lower() in ["resize2d", "resize1d"]:
                        if "resize2d" in op.short_name.lower():
                            desc = "Target output spatial dimensions as `[height, width]` or `(height, width)`. The output tensor will have these exact height and width values."
                        else:
                            desc = "Target output spatial dimension. The output tensor will have this exact size."
                    
                    f.write(f"- **{attr.name}** (`{type_str}`{default_str}): {desc}\n")
                f.write("\n")
        
        # Returns section with improved formatting
        if op.results:
            f.write("## Returns\n\n")
            for result in op.results:
                type_str = result.type if result.type else "Tensor"
                # Clean up return description
                desc = result.description
                if desc:
                    # Remove patterns like "Output tensor: Tensor" or "Forge tensor"
                    if ":" in desc:
                        parts = desc.split(":", 1)
                        if parts[0].strip().lower() in ["output tensor", "tensor", "forge tensor"]:
                            desc = parts[1].strip() if len(parts) > 1 else ""
                    # Remove standalone type words at the start
                    desc_words = desc.split()
                    if desc_words and desc_words[0] in ["Tensor", "Output", "Forge"]:
                        if len(desc_words) > 1:
                            desc = ' '.join(desc_words[1:])
                
                # Enhance return description based on operation type
                if not desc or desc.lower() in ["output tensor", "tensor", "result"]:
                    op_name_lower = op.short_name.lower()
                    if op_name_lower == "resize2d":
                        desc = "Output tensor with resized spatial dimensions. The output shape is `(N, C, H_out, W_out)` if `channel_last=False` or `(N, H_out, W_out, C)` if `channel_last=True`, where `H_out` and `W_out` are the values specified in the `sizes` parameter. The batch size `N` and number of channels `C` remain unchanged."
                    elif op_name_lower == "resize1d":
                        desc = "Output tensor with resized spatial dimension. The output shape preserves the batch and channel dimensions while modifying the spatial dimension according to the `sizes` parameter."
                    elif op_name_lower in ["abs", "relu", "sigmoid", "tanh"]:
                        desc = f"Output tensor with the same shape as the input. Each element is the result of applying the {op_name_lower} function to the corresponding input element."
                    elif op_name_lower in ["add", "subtract", "multiply", "divide"]:
                        desc = "Output tensor with the same shape as the broadcasted input tensors. Each element is the result of the elementwise operation."
                    elif op_name_lower == "conv2d":
                        desc = "Output tensor containing the convolution result. The output shape depends on the input shape, kernel size, padding, and stride parameters."
                    elif op_name_lower == "matmul":
                        desc = "Output tensor containing the matrix multiplication result. The output shape is determined by the input tensor shapes following matrix multiplication rules."
                    elif op_name_lower == "reshape":
                        desc = "Output tensor with the new shape. The total number of elements remains the same as the input."
                    else:
                        desc = desc or "Output tensor with shape and type determined by the operation."
                
                # Clean up type_str - remove "Output tensor" prefix if present
                if type_str.lower() == "output tensor":
                    type_str = "Tensor"
                f.write(f"- **{result.name}** (`{type_str}`): {desc}\n")
            f.write("\n")
        
        # Mathematical definition
        if op.mathematical_definition:
            f.write("## Mathematical Definition\n\n")
            f.write(f"{op.mathematical_definition}\n\n")
        elif op.short_name.lower() == "resize2d":
            # Add mathematical definition for Resize2d
            f.write("## Mathematical Definition\n\n")
            f.write("### Nearest Neighbor Interpolation\n\n")
            f.write("For nearest neighbor interpolation, each output pixel value is taken from the nearest input pixel:\n\n")
            f.write("```\n")
            f.write("output[i, j] = input[round(i * H_in / H_out), round(j * W_in / W_out)]\n")
            f.write("```\n\n")
            f.write("### Bilinear Interpolation\n\n")
            f.write("For bilinear interpolation, each output pixel is computed as a weighted average of the four nearest input pixels:\n\n")
            f.write("```\n")
            f.write("output[i, j] = Σ(weight_k * input[k]) for k in {top-left, top-right, bottom-left, bottom-right}\n")
            f.write("```\n\n")
            f.write("The weights are computed based on the distance from the output pixel to the surrounding input pixels.\n\n")
        
        # Examples
        if op.examples:
            f.write("## Examples\n\n")
            for example in op.examples:
                f.write(f"```python\n{example}\n```\n\n")
        
        # Notes
        if op.notes:
            f.write("## Notes\n\n")
            for note in op.notes:
                f.write(f"{note}\n\n")
        
        # Related Operations - add actual links for known related operations
        f.write("## Related Operations\n\n")
        related_ops = []
        op_name_lower = op.short_name.lower()
        
        # Define related operations mapping
        if op_name_lower == "resize2d":
            related_ops = [
                ("forge.op.Resize1d", "forge_op_resize1d.md", "Resize 1D tensors (e.g., sequences)"),
                ("forge.op.Upsample2d", "forge_op_upsample2d.md", "Upsample using scale factors instead of target sizes"),
                ("forge.op.Downsample2d", "forge_op_downsample2d.md", "Downsample operation"),
                ("forge.op.Transpose", "forge_op_transpose.md", "Rearrange tensor dimensions")
            ]
        elif op_name_lower == "resize1d":
            related_ops = [
                ("forge.op.Resize2d", "forge_op_resize2d.md", "Resize 2D tensors"),
                ("forge.op.Upsample2d", "forge_op_upsample2d.md", "Upsample operation"),
            ]
        elif op_name_lower in ["abs", "relu", "sigmoid", "tanh", "gelu"]:
            # Activation functions
            activation_ops = ["relu", "sigmoid", "tanh", "gelu", "leakyrelu"]
            for act in activation_ops:
                if act != op_name_lower:
                    related_ops.append((f"forge.op.{act.capitalize()}", f"forge_op_{act}.md", f"{act.capitalize()} activation function"))
        elif op_name_lower in ["add", "subtract", "multiply", "divide"]:
            binary_ops = ["add", "subtract", "multiply", "divide", "max", "min"]
            for bin_op in binary_ops:
                if bin_op != op_name_lower:
                    related_ops.append((f"forge.op.{bin_op.capitalize()}", f"forge_op_{bin_op}.md", f"Elementwise {bin_op} operation"))
        elif op_name_lower in ["reshape", "transpose", "squeeze", "unsqueeze"]:
            tm_ops = ["reshape", "transpose", "squeeze", "unsqueeze", "select", "index"]
            for tm_op in tm_ops:
                if tm_op != op_name_lower:
                    related_ops.append((f"forge.op.{tm_op.capitalize()}", f"forge_op_{tm_op}.md", f"{tm_op.capitalize()} tensor manipulation operation"))
        
        if related_ops:
            for related_name, related_file, related_desc in related_ops:
                f.write(f"- [{related_name}](./{related_file}): {related_desc}\n")
            f.write("\n")
        else:
            f.write("*Related operations will be automatically linked here in future updates.*\n\n")
        
        # Traits and Interfaces (if needed)
        if op.traits or op.interfaces:
            f.write("## Implementation Details\n\n")
            if op.traits:
                f.write("**Traits:** " + ", ".join(op.traits) + "\n\n")
            if op.interfaces:
                f.write("**Interfaces:** " + ", ".join(op.interfaces) + "\n\n")


def generate_index_page(operations: List[Operation], output_dir: Path) -> None:
    """Generate the main index page with operations grouped by categories."""
    filepath = output_dir / "operations.md"
    
    # Group operations by category
    ops_by_category: Dict[str, List[Operation]] = {}
    for op in operations:
        # Use the category from the operation itself (from discovery)
        category = op.category
        if category not in ops_by_category:
            ops_by_category[category] = []
        ops_by_category[category].append(op)
    
    # Define category order matching manager's requirements
    category_order = [
        "Elementwise Operations",
        "Convolution Operations",
        "Pooling Operations",
        "Normalization Operations",
        "Tensor Manipulation",
        "Reduction Operations",
        "Linear Operations",
        "Activation Functions",
        "Memory Operations",
        "Other Operations"
    ]
    
    # Sort categories according to order, then alphabetically for any extras
    sorted_categories = []
    for cat in category_order:
        if cat in ops_by_category:
            sorted_categories.append(cat)
    # Add any remaining categories not in the predefined order
    for cat in sorted(ops_by_category.keys()):
        if cat not in sorted_categories:
            sorted_categories.append(cat)
    
    def slugify(text: str) -> str:
        """Convert text to anchor link format."""
        return text.lower().replace(" ", "-")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Forge Operations Reference\n\n")
        f.write("Welcome to the Forge Operations Reference. This page provides a comprehensive guide to all supported operations in the Forge framework.\n\n")
        
        # Overview section
        f.write("## Overview\n\n")
        f.write("Forge operations are organized into logical categories based on their functionality. Each operation is documented with detailed information including function signatures, parameters, examples, and usage notes.\n\n")
        
        # Quick Navigation
        f.write("## Quick Navigation\n\n")
        for category in sorted_categories:
            anchor = slugify(category)
            # Create a short description for each category
            desc_map = {
                "Elementwise Operations": "Mathematical operations applied element-wise",
                "Convolution Operations": "Convolution and related transformations",
                "Pooling Operations": "Pooling and downsampling operations",
                "Normalization Operations": "Batch and layer normalization",
                "Tensor Manipulation": "Reshaping, slicing, and tensor operations",
                "Reduction Operations": "Aggregation and reduction operations",
                "Linear Operations": "Matrix multiplication and linear transformations",
                "Activation Functions": "Non-linear activation functions",
                "Memory Operations": "Cache and memory management operations",
                "Other Operations": "Miscellaneous operations"
            }
            desc = desc_map.get(category, "Operations in this category")
            f.write(f"- [{category}](#{anchor}) - {desc}\n")
        f.write("\n---\n\n")
        
        # Category sections with tables
        for category in sorted_categories:
            anchor = slugify(category)
            f.write(f"## {category}\n\n")
            
            # Add category description
            if category == "Elementwise Operations":
                f.write("Elementwise operations apply a function to each element of the input tensor(s) independently.\n\n")
            elif category == "Convolution Operations":
                f.write("Convolution operations perform spatial filtering and feature extraction.\n\n")
            elif category == "Pooling Operations":
                f.write("Pooling operations reduce spatial dimensions and provide translation invariance.\n\n")
            elif category == "Normalization Operations":
                f.write("Normalization operations stabilize training and improve convergence.\n\n")
            elif category == "Tensor Manipulation":
                f.write("Operations for reshaping, slicing, and manipulating tensor structure.\n\n")
            elif category == "Reduction Operations":
                f.write("Operations that reduce tensor dimensions by aggregation.\n\n")
            elif category == "Linear Operations":
                f.write("Matrix multiplication and linear transformations.\n\n")
            elif category == "Activation Functions":
                f.write("Non-linear activation functions for neural networks.\n\n")
            elif category == "Memory Operations":
                f.write("Operations for cache and memory management.\n\n")
            elif category == "Other Operations":
                f.write("Miscellaneous operations.\n\n")
            
            # Create table
            ops = sorted(ops_by_category[category], key=lambda x: x.short_name)
            if ops:
                f.write("| Operation | Description | Link |\n")
                f.write("|-----------|-------------|------|\n")
                for op in ops:
                    filename = sanitize_filename(op.name)
                    # Extract just the operation name (e.g., "Abs" from "forge.op.Abs")
                    op_display_name = op.name.split(".")[-1] if "." in op.name else op.name
                    f.write(f"| **{op_display_name}** | {op.description} | [{op.name}](./operations/{filename}.md) |\n")
                f.write("\n")
        
        # Documentation Structure section
        f.write("---\n\n")
        f.write("## Documentation Structure\n\n")
        f.write("Each operation documentation page includes:\n\n")
        f.write("- **Overview**: Brief description of what the operation does\n\n")
        f.write("- **Function Signature**: Python API signature with type hints\n\n")
        f.write("- **Parameters**: Detailed parameter descriptions with types and defaults\n\n")
        f.write("- **Returns**: Return value description\n\n")
        f.write("- **Mathematical Definition**: Mathematical formula (where applicable)\n\n")
        f.write("- **Examples**: Code examples demonstrating usage\n\n")
        f.write("- **Notes**: Important implementation details and warnings\n\n")
        f.write("- **Related Operations**: Links to related operations\n\n")
        f.write("\n---\n\n")
        f.write("*This documentation is automatically generated from operation definitions. For the most up-to-date information, refer to the source code.*\n")


def convert_discovered_to_operation(discovered) -> Operation:
    """Convert a DiscoveredOperation to an Operation."""
    # Extract short description from docstring (first line before Parameters/Returns)
    docstring = discovered.docstring.strip()
    
    # Find the first meaningful line (skip empty lines, get text before Parameters/Returns)
    lines = docstring.split('\n')
    short_desc = discovered.name  # Default
    detailed_desc = ""
    
    # Collect all description lines before Parameters/Returns
    desc_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Stop at Parameters or Returns section
        if stripped.startswith("Parameters") or stripped.startswith("Returns"):
            break
        if stripped and not stripped.startswith("---"):
            desc_lines.append(stripped)
    
    # Use first line as short description, rest as detailed
    if desc_lines:
        short_desc = desc_lines[0]
        if len(desc_lines) > 1:
            detailed_desc = "\n\n".join(desc_lines[1:])
    
    # Clean up descriptions
    short_desc = short_desc.strip()
    detailed_desc = detailed_desc.strip()
    
    # Clean up TTIR references
    short_desc = short_desc.replace("TTIR", "Forge").replace("ttir", "Forge")
    detailed_desc = detailed_desc.replace("TTIR", "Forge").replace("ttir", "Forge")
    
    # If description is too short or seems wrong, try to infer from function name
    if len(short_desc) < 3 or short_desc.lower() in ["sigmoid", "tm", "op"]:
        # Try to create a better description from function name
        op_name_lower = discovered.name.lower()
        if op_name_lower == "abs":
            short_desc = "Elementwise absolute value operation"
        elif op_name_lower in ["add", "subtract", "multiply", "divide"]:
            short_desc = f"Elementwise {op_name_lower} of two tensors"
        elif op_name_lower == "relu":
            short_desc = "Rectified Linear Unit (ReLU) activation"
        elif op_name_lower == "sigmoid":
            short_desc = "Sigmoid activation function"
        elif op_name_lower == "conv2d":
            short_desc = "2D convolution operation"
        elif op_name_lower == "matmul":
            short_desc = "Matrix multiplication operation"
        elif op_name_lower == "constantpad":
            short_desc = "Constant padding operation"
            detailed_desc = "Applies constant padding to the input tensor. This is a low-level padding operation that directly specifies padding values for each dimension."
        elif "tm" in short_desc.lower() or short_desc.lower() == "tm":
            # Generic tensor manipulation operations
            if op_name_lower in ["reshape", "transpose", "squeeze", "unsqueeze", "select", "index", "broadcast", "pad"]:
                short_desc = f"{op_name_lower.capitalize()} tensor manipulation operation"
        # Keep the original if we couldn't improve it
    
    # Convert parameters to operands and attributes
    operands = []
    attributes = []
    
    for param in discovered.parameters:
        # Keep 'name' parameter as an attribute (don't skip it)
        
        # Determine if it's an operand (Tensor) or attribute
        param_type = param.get("type", "").lower()
        is_tensor = "tensor" in param_type or "parameter" in param_type
        
        # Clean up description (remove extra whitespace, newlines, and redundant type prefixes)
        desc = param.get("description", "").strip()
        desc = ' '.join(desc.split())  # Normalize whitespace
        
        # Remove redundant type prefixes from description (e.g., "Tensor First operand" -> "First operand")
        type_words = ["Tensor", "Parameter", "Optional", "Union", "List", "Tuple", "int", "float", "bool", "str", "Tenor"]
        desc_words = desc.split()
        if desc_words and desc_words[0] in type_words:
            # Check if first word is a type (possibly followed by comma or colon)
            first_word = desc_words[0].rstrip(',:')
            if first_word in type_words and len(desc_words) > 1:
                # Remove the type word
                desc = ' '.join(desc_words[1:])
        
        # Remove redundant phrases like "optional Optional" or "Tensor Tensor"
        desc = desc.replace("optional Optional", "Optional").replace("Tensor Tensor", "Tensor")
        desc = desc.replace("Tenor, optional", "").replace("Tenor, Optional", "Optional").replace("Tenor,", "").strip()
        # Clean up TTIR references
        desc = desc.replace("TTIR format", "Forge format").replace("ttir format", "Forge format")
        desc = desc.replace("TTIR", "Forge").replace("ttir", "Forge")
        if desc.startswith(", "):
            desc = desc[2:]
        desc = desc.strip()
        
        # Enhance specific parameter descriptions
        if param["name"] == "padding" and "dim0_low" in desc.lower():
            desc = "Padding values in Forge format: `[dim0_low, dim0_high, dim1_low, dim1_high, ...]`. Length must be 2 * rank of input tensor. Each dimension has a low and high padding value."
        
        if is_tensor:
            operands.append(Operand(
                name=param["name"],
                type=param.get("type", "Tensor"),
                description=desc or f"{param['name']} tensor"
            ))
        else:
            attributes.append(Attribute(
                name=param["name"],
                mlir_type=param.get("type", "Any"),
                description=desc or f"{param['name']} parameter",
                default=param.get("default")
            ))
    
    # Add output operand
    if discovered.return_type:
        operands.append(Operand(
            name="output",
            type=discovered.return_type,
            description=discovered.return_description or "Output tensor"
        ))
    
    # Convert name to forge.op format
    short_name = discovered.name.lower()
    full_name = f"forge.op.{discovered.name}"
    
    return Operation(
        name=full_name,
        short_name=short_name,
        category=discovered.category,
        description=short_desc,
        detailed_description=detailed_desc,
        operands=operands,
        results=[Operand("result", discovered.return_type or "Tensor", "Output tensor")],
        attributes=attributes,
        examples=[],  # Could be extracted from docstring examples if present
        signature=discovered.signature  # Store the actual Python signature
    )


def load_operations_from_codebase(project_root: Path) -> List[Operation]:
    """
    Automatically discover operations from forge/forge/op/*.py files.
    This is the primary method - operations are discovered automatically.
    """
    try:
        import sys
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from discover_operations import discover_operations, DiscoveredOperation
        
        print("Discovering operations from codebase...")
        discovered_ops = discover_operations(project_root)
        
        if not discovered_ops:
            print("Warning: No operations discovered. Falling back to manual data.")
            return load_operations_from_data(project_root)
        
        print(f"Discovered {len(discovered_ops)} operations from codebase")
        
        # Convert to Operation format
        operations = []
        for discovered in discovered_ops:
            try:
                op = convert_discovered_to_operation(discovered)
                operations.append(op)
            except Exception as e:
                print(f"Warning: Failed to convert {discovered.name}: {e}")
                continue
        
        return operations
        
    except ImportError as e:
        print(f"Warning: Could not import operation discoverer: {e}")
        print("Falling back to manual operation data.")
        return load_operations_from_data(project_root)


def load_operations_from_data(project_root: Path) -> List[Operation]:
    """
    Load operations from a structured data source (fallback method).
    This imports from ops_data.py which contains manually curated operation definitions.
    """
    import sys
    import importlib.util
    from pathlib import Path
    
    # Get the ops_data.py file path
    script_dir = Path(__file__).parent
    ops_data_path = script_dir / "ops_data.py"
    
    try:
        # Load the module directly from file
        spec = importlib.util.spec_from_file_location("ops_data", ops_data_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load ops_data module")
        ops_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops_data)
        return ops_data.get_all_operations()
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"Warning: Could not import ops_data: {e}")
        print("Using minimal example operations.")
        # Fallback to minimal examples
        return [
            Operation(
                name="forge.op.Abs",
                short_name="abs",
                category="Elementwise Operations",
                description="Elementwise absolute value operation.",
                operands=[
                    Operand("operandA", "Tensor", "The input tensor"),
                    Operand("output", "Tensor", "The output tensor")
                ],
                results=[Operand("result", "Tensor", "The result tensor")]
            )
        ]


def main():
    """Main function to generate all documentation."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs" / "src"
    ops_docs_dir = docs_dir / "operations"
    
    # Create operations directory if it doesn't exist
    ops_docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load operations - try automatic discovery first, fall back to manual data
    print("Loading operations...")
    operations = load_operations_from_codebase(project_root)
    print(f"Loaded {len(operations)} operations")
    
    # Generate individual operation pages
    print("Generating operation pages...")
    for op in operations:
        generate_operation_page(op, ops_docs_dir)
        print(f"  Generated: {op.name}")
    
    # Generate index page
    print("Generating index page...")
    generate_index_page(operations, docs_dir)
    print("  Generated: operations.md")
    
    print("\nDocumentation generation complete!")
    print(f"Output directory: {ops_docs_dir}")


if __name__ == "__main__":
    main()

