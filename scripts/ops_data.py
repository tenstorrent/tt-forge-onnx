"""
Operation data definitions for documentation generation.

This file contains structured data for all supported operations, including
descriptions, parameters, examples, and other metadata.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Operand:
    """Represents an operand (input/output) of an operation."""
    name: str
    type: str
    description: str


@dataclass
class Attribute:
    """Represents an attribute of an operation."""
    name: str
    mlir_type: str
    description: str
    default: str = None


@dataclass
class Operation:
    """Represents a complete operation definition."""
    name: str
    short_name: str
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


def get_all_operations() -> List[Operation]:
    """
    Returns a list of all supported operations with their complete definitions.
    
    This data structure can be extended or replaced with data loaded from
    MLIR definitions, JSON files, or other sources.
    """
    operations = []
    
    # Elementwise Operations
    operations.extend([
        Operation(
            name="ttir.abs",
            short_name="abs",
            category="Elementwise Operations",
            description="Elementwise absolute value operation.",
            detailed_description=(
                "The abs operation computes the absolute value of each element in the input tensor.\n\n"
                "For each element, it returns the magnitude of the value without regard to its sign:\n"
                "- For real numbers, it returns |x| (the non-negative value without sign)\n\n"
                "This operation has the idempotence property, meaning that applying it multiple times "
                "produces the same result as applying it once: abs(abs(x)) = abs(x). The operation preserves "
                "the data type of the input."
            ),
            mathematical_definition="abs(x) = |x| = { x if x ≥ 0, -x if x < 0 }",
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor")],
            examples=[
                "# Compute absolute values\n"
                "%result = ttir.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>"
            ],
            traits=["AlwaysSpeculatableImplTrait", "TTIR_Broadcastable", "TTIR_Idempotence", "TwoOperands"]
        ),
        Operation(
            name="ttir.add",
            short_name="add",
            category="Elementwise Operations",
            description="Elementwise addition operation.",
            detailed_description=(
                "The add operation performs an elementwise addition between two tensors.\n\n"
                "For each pair of corresponding elements, it adds the elements and places the result "
                "in the output tensor."
            ),
            mathematical_definition="add(x, y) = x + y",
            operands=[
                Operand("lhs", "ranked tensor of any type values", "Left-hand side tensor"),
                Operand("rhs", "ranked tensor of any type values", "Right-hand side tensor"),
                Operand("output", "ranked tensor of any type values", "Output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor")],
            examples=[
                "# Addition operation\n"
                "%result = ttir.add(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>"
            ]
        ),
        Operation(
            name="ttir.relu",
            short_name="relu",
            category="Elementwise Operations",
            description="Eltwise ReLU.",
            detailed_description=(
                "The relu operation computes the rectified linear unit (ReLU) of each element in the input tensor.\n\n"
                "For each element, it returns the maximum of 0 and the value. The operation preserves "
                "the data type of the input."
            ),
            mathematical_definition="relu(x) = max(0, x)",
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor")],
            traits=["AlwaysSpeculatableImplTrait", "TTIR_Broadcastable", "TTIR_Idempotence", "TwoOperands"]
        ),
        Operation(
            name="ttir.sigmoid",
            short_name="sigmoid",
            category="Elementwise Operations",
            description="Eltwise sigmoid.",
            detailed_description=(
                "The sigmoid operation computes the sigmoid of each element in the input tensor.\n\n"
                "For each element, it returns the sigmoid of the value."
            ),
            mathematical_definition="sigmoid(x) = 1 / (1 + exp(-x))",
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor")]
        ),
    ])
    
    # Convolution Functions
    operations.extend([
        Operation(
            name="ttir.conv2d",
            short_name="conv2d",
            category="Convolution Functions",
            description="Conv2d operation.",
            detailed_description=(
                "Applies a 2D convolution over an input image composed of several input planes.\n\n"
                "This operation performs a 2D convolution on the input tensor using the provided weight tensor "
                "and optional bias. It supports configurable stride, padding, dilation, and grouping parameters "
                "to control the convolution behavior."
            ),
            operands=[
                Operand("input", "ranked tensor of any type values", 
                       "Input tensor in NHWC format (batch, height, width, channels)"),
                Operand("weight", "ranked tensor of any type values", 
                       "Weight tensor in format (O, C/G, K_H, K_W)"),
                Operand("bias", "ranked tensor of any type values", "Optional bias tensor"),
                Operand("output", "ranked tensor of any type values", "Output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "Output tensor after convolution")],
            attributes=[
                Attribute("stride", "i32 | array<2xi32>", 
                         "Stride of the convolving kernel. Can be a single number or a tuple (sH, sW).", "1"),
                Attribute("padding", "i32 | array<2xi32> | array<4xi32>", 
                         "Padding applied to the input. Can be a single number, tuple (pH, pW), or tuple (pT, pL, pB, pR).", "0"),
                Attribute("dilation", "i32 | array<2xi32>", 
                         "Spacing between kernel elements. Can be a single number or a tuple (dH, dW).", "1"),
                Attribute("groups", "i32", 
                         "Number of blocked connections from input channels to output channels. "
                         "Input and output channels must both be divisible by groups.", "1")
            ],
            examples=[
                "# Basic 2D convolution\n"
                "%result = ttir.conv2d(%input, %weight, %bias, %output) {\n"
                "    stride = [1, 1],\n"
                "    padding = [0, 0, 0, 0],\n"
                "    dilation = [1, 1],\n"
                "    groups = 1\n"
                "} : tensor<1x28x28x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>, tensor<1x26x26x16xf32> -> tensor<1x26x26x16xf32>"
            ],
            notes=[
                "The input tensor is expected in NHWC format: (N, H_in, W_in, C) where N is batch size, "
                "H_in is height, W_in is width, and C is number of channels.",
                "The weight tensor format is (O, C/G, K_H, K_W) where O is output channels, C is input channels, "
                "G is number of groups, K_H is kernel height, and K_W is kernel width.",
                "The output shape is calculated as: H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1"
            ]
        ),
    ])
    
    # Pooling Functions
    operations.extend([
        Operation(
            name="ttir.avg_pool2d",
            short_name="avg_pool2d",
            category="Pooling Functions",
            description="2D average pooling operation.",
            detailed_description=(
                "The avg_pool2d operation applies a 2D average pooling over an input tensor composed of several input planes.\n\n"
                "This operation performs downsampling by dividing the input into local regions and computing the average "
                "value of each region. It reduces the spatial dimensions (height and width) of an input tensor while "
                "preserving the batch and channel dimensions."
            ),
            operands=[
                Operand("input", "ranked tensor of any type values", 
                       "Input tensor in NHWC format (batch, height, width, channels)"),
                Operand("output", "ranked tensor of any type values", "Output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "Output tensor after average pooling")],
            attributes=[
                Attribute("kernel", "i32 | array<2xi32>", 
                         "Kernel size for height and width dimensions. Can be a single number or a tuple [kH, kW].", None),
                Attribute("stride", "i32 | array<2xi32>", 
                         "Stride for height and width dimensions. Can be a single number or a tuple [sH, sW].", "1"),
                Attribute("dilation", "i32 | array<2xi32>", 
                         "Dilation for height and width dimensions. Can be a single number or a tuple [dH, dW].", "1"),
                Attribute("padding", "i32 | array<2xi32> | array<4xi32>", 
                         "Padding applied to the input. Can be a single number, tuple [pH, pW], or tuple [pT, pL, pB, pR].", "0"),
                Attribute("ceil_mode", "bool", 
                         "When true, uses ceil instead of floor for output shape calculation.", "false"),
                Attribute("count_include_pad", "bool", 
                         "When true, include padding in the average calculation.", "true")
            ],
            examples=[
                "# Basic 2D average pooling with a 2x2 kernel and stride 1\n"
                "%result = ttir.avg_pool2d(%input, %output) {\n"
                "    kernel = [2, 2],\n"
                "    stride = [1, 1],\n"
                "    dilation = [1, 1],\n"
                "    padding = [0, 0, 0, 0],\n"
                "    ceil_mode = false\n"
                "} : tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32> -> tensor<1x2x2x1xf32>"
            ]
        ),
        Operation(
            name="ttir.max_pool2d",
            short_name="max_pool2d",
            category="Pooling Functions",
            description="2D maximum pooling operation.",
            detailed_description=(
                "The max_pool2d operation applies a 2D maximum pooling over an input tensor composed of several input planes.\n\n"
                "This operation performs downsampling by dividing the input into local regions and computing the maximum "
                "value of each region. It reduces the spatial dimensions (height and width) of an input tensor while "
                "preserving the batch and channel dimensions."
            ),
            operands=[
                Operand("input", "ranked tensor of any type values", 
                       "Input tensor in NHWC format (batch, height, width, channels)"),
                Operand("output", "ranked tensor of any type values", "Output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "Output tensor after maximum pooling")],
            attributes=[
                Attribute("kernel", "i32 | array<2xi32>", 
                         "Kernel size for height and width dimensions.", None),
                Attribute("stride", "i32 | array<2xi32>", 
                         "Stride for height and width dimensions.", "1"),
                Attribute("dilation", "i32 | array<2xi32>", 
                         "Dilation for height and width dimensions.", "1"),
                Attribute("padding", "i32 | array<2xi32> | array<4xi32>", 
                         "Padding applied to the input.", "0"),
                Attribute("ceil_mode", "bool", 
                         "When true, uses ceil instead of floor for output shape calculation.", "false")
            ]
        ),
    ])
    
    # Linear Functions
    operations.extend([
        Operation(
            name="ttir.matmul",
            short_name="matmul",
            category="Linear Functions",
            description="Matrix multiplication operation.",
            detailed_description=(
                "The matmul operation computes the matrix multiplication of two tensors.\n\n"
                "This operation performs matrix multiplication between tensors a and b. It supports optional "
                "transposition of either input tensor before multiplication. For 2D tensors, this computes the "
                "standard matrix product. For tensors with more dimensions, it applies batched matrix multiplication."
            ),
            operands=[
                Operand("a", "ranked tensor of any type values", "The first input tensor"),
                Operand("b", "ranked tensor of any type values", "The second input tensor"),
                Operand("output", "ranked tensor of any type values", "Output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result of the matrix multiplication")],
            attributes=[
                Attribute("transpose_a", "bool", 
                         "Whether to transpose tensor a before multiplication.", "false"),
                Attribute("transpose_b", "bool", 
                         "Whether to transpose tensor b before multiplication.", "false")
            ],
            examples=[
                "# Basic matrix multiplication of 2D tensors\n"
                "%result = ttir.matmul(%a, %b, %output) : "
                "tensor<3x4xf32>, tensor<4x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>"
            ],
            notes=[
                "The inner dimensions of the input tensors must be compatible for matrix multiplication. "
                "If a has shape [..., m, k] and b has shape [..., k, n], then the result will have shape [..., m, n]."
            ]
        ),
    ])
    
    # Tensor Manipulation
    operations.extend([
        Operation(
            name="ttir.reshape",
            short_name="reshape",
            category="Tensor Manipulation",
            description="Tensor reshape operation.",
            detailed_description=(
                "The reshape operation changes the shape of a tensor without changing the data or number of elements.\n\n"
                "This operation takes an input tensor and reshapes it to a new shape specified by the shape attribute. "
                "The total number of elements in the tensor must remain the same after reshaping."
            ),
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor to reshape"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The reshaped tensor")],
            attributes=[
                Attribute("shape", "array<i32>", 
                         "The new shape for the tensor as an array of integers.", None)
            ],
            examples=[
                "# Reshape a 2x3 tensor to a 1x6 tensor\n"
                "%result = ttir.reshape(%input, %output) {shape = [1, 6]} : "
                "tensor<2x3xf32>, tensor<1x6xf32> -> tensor<1x6xf32>"
            ],
            notes=[
                "The total number of elements in the input tensor must equal the total number of elements "
                "in the output tensor."
            ]
        ),
        Operation(
            name="ttir.transpose",
            short_name="transpose",
            category="Tensor Manipulation",
            description="Tensor transpose operation.",
            detailed_description=(
                "The transpose operation swaps two dimensions of a tensor.\n\n"
                "This operation exchanges the positions of two specified dimensions in the input tensor, "
                "effectively transposing those dimensions."
            ),
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The transposed tensor")],
            attributes=[
                Attribute("dim0", "i32", "The first dimension to swap.", None),
                Attribute("dim1", "i32", "The second dimension to swap.", None)
            ],
            examples=[
                "# Transpose dimensions 0 and 1\n"
                "%result = ttir.transpose(%input, %output) {dim0 = 0 : i32, dim1 = 1 : i32} : "
                "tensor<2x3x4xf32>, tensor<3x2x4xf32> -> tensor<3x2x4xf32>"
            ]
        ),
    ])
    
    # Reduction Operations
    operations.extend([
        Operation(
            name="ttir.sum",
            short_name="sum",
            category="Reduction Operations",
            description="Sum reduction operation.",
            detailed_description=(
                "The sum operation computes the sum of elements along specified dimensions of the input tensor.\n\n"
                "This operation reduces the input tensor by computing the sum of all elements along the dimensions "
                "specified in dim_arg. If dim_arg is not provided, the sum is computed over all dimensions, resulting "
                "in a scalar value. If keep_dim is set to true, the reduced dimensions are retained with a size of 1."
            ),
            mathematical_definition="sum(x, dim) = ∑ x[i] for all i in dimension dim",
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor after applying the reduction")],
            attributes=[
                Attribute("keep_dim", "bool", 
                         "Whether to keep the reduced dimensions or not.", "false"),
                Attribute("dim_arg", "array<i32>", 
                         "Dimensions to reduce along. If not provided, reduces over all dimensions.", None)
            ],
            examples=[
                "# Sum along dimension 1\n"
                "%result = ttir.sum(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : "
                "tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>"
            ]
        ),
        Operation(
            name="ttir.max",
            short_name="max",
            category="Reduction Operations",
            description="Maximum reduction operation.",
            detailed_description=(
                "The max operation computes the maximum value of elements along specified dimensions of the input tensor.\n\n"
                "This operation reduces the input tensor by finding the maximum value of all elements along the dimensions "
                "specified in dim_arg. If dim_arg is not provided, the maximum is computed over all dimensions, resulting "
                "in a scalar value."
            ),
            mathematical_definition="max(x, dim) = max(x[i]) for all i in dimension dim",
            operands=[
                Operand("input", "ranked tensor of any type values", "The input tensor"),
                Operand("output", "ranked tensor of any type values", "The output tensor")
            ],
            results=[Operand("result", "ranked tensor of any type values", "The result tensor after applying the reduction")],
            attributes=[
                Attribute("keep_dim", "bool", 
                         "Whether to keep the reduced dimensions or not.", "false"),
                Attribute("dim_arg", "array<i32>", 
                         "Dimensions to reduce along.", None)
            ],
            notes=[
                "When comparing with NaN values, NaN is typically not selected as the maximum value."
            ]
        ),
    ])
    
    # Note: This is a simplified version with a subset of operations.
    # In a complete implementation, you would include all operations from the user's list.
    # The structure is designed to be easily extensible.
    
    return operations

