"""
ONNX Arithmetic operation converters (Add, Sub, Mul, Div, MatMul).
"""
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, MatMulNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter


def _are_shapes_equal(shape_a: Tuple, shape_b: Tuple) -> bool:
    """Check if two shapes are exactly equal."""
    return shape_a == shape_b


def _are_shapes_compatible_for_broadcasting(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are compatible for NumPy-style broadcasting (OPSET 7+).
    
    Two shapes are compatible if:
    - They are equal, OR
    - For each dimension (aligned from right), they are equal OR one is 1 OR one is missing
    """
    if shape_a == shape_b:
        return True
    
    # Align shapes from the right
    len_a, len_b = len(shape_a), len(shape_b)
    max_len = max(len_a, len_b)
    
    for i in range(max_len):
        # Get dimensions from right to left
        dim_a = shape_a[-(i+1)] if i < len_a else 1
        dim_b = shape_b[-(i+1)] if i < len_b else 1
        
        # Dimensions are compatible if:
        # - They are equal, OR
        # - One of them is 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False
    
    return True


def _validate_limited_broadcasting(shape_a: Tuple, shape_b: Tuple, 
                                    axis: Optional[int], op_type: str) -> None:
    """
    Validate broadcasting for OPSET 1-6 (limited broadcasting with axis attribute).
    
    In OPSET 1-6, broadcasting is limited:
    - B's shape must match a contiguous subset of A's shape
    - If axis is specified, B aligns starting at that axis
    - If axis is not specified, suffix matching is used (B aligns from right)
    
    Args:
        shape_a: Shape of tensor A (left operand)
        shape_b: Shape of tensor B (right operand)
        axis: Optional axis attribute (None means suffix matching)
        op_type: Operation type for error messages
        
    Raises:
        ValueError: If shapes are not compatible for limited broadcasting
    """
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)
    
    if axis is None:
        # Suffix matching: B must match the suffix of A
        if len(shape_b_list) > len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                f"Shape B {shape_b} has more dimensions than shape A {shape_a}. "
                f"B must match the suffix of A."
            )
        
        # Check from right to left
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[-(i+1)]
            dim_b = shape_b_list[-(i+1)]
            
            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at position {len(shape_a_list) - i - 1}: {dim_a} vs {dim_b}"
                )
    else:
        # Axis-specified matching: B aligns starting at axis
        if axis < 0 or axis >= len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Axis {axis} is out of range for shape A {shape_a} "
                f"(valid range: 0 to {len(shape_a_list) - 1})"
            )
        
        if len(shape_b_list) > len(shape_a_list) - axis:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Shape B {shape_b} has {len(shape_b_list)} dimensions, but only "
                f"{len(shape_a_list) - axis} dimensions available starting at axis {axis} "
                f"in shape A {shape_a}"
            )
        
        # Check dimensions starting at axis
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[axis + i]
            dim_b = shape_b_list[i]
            
            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at axis {axis + i}: {dim_a} vs {dim_b}"
                )


def _validate_broadcast_attributes(op_type: str, attrs: Dict[str, Any], 
                                   input_tensors: Dict[str, TensorInfo],
                                   opset: int) -> None:
    """
    Validate broadcast and axis attributes based on opset version.
    
    This function only validates - it does not return processed attributes.
    Raises ValueError if validation fails.
    
    Args:
        op_type: Operation type (Add, Sub, Mul, Div)
        attrs: Extracted attributes dictionary
        input_tensors: Dictionary of input tensor information
        opset: Opset version
        
    Raises:
        ValueError: If shapes are incompatible for broadcasting
    """
    # Extract broadcast and axis attributes
    broadcast = attrs.get("broadcast", 0)
    axis = attrs.get("axis", None)
    
    # Get input shapes and validate
    if len(input_tensors) < 2:
        raise ValueError(
            f"{op_type} node: Expected 2 inputs, got {len(input_tensors)}"
        )
    
    input_names = list(input_tensors.keys())
    tensor_a = input_tensors[input_names[0]]
    tensor_b = input_tensors[input_names[1]]
    
    shape_a = tensor_a.shape
    shape_b = tensor_b.shape
    
    if shape_a is None or shape_b is None:
        logger.warning(
            f"{op_type} node: Cannot validate broadcasting - one or both shapes are unknown. "
            f"Shape A: {shape_a}, Shape B: {shape_b}"
        )
        return  # Skip validation if shapes are unknown
    
    shapes_match = _are_shapes_equal(shape_a, shape_b)
    shapes_compatible_multidir = _are_shapes_compatible_for_broadcasting(shape_a, shape_b)
    
    # Handle based on opset version
    if opset <= 6:
        # OPSET 1-6: Limited broadcasting, requires broadcast=1 attribute
        if not shapes_match:
            if broadcast == 0:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET {opset}): "
                    f"Shapes {shape_a} and {shape_b} don't match and broadcast=0. "
                    f"Set broadcast=1 to enable broadcasting."
                )
            else:
                # broadcast=1 is set, validate limited broadcasting
                _validate_limited_broadcasting(shape_a, shape_b, axis, op_type)
                if axis is not None:
                    logger.debug(
                        f"{op_type} node: Using axis={axis} for broadcasting (OPSET {opset})"
                    )
    
    else:
        # OPSET 7+: Multidirectional broadcasting always enabled
        if broadcast != 0 or axis is not None:
            logger.warning(
                f"{op_type} node: 'broadcast' and 'axis' attributes are not supported "
                f"in OPSET {opset} (removed in OPSET 7+). These attributes will be ignored. "
                f"Multidirectional broadcasting is always enabled."
            )
        
        # Validate shapes are compatible for multidirectional broadcasting
        if not shapes_match and not shapes_compatible_multidir:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET {opset}): "
                f"Shapes {shape_a} and {shape_b} are not compatible for "
                f"multidirectional broadcasting. "
                f"Two dimensions are compatible if they are equal OR one is 1."
            )


class BinaryArithmeticConverter(OnnxOpConverter):
    """
    Unified converter for binary arithmetic operations: Add, Sub, Mul, Div.
    
    This converter handles all four operations (Add, Sub, Mul, Div) using a single
    implementation, while maintaining separate operator nodes for each operation type.
    The operator nodes use PyTorch API (torch.add, torch.sub, torch.mul, torch.div).
    
    Broadcasting Handling:
    - OPSET 1-6: Validates `broadcast=1` attribute and handles `axis` attribute
    - OPSET 7+: Multidirectional broadcasting always enabled (attributes ignored)
    - PyTorch operations handle broadcasting automatically
    """
    
    # Mapping from ONNX op type to corresponding node class
    _OP_NODE_MAP = {
        "Add": AddNode,
        "Sub": SubNode,
        "Mul": MulNode,
        "Div": DivNode,
    }
    
    @classmethod
    def get_converter(cls, opset: int):
        """
        Get converter with opset captured.
        
        Returns a wrapper function that includes opset in the call.
        """
        base_converter = super().get_converter(opset)
        
        def converter_with_opset(node_proto, input_tensors, output_tensors, attrs, 
                                  node_index, graph_proto=None):
            return base_converter(node_proto, input_tensors, output_tensors, attrs,
                                 node_index, graph_proto, opset=opset)
        
        return converter_with_opset
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None, opset: int = 1) -> List:
        """
        Convert binary arithmetic operations (Add, Sub, Mul, Div).
        
        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (may include broadcast, axis)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)
            
        Returns:
            List containing a single node instance (AddNode, SubNode, MulNode, or DivNode)
        """
        op_type = node_proto.op_type
        
        # Get the appropriate node class for this operation
        node_class = cls._OP_NODE_MAP.get(op_type)
        if node_class is None:
            raise ValueError(
                f"Unsupported binary arithmetic operation: {op_type}. "
                f"Supported operations: {list(cls._OP_NODE_MAP.keys())}"
            )
        
        # Validate broadcast/axis attributes
        # This validates shapes based on opset version and attributes
        _validate_broadcast_attributes(
            op_type=op_type,
            attrs=attrs,
            input_tensors=input_tensors,
            opset=opset
        )
        
        # Generate node name if not provided
        node_name = node_proto.name if node_proto.name else f"{op_type}_{node_index}"
        
        # Create and return the appropriate node (no processed attributes)
        return [node_class.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class MatMulConverter(OnnxOpConverter):
    """Converter for ONNX MatMul operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """MatMul opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"MatMul_{node_index}"
        return [MatMulNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]

