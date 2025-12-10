"""
Input validation and graceful handling utilities for ONNX converters.
"""
from typing import Dict, List, Any, Optional, Tuple
from onnx import NodeProto
from loguru import logger
from forge.transpiler.ir.types import TensorInfo


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_attributes(node_proto: NodeProto,
                       attrs: Dict[str, Any],
                       required_attrs: List[str] = None,
                       attr_ranges: Dict[str, Tuple[Any, Any]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate node attributes.
    
    Args:
        node_proto: ONNX node proto
        attrs: Extracted attributes dictionary
        required_attrs: List of required attribute names
        attr_ranges: Dictionary mapping attribute names to (min, max) value ranges
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required attributes
    if required_attrs:
        for attr_name in required_attrs:
            if attr_name not in attrs or attrs[attr_name] is None:
                return False, (
                    f"Node {node_proto.name or node_proto.op_type} requires attribute '{attr_name}' "
                    f"but it was not found or is None"
                )
    
    # Check attribute value ranges
    if attr_ranges:
        for attr_name, (min_val, max_val) in attr_ranges.items():
            if attr_name in attrs and attrs[attr_name] is not None:
                attr_val = attrs[attr_name]
                if isinstance(attr_val, (int, float)):
                    if attr_val < min_val or attr_val > max_val:
                        return False, (
                            f"Node {node_proto.name or node_proto.op_type} attribute '{attr_name}' "
                            f"has value {attr_val}, but must be in range [{min_val}, {max_val}]"
                        )
    
    return True, None


def validate_constant_input(node_proto: NodeProto,
                           input_index: int,
                           graph_proto,
                           input_name: str = None) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Validate and extract constant value from an input tensor (for opset >= 11/13 operations).
    
    Args:
        node_proto: ONNX node proto
        input_index: Index of the input to check
        graph_proto: ONNX graph proto (for accessing initializers)
        input_name: Optional input name (if None, uses node_proto.input[input_index])
        
    Returns:
        Tuple of (is_valid, constant_value, error_message)
        - is_valid: True if input is a constant or optional
        - constant_value: Extracted constant value (None if optional and not provided)
        - error_message: Error message if validation failed
    """
    if input_index >= len(node_proto.input):
        # Optional input not provided - this is valid
        return True, None, None
    
    input_name = input_name or node_proto.input[input_index]
    
    if graph_proto is None:
        return False, None, (
            f"Node {node_proto.name or node_proto.op_type} requires constant input '{input_name}' "
            f"but graph_proto is not available"
        )
    
    # Try to find in initializers
    for init in graph_proto.initializer:
        if init.name == input_name:
            try:
                from onnx import numpy_helper
                import numpy
                constant_array = numpy_helper.to_array(init)
                # Convert to Python native type
                if constant_array.size == 1:
                    constant_value = constant_array.item()
                    if isinstance(constant_value, numpy.ndarray):
                        constant_value = constant_value.tolist()
                else:
                    constant_value = constant_array.tolist()
                return True, constant_value, None
            except Exception as e:
                return False, None, (
                    f"Node {node_proto.name or node_proto.op_type} failed to extract constant "
                    f"from input '{input_name}': {e}"
                )
    
    # Input not found in initializers - it's dynamic (not supported yet)
    return False, None, (
        f"Node {node_proto.name or node_proto.op_type} requires constant input '{input_name}' "
        f"but it was not found in initializers. Dynamic input tensors are not yet supported."
    )


def handle_validation_error(node_proto: NodeProto,
                           error_msg: str,
                           strict: bool = False) -> bool:
    """
    Handle validation errors gracefully.
    
    Args:
        node_proto: ONNX node proto
        error_msg: Error message
        strict: If True, raise exception. If False, log warning and return False.
        
    Returns:
        True if error was handled gracefully, False if should skip this node
    """
    if strict:
        raise ValidationError(f"{node_proto.op_type} (node: {node_proto.name}): {error_msg}")
    else:
        logger.warning(f"{node_proto.op_type} (node: {node_proto.name}): {error_msg}")
        return False

