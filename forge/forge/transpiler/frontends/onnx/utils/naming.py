"""
Node naming utilities for ONNX to TIR conversion.
Provides consistent naming conventions for TIR nodes.
"""
import re
from typing import Optional
from loguru import logger


def sanitize_name(name: str) -> str:
    """
    Sanitize a node name to be Python-identifier friendly.
    
    Replaces invalid characters with underscores and ensures the name
    doesn't start with a digit.
    
    Args:
        name: Original node name
        
    Returns:
        Sanitized name safe for use as Python identifier
    """
    if not name:
        return name
    
    # Replace invalid characters with underscores
    # Invalid: ., /, :, -, spaces, and other special chars
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure name doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f'node_{sanitized}'
    
    # Ensure name is not empty
    if not sanitized:
        sanitized = 'unnamed_node'
    
    return sanitized


def ensure_unique_name(name: str, existing_names: set, counter: int = 0) -> str:
    """
    Ensure a node name is unique by appending a counter if needed.
    
    Args:
        name: Proposed node name
        existing_names: Set of existing node names
        counter: Starting counter value
        
    Returns:
        Unique node name
    """
    if name not in existing_names:
        return name
    
    # Try appending counter
    base_name = name
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    
    unique_name = f"{base_name}_{counter}"
    return unique_name


def generate_clean_variable_name(op_type: str, counter: int) -> str:
    """
    Generate a clean Python variable name following ForgeWriter pattern.
    
    Examples:
        Conv2d -> conv2d_0
        Relu -> relu_1
        MatMul -> matmul_2
        Add -> add_3
    
    Args:
        op_type: Operation type (e.g., "Conv2d", "Relu", "MatMul")
        counter: Counter for uniqueness (per operation type)
        
    Returns:
        Clean variable name (e.g., "conv2d_0", "relu_1")
    """
    # Convert to lowercase and handle CamelCase
    op_type_lower = op_type.lower()
    
    # Handle special cases for common operations
    # Map to ForgeWriter-style names
    op_name_map = {
        'conv2d': 'conv2d',
        'conv1d': 'conv1d',
        'conv3d': 'conv3d',
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanh': 'tanh',
        'softmax': 'softmax',
        'logsoftmax': 'log_softmax',
        'leakyrelu': 'leaky_relu',
        'matmul': 'matmul',
        'add': 'add',
        'subtract': 'subtract',
        'multiply': 'multiply',
        'divide': 'divide',
        'maxpool2d': 'max_pool2d',
        'averagepool2d': 'average_pool2d',
        'globalaveragepool': 'global_average_pool',
        'batchnorm': 'batchnorm',
        'transpose': 'transpose',
        'reshape': 'reshape',
        'squeeze': 'squeeze',
        'unsqueeze': 'unsqueeze',
        'concat': 'concat',
        'clip': 'clip',
        'cast': 'cast',
        'pad': 'pad',
        'reducesum': 'reduce_sum',
        'reducemean': 'reduce_mean',
        'reducemax': 'reduce_max',
        'flatten': 'flatten',
        'gemm': 'gemm',
    }
    
    # Use mapped name if available, otherwise use lowercase op_type
    base_name = op_name_map.get(op_type_lower, op_type_lower)
    
    # Generate name with counter
    return f"{base_name}_{counter}"

