# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced attribute extraction for ONNX nodes.
Comprehensive support for all ONNX attribute types with proper error handling.
"""
import onnx
from onnx import numpy_helper
from typing import Dict, Any
from loguru import logger


class AttributeParsingError(ValueError):
    """Exception raised when an attribute cannot be parsed."""


def extract_attr_value(attr) -> Any:
    """
    Extract a single attribute value from ONNX attribute.

    Supports all ONNX attribute types:
    - INT, FLOAT, STRING (scalars)
    - INTS, FLOATS, STRINGS (lists)
    - TENSOR, TENSORS (tensors)
    - SPARSE_TENSOR, SPARSE_TENSORS (sparse tensors)
    - GRAPH, GRAPHS (subgraphs)
    - TYPE_PROTO, TYPE_PROTOS (type information)

    Args:
        attr: ONNX AttributeProto object

    Returns:
        Extracted attribute value (type depends on attribute type)

    Raises:
        AttributeParsingError: If attribute type is unsupported or cannot be parsed
    """
    try:
        # Scalar types
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            # Handle both bytes and string
            if isinstance(attr.s, bytes):
                return attr.s.decode("utf-8")
            return attr.s

        # List types
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]

        # Tensor types
        elif attr.type == onnx.AttributeProto.TENSOR:
            if not attr.HasField("t"):
                raise AttributeParsingError(f"TENSOR attribute '{attr.name}' has no tensor field")
            return numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.TENSORS:
            if not attr.tensors:
                raise AttributeParsingError(f"TENSORS attribute '{attr.name}' has no tensors")
            return [numpy_helper.to_array(t) for t in attr.tensors]

        # Sparse tensor types (opset >= 11)
        elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
            if not attr.HasField("sparse_tensor"):
                raise AttributeParsingError(f"SPARSE_TENSOR attribute '{attr.name}' has no sparse_tensor field")
            # Return the sparse tensor proto (converters can handle conversion to dense if needed)
            return attr.sparse_tensor
        elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
            if not attr.sparse_tensors:
                raise AttributeParsingError(f"SPARSE_TENSORS attribute '{attr.name}' has no sparse_tensors")
            return list(attr.sparse_tensors)

        # Graph types (subgraphs)
        elif attr.type == onnx.AttributeProto.GRAPH:
            if not attr.HasField("g"):
                raise AttributeParsingError(f"GRAPH attribute '{attr.name}' has no graph field")
            return attr.g
        elif attr.type == onnx.AttributeProto.GRAPHS:
            if not attr.graphs:
                raise AttributeParsingError(f"GRAPHS attribute '{attr.name}' has no graphs")
            return list(attr.graphs)

        # Type proto types (opset >= 15)
        elif attr.type == onnx.AttributeProto.TYPE_PROTO:
            if not attr.HasField("tp"):
                raise AttributeParsingError(f"TYPE_PROTO attribute '{attr.name}' has no type_proto field")
            return attr.tp
        elif attr.type == onnx.AttributeProto.TYPE_PROTOS:
            if not attr.type_protos:
                raise AttributeParsingError(f"TYPE_PROTOS attribute '{attr.name}' has no type_protos")
            return list(attr.type_protos)

        else:
            raise AttributeParsingError(
                f"Unsupported attribute type: {attr.type} for attribute '{attr.name}'. "
                f"ONNX AttributeProto type enum: {attr.type}"
            )

    except AttributeParsingError:
        raise
    except Exception as e:
        raise AttributeParsingError(
            f"Failed to extract value from attribute '{attr.name}' of type {attr.type}: {e}"
        ) from e


def extract_attributes(node, strict: bool = False) -> Dict[str, Any]:
    """
    Extract attributes from an ONNX node as-is (without name mapping).

    This function extracts ONNX attribute values without converting attribute names.
    Each OnnxOpConverter subclass is responsible for converting ONNX attribute names
    to PyTorch-compatible names as needed.

    Args:
        node: ONNX NodeProto object
        strict: If True, raise exceptions on parsing errors. If False, log warnings and skip.

    Returns:
        Dictionary of extracted attributes with original ONNX attribute names

    Raises:
        AttributeParsingError: If strict=True and an attribute cannot be parsed
    """
    attrs = {}

    for attr in node.attribute:
        attr_name = attr.name

        try:
            attr_value = extract_attr_value(attr)
        except AttributeParsingError as e:
            if strict:
                raise
            else:
                logger.warning(f"Skipping attribute '{attr_name}' in node '{node.name or node.op_type}': {e}")
                continue

        # Store attribute with original ONNX name
        # Converters will handle name mapping to PyTorch format
        attrs[attr_name] = attr_value

    return attrs
