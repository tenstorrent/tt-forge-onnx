# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Constant operation converter with opset version support.

This module provides the converter for ONNX Constant operations, which produce
constant tensor values. Constants are handled specially - they return ConstantResult
instead of TIR nodes, and are stored directly in the graph's constants dictionary.

Opset version differences:
- v1-v8: Value stored in 'value' attribute as TensorProto
- v9+: Value can be stored in 'value' or 'sparse_value' attribute
- v12+: Additional scalar/list attributes (value_int, value_float, value_strings, etc.)
"""
from typing import Dict, Any
from collections import OrderedDict
from onnx import NodeProto
import onnx
from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.converter_result import ConstantResult
from loguru import logger


class ConstantConverter(OnnxOpConverter):
    """
    Converter for ONNX Constant operation with opset version support.

    Converts ONNX Constant nodes to ConstantResult, which contains the constant
    tensor value. Constants are stored directly in the graph's constants dictionary
    rather than being added as TIR nodes.
    """

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
    ) -> ConstantResult:
        """
        Convert ONNX Constant operation to ConstantResult.

        Opset version differences:
        - Opset v1-v8: Value stored in 'value' attribute as TensorProto
        - Opset v9+: Value can be stored in 'value' or 'sparse_value' attribute
        - Opset v12+: Additional scalar/list attributes (value_int, value_float,
          value_strings, value_ints, value_floats)

        The converter tries multiple attribute sources in priority order:
        1. 'value' (TensorProto) - most common
        2. 'sparse_value' (SparseTensorProto) - opset 9+
        3. Scalar/list attributes - opset 12+

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo (unused, constants have no inputs)
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (may contain value, sparse_value, or scalar attributes)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (used to determine which attributes to check)

        Returns:
            ConstantResult containing the constant tensor value and output name

        Raises:
            ValueError: If no valid value attribute is found or conversion fails
        """
        return cls._convert_constant_impl(node_proto, attrs, node_index, output_tensors)

    @classmethod
    def _convert_constant_impl(
        cls, node_proto: NodeProto, attrs: Dict[str, Any], node_index: int, output_tensors: OrderedDict[str, TensorInfo]
    ) -> ConstantResult:
        """
        Common implementation for Constant conversion.
        Extracts constant value from attributes and returns ConstantResult
        (constant will be stored in graph.constants by the engine).

        Args:
            node_proto: ONNX node proto
            attrs: Extracted attributes
            node_index: Index of the node in the graph
            output_tensors: Output tensor info dictionary

        Returns:
            ConstantResult containing the constant tensor value and output name.
        """
        node_name = node_proto.name if node_proto.name else f"Constant_{node_index}"

        # Get output name (Constant nodes have exactly one output)
        if not node_proto.output:
            raise ValueError(f"Constant node {node_name} has no output")
        output_name = node_proto.output[0]

        # Extract constant value from attributes
        # Priority order: value (TensorProto) > sparse_value > scalar/list attributes
        # This matches ONNX spec behavior where multiple formats are supported
        constant_value = None

        # Try 'value' attribute (TensorProto) - most common format (opset v1+)
        # Note: extract_attributes may have already converted TensorProto to numpy array
        if "value" in attrs and attrs["value"] is not None:
            try:
                from onnx import numpy_helper

                value_attr = attrs["value"]

                # Handle different formats of the value attribute
                # extract_attributes may have already converted TensorProto to numpy
                if isinstance(value_attr, (list, tuple)) or hasattr(value_attr, "shape"):
                    # Already converted to numpy array or list/tuple
                    import numpy as np

                    if isinstance(value_attr, np.ndarray):
                        constant_value = value_attr
                    else:
                        # Convert list/tuple to numpy array
                        constant_value = np.array(value_attr)
                elif hasattr(value_attr, "data_type"):
                    # It's still a TensorProto - convert to numpy using ONNX helper
                    np_array = numpy_helper.to_array(value_attr)
                    constant_value = np_array
                else:
                    # Fallback: try to get raw TensorProto from node_proto
                    # This handles cases where extract_attributes didn't process it
                    for attr in node_proto.attribute:
                        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                            np_array = numpy_helper.to_array(attr.t)
                            constant_value = np_array
                            break
            except Exception as e:
                logger.warning(f"Failed to extract value from 'value' attribute in Constant {node_name}: {e}")

        # Try 'sparse_value' attribute (SparseTensorProto) - opset >= 9
        # Note: Full sparse tensor support would require shape and indices, but we
        # simplify by extracting just the values (converting to dense)
        if constant_value is None and "sparse_value" in attrs and attrs["sparse_value"] is not None:
            try:
                from onnx import numpy_helper

                sparse_value_attr = attrs["sparse_value"]
                # Convert sparse tensor to dense (simplified approach)
                # Full sparse support would require handling indices and shape
                logger.warning(f"Constant {node_name} uses sparse_value, converting to dense (may be inefficient)")
                # Extract values from sparse tensor
                if hasattr(sparse_value_attr, "values"):
                    np_array = numpy_helper.to_array(sparse_value_attr.values)
                    constant_value = np_array
            except Exception as e:
                logger.warning(f"Failed to extract value from 'sparse_value' attribute in Constant {node_name}: {e}")

        # Try scalar value attributes (opset >= 12)
        # These provide alternative ways to specify constants for simple values
        if constant_value is None:
            if "value_int" in attrs and attrs["value_int"] is not None:
                constant_value = attrs["value_int"]
            elif "value_float" in attrs and attrs["value_float"] is not None:
                constant_value = attrs["value_float"]
            elif "value_strings" in attrs and attrs["value_strings"] is not None:
                # String constants - convert to tensor of indices or handle specially
                logger.warning(f"Constant {node_name} uses value_strings, which may not be fully supported")
                constant_value = attrs["value_strings"]
            elif "value_ints" in attrs and attrs["value_ints"] is not None:
                import numpy as np

                constant_value = np.array(attrs["value_ints"], dtype=np.int64)
            elif "value_floats" in attrs and attrs["value_floats"] is not None:
                import numpy as np

                constant_value = np.array(attrs["value_floats"], dtype=np.float32)

        if constant_value is None:
            raise ValueError(
                f"Constant {node_name} has no valid value attribute. "
                f"Expected one of: value, sparse_value, value_int, value_float, value_strings, value_ints, value_floats"
            )

        # Convert extracted value to PyTorch tensor
        # This handles various input formats (numpy array, scalar, list) and
        # ensures proper dtype conversion from ONNX to PyTorch
        import torch
        import numpy as np

        if isinstance(constant_value, np.ndarray):
            # Get dtype from ONNX attribute if available (preserves original type)
            # Otherwise infer from numpy array dtype
            onnx_dtype = None

            # Try to get dtype from raw node_proto attributes
            # This preserves the original ONNX dtype before numpy conversion
            for attr in node_proto.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    onnx_dtype = attr.t.data_type
                    break
                elif attr.name == "sparse_value" and attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                    if hasattr(attr.sparse_tensor, "values") and hasattr(attr.sparse_tensor.values, "data_type"):
                        onnx_dtype = attr.sparse_tensor.values.data_type
                        break

            # Convert ONNX dtype to PyTorch dtype, or infer from numpy array
            if onnx_dtype is not None:
                torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
            else:
                # Fallback: infer PyTorch dtype from numpy array dtype
                torch_dtype = onnx_dtype_to_torch_dtype(constant_value.dtype)

            torch_tensor = torch.from_numpy(constant_value).to(torch_dtype)
        elif isinstance(constant_value, (int, float)):
            # Scalar value (from value_int, value_float attributes)
            # Use appropriate dtype: int64 for integers, float32 for floats
            if isinstance(constant_value, int):
                torch_tensor = torch.tensor(constant_value, dtype=torch.int64)
            else:
                torch_tensor = torch.tensor(constant_value, dtype=torch.float32)
        elif isinstance(constant_value, list):
            # List of values (from value_ints, value_floats attributes)
            # Convert to numpy array first, then to PyTorch tensor
            np_array = np.array(constant_value)
            torch_dtype = onnx_dtype_to_torch_dtype(np_array.dtype)
            torch_tensor = torch.from_numpy(np_array).to(torch_dtype)
        else:
            raise ValueError(f"Constant {node_name} has unsupported value type: {type(constant_value)}")

        # Return ConstantResult instead of TIR nodes
        # The engine will handle ConstantResult specially by storing it in graph.constants
        return ConstantResult(value=torch_tensor, output_name=output_name)
