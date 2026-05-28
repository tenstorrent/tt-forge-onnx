# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX ConstantOfShape operation converter.

Converts ONNX ConstantOfShape operations directly to a ConstantResult.
ConstantOfShape creates a tensor filled with a constant value whose shape is
provided by an input tensor that must itself be a constant initializer.

Because the shape is always fully known at transpilation time, we eagerly
compute the output tensor and return a ConstantResult rather than creating a
FullNode.  This avoids the FullNode -> engine conversion two-step and stores
the value directly in tir_graph.computed_constants.

Supports opset versions 9+.
"""
from typing import Dict, Any
from collections import OrderedDict
from onnx import NodeProto
import torch
import numpy as np
from onnx import numpy_helper
from loguru import logger

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.converter_result import ConstantResult
from forge.transpiler.frontends.onnx.utils.validation import (
    validate_constant_input,
    ConverterValidationError,
)
import onnx


class ConstantOfShapeConverter(OnnxOpConverter):
    """
    Converter for ONNX ConstantOfShape operation.

    Supports opset versions 9+.

    Conversion strategy:
    - Extract shape from the (constant) input tensor.
    - Extract the fill value from the 'value' attribute (default 0.0 / float32).
    - Eagerly compute torch.full(shape, fill_value, dtype=dtype).
    - Return a ConstantResult so the engine stores the tensor directly in
      tir_graph.computed_constants — no FullNode is created.
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
        opset: int = 9,
        tir_graph=None,
    ) -> ConstantResult:
        """
        Convert ONNX ConstantOfShape operation to a ConstantResult.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (may contain 'value' attribute)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer (for constant extraction)
            opset: Opset version (9+)
            tir_graph: TIRGraph being constructed (unused here, passed for API consistency)

        Returns:
            ConstantResult with the computed fill tensor and its output name

        Raises:
            ConverterValidationError: If inputs are invalid or parameters cannot be extracted
        """
        node_name = node_proto.name or f"ConstantOfShape_{node_index}"

        try:
            # Validate inputs
            cls._validate_inputs(node_proto, input_tensors)

            # Extract shape from input (must be a constant initializer)
            shape = cls._extract_shape(node_proto, graph_proto, node_name, tir_graph=tir_graph)

            # Extract fill value and dtype from 'value' attribute (or use default)
            fill_value, dtype = cls._extract_value(node_proto, attrs, node_name)

            # Compute the constant tensor eagerly — no FullNode needed
            tensor = torch.full(shape, fill_value, dtype=dtype)

            output_name = node_proto.output[0]
            logger.trace(f"  -> ConstantOfShape '{node_name}': shape={shape}, fill={fill_value}, dtype={dtype}")
            return ConstantResult(value=tensor, output_name=output_name)

        except (ConverterValidationError, ValueError) as e:
            logger.error(f"ConstantOfShape node '{node_name}': {e}")
            raise

    @classmethod
    def _validate_inputs(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
    ) -> None:
        """
        Validate that required inputs are present.

        Raises:
            ConverterValidationError: If inputs are invalid
        """
        if len(node_proto.input) < 1:
            raise ConverterValidationError("ConstantOfShape requires 1 input (shape tensor)")

        shape_input_name = node_proto.input[0]
        if shape_input_name not in input_tensors:
            raise ConverterValidationError(f"ConstantOfShape shape input '{shape_input_name}' not found")

    @classmethod
    def _extract_shape(
        cls,
        node_proto: NodeProto,
        graph_proto,
        node_name: str,
        tir_graph=None,
    ) -> tuple:
        """
        Extract shape from input tensor (must be constant).

        Args:
            node_proto: ONNX node proto
            graph_proto: Graph proto for constant extraction
            node_name: Node name for error messages

        Returns:
            Tuple[int, ...]: Shape tuple

        Raises:
            ConverterValidationError: If shape cannot be extracted or is invalid
        """
        shape_input_name = node_proto.input[0]

        # Extract shape tensor (must be constant)
        is_valid, shape_array, error_msg = validate_constant_input(
            node_proto,
            input_index=0,
            graph_proto=graph_proto,
            input_name=shape_input_name,
            tir_graph=tir_graph,
        )

        if not is_valid or shape_array is None:
            raise ConverterValidationError(
                f"ConstantOfShape node '{node_name}': " f"Shape input must be a constant initializer. {error_msg or ''}"
            )

        # Convert to numpy array for processing
        # Handle different input types from validate_constant_input
        # It returns: scalar for size==1, list for size>1 or size==0
        if isinstance(shape_array, (list, tuple)):
            # List of values (including empty list)
            if len(shape_array) == 0:
                # Empty list = scalar output (empty shape tensor)
                shape_array = np.array([], dtype=np.int64)
            else:
                # List of values - convert to array
                shape_array = np.array(shape_array, dtype=np.int64)
        elif isinstance(shape_array, np.ndarray):
            # Already a numpy array - ensure it's 1D
            if shape_array.ndim == 0:
                # 0D array - convert to 1D
                if shape_array.size == 0:
                    shape_array = np.array([], dtype=np.int64)
                else:
                    shape_array = np.array([shape_array.item()], dtype=np.int64)
            elif shape_array.ndim > 1:
                # Multi-dimensional - flatten to 1D
                shape_array = shape_array.flatten().astype(np.int64)
            else:
                # Already 1D - ensure dtype is int64
                shape_array = shape_array.astype(np.int64)
        else:
            # Scalar value (from validate_constant_input when size==1)
            # Wrap in array to make it 1D
            shape_array = np.array([shape_array], dtype=np.int64)

        # Final validation: must be 1D
        if shape_array.ndim != 1:
            raise ConverterValidationError(
                f"ConstantOfShape node '{node_name}': "
                f"Shape input must be 1D tensor, got {shape_array.ndim}D after processing"
            )

        # Validate all values are non-negative integers
        if np.any(shape_array < 0):
            raise ConverterValidationError(
                f"ConstantOfShape node '{node_name}': " f"All shape values must be >= 0, got {shape_array.tolist()}"
            )

        # Convert to tuple of integers
        shape = tuple(int(x) for x in shape_array)

        # Handle empty shape (scalar output)
        if len(shape) == 0:
            shape = ()  # Scalar

        return shape

    @classmethod
    def _extract_value(
        cls,
        node_proto: NodeProto,
        attrs: Dict[str, Any],
        node_name: str,
    ) -> tuple:
        """
        Extract fill value from 'value' attribute.

        Args:
            node_proto: ONNX node proto
            attrs: Extracted attributes dictionary
            node_name: Node name for error messages

        Returns:
            Tuple[float, torch.dtype]: (fill_value, dtype)

        Raises:
            ConverterValidationError: If value attribute is invalid
        """
        # Check for 'value' attribute
        if "value" in attrs:
            value_attr = attrs["value"]

            try:
                # extract_attributes converts TENSOR attributes to numpy arrays
                # Handle both numpy array (from extract_attributes) and TensorProto (raw)
                if isinstance(value_attr, np.ndarray):
                    # Already a numpy array from extract_attributes
                    value_array = value_attr
                    # Get dtype from original TensorProto in node_proto
                    onnx_dtype = None
                    for attr in node_proto.attribute:
                        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                            onnx_dtype = attr.t.data_type
                            break
                    if onnx_dtype is None:
                        # Fallback: infer from numpy dtype
                        if value_array.dtype == np.float32:
                            onnx_dtype = onnx.TensorProto.FLOAT
                        elif value_array.dtype == np.float64:
                            onnx_dtype = onnx.TensorProto.DOUBLE
                        elif value_array.dtype == np.int32:
                            onnx_dtype = onnx.TensorProto.INT32
                        elif value_array.dtype == np.int64:
                            onnx_dtype = onnx.TensorProto.INT64
                        elif value_array.dtype == np.bool_:
                            onnx_dtype = onnx.TensorProto.BOOL
                        elif value_array.dtype == np.float16:
                            onnx_dtype = onnx.TensorProto.FLOAT16
                        else:
                            # Default to float32 if unknown
                            onnx_dtype = onnx.TensorProto.FLOAT
                elif hasattr(value_attr, "data_type"):
                    # Still a TensorProto - convert to numpy array
                    value_array = numpy_helper.to_array(value_attr)
                    onnx_dtype = value_attr.data_type
                else:
                    # Try to get TensorProto from node_proto directly
                    value_array = None
                    onnx_dtype = None
                    for attr in node_proto.attribute:
                        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                            value_array = numpy_helper.to_array(attr.t)
                            onnx_dtype = attr.t.data_type
                            break

                    if value_array is None:
                        raise ConverterValidationError(
                            f"ConstantOfShape node '{node_name}': " f"'value' attribute format not recognized"
                        )

                # Validate it's a one-element tensor
                if value_array.size != 1:
                    raise ConverterValidationError(
                        f"ConstantOfShape node '{node_name}': "
                        f"'value' attribute must be a one-element tensor, got shape {value_array.shape}"
                    )

                fill_value = float(value_array.item())
                dtype = onnx_dtype_to_torch_dtype(onnx_dtype)

            except Exception as e:
                raise ConverterValidationError(
                    f"ConstantOfShape node '{node_name}': " f"Failed to extract value from 'value' attribute: {e}"
                )
        else:
            # Default: 0.0, float32
            fill_value = 0.0
            dtype = torch.float32

        return fill_value, dtype
