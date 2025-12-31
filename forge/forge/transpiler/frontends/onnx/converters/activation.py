# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Activation operation converters.

This module provides converters for ONNX activation operations:
- Relu, Sigmoid, Tanh: Simple element-wise activations
- Softmax, LogSoftmax: Normalization operations with dimension parameter
- LeakyRelu: Parametric activation with negative slope
- Dropout: Stochastic regularization operation with training/inference modes

All converters follow the OnnxOpConverter pattern and handle opset version differences
where applicable.
"""
from typing import List, Dict, Any
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.activation import (
    ReluNode,
    SigmoidNode,
    TanhNode,
    SoftmaxNode,
    LogSoftmaxNode,
    LeakyReluNode,
    DropoutNode,
)
from forge.transpiler.operations.other import IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class ReluConverter(OnnxOpConverter):
    """
    Converter for ONNX Relu operation.

    Converts ONNX Relu to TIR ReluNode. Relu is a simple element-wise activation
    function: output = max(0, input). No opset version differences.
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
    ) -> List:
        """
        Convert ONNX Relu operation to TIR ReluNode.

        Relu opset v1+: No version differences in converter logic.
        The 'consumed_inputs' attribute in v1 is a legacy optimization attribute
        and should be ignored. Type constraints and function body (v14+) are
        handled by ONNX shape inference, not the converter.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (consumed_inputs is ignored)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (unused, no version differences)

        Returns:
            List containing a single ReluNode
        """
        node_name = node_proto.name if node_proto.name else f"Relu_{node_index}"
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [ReluNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]


class SigmoidConverter(OnnxOpConverter):
    """
    Converter for ONNX Sigmoid operation.

    Converts ONNX Sigmoid to TIR SigmoidNode. Sigmoid is an element-wise activation:
    output = 1 / (1 + exp(-input)). No opset version differences.
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
    ) -> List:
        """
        Convert ONNX Sigmoid operation to TIR SigmoidNode.

        Sigmoid opset v1+: No version differences in converter logic.
        The 'consumed_inputs' attribute in v1 is a legacy optimization attribute
        and should be ignored. Type constraints (bfloat16 in v13+) are handled
        by ONNX shape inference, not the converter.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (consumed_inputs is ignored)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (unused, no version differences)

        Returns:
            List containing a single SigmoidNode
        """
        node_name = node_proto.name if node_proto.name else f"Sigmoid_{node_index}"
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [SigmoidNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]


class TanhConverter(OnnxOpConverter):
    """
    Converter for ONNX Tanh operation.

    Converts ONNX Tanh to TIR TanhNode. Tanh is an element-wise activation:
    output = tanh(input). No opset version differences.
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
    ) -> List:
        """
        Convert ONNX Tanh operation to TIR TanhNode.

        Tanh opset v1+: No version differences in converter logic.
        The 'consumed_inputs' attribute in v1 is a legacy optimization attribute
        and should be ignored. Type constraints (bfloat16 in v13+) are handled
        by ONNX shape inference, not the converter.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (consumed_inputs is ignored)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (unused, no version differences)

        Returns:
            List containing a single TanhNode
        """
        node_name = node_proto.name if node_proto.name else f"Tanh_{node_index}"
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [TanhNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]


class SoftmaxConverter(OnnxOpConverter):
    """
    Converter for ONNX Softmax operation.

    Converts ONNX Softmax to TIR SoftmaxNode. Softmax applies exponential normalization
    along a specified dimension: output = exp(input) / sum(exp(input), dim=axis).
    Handles opset version differences in axis default value.
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
    ) -> List:
        """
        Convert ONNX Softmax operation to TIR SoftmaxNode.

        Opset version differences:
        - Opset v1-v12: axis defaults to 1 (second dimension)
        - Opset v13+: axis defaults to -1 (last dimension)

        When axis is not provided in attrs, ONNX doesn't automatically set it,
        so we need to use the correct default based on opset version.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (axis may be missing)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (used to determine axis default)

        Returns:
            List containing a single SoftmaxNode
        """
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        # Determine axis default based on opset version
        # This matches ONNX spec behavior for missing axis attribute
        if opset < 13:
            axis = attrs.get("axis", 1)
        else:
            axis = attrs.get("axis", -1)

        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [SoftmaxNode.create(name=node_name, inputs=input_dict, outputs=output_dict, dim=axis)]


class LogSoftmaxConverter(OnnxOpConverter):
    """
    Converter for ONNX LogSoftmax operation.

    Converts ONNX LogSoftmax to TIR LogSoftmaxNode. LogSoftmax applies log-softmax
    normalization along a specified dimension: output = log(softmax(input, dim=axis)).
    Handles opset version differences in axis default value.
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
    ) -> List:
        """
        Convert ONNX LogSoftmax operation to TIR LogSoftmaxNode.

        Opset version differences:
        - Opset v1-v12: axis defaults to 1 (second dimension)
        - Opset v13+: axis defaults to -1 (last dimension)

        When axis is not provided in attrs, ONNX doesn't automatically set it,
        so we need to use the correct default based on opset version.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (axis may be missing)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (used to determine axis default)

        Returns:
            List containing a single LogSoftmaxNode
        """
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        # Determine axis default based on opset version
        # This matches ONNX spec behavior for missing axis attribute
        if opset < 13:
            axis = attrs.get("axis", 1)
        else:
            axis = attrs.get("axis", -1)

        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [LogSoftmaxNode.create(name=node_name, inputs=input_dict, outputs=output_dict, dim=axis)]


class LeakyReluConverter(OnnxOpConverter):
    """
    Converter for ONNX LeakyRelu operation.

    Converts ONNX LeakyRelu to TIR LeakyReluNode. LeakyRelu is a parametric activation:
    output = max(alpha * input, input). No opset version differences.
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
    ) -> List:
        """
        Convert ONNX LeakyRelu operation to TIR LeakyReluNode.

        LeakyRelu opset v1+: No version differences in converter logic.
        The alpha attribute defaults to 0.01 if not provided.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes (alpha defaults to 0.01)
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (unused, no version differences)

        Returns:
            List containing a single LeakyReluNode
        """
        node_name = node_proto.name if node_proto.name else f"LeakyRelu_{node_index}"
        # Extract alpha (negative slope) with default value 0.01
        alpha = attrs.get("alpha", 0.01)

        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [LeakyReluNode.create(name=node_name, inputs=input_dict, outputs=output_dict, negative_slope=alpha)]


class DropoutConverter(OnnxOpConverter):
    """
    Converter for ONNX Dropout operation with optimizations.

    Converts ONNX Dropout to TIR DropoutNode or IdentityNode (optimized cases).
    Handles multiple opset versions with different input/attribute patterns.

    Optimizations:
    - Inference mode (training=False) → IdentityNode (no-op)
    - Zero dropout (ratio=0) → IdentityNode (no-op)

    Opset differences:
    - v1-v6: Uses 'is_test' attribute and 'ratio' attribute
    - v7-v10: Uses 'ratio' attribute, training mode from graph context
    - v12+: Uses 'ratio' and 'training_mode' inputs, 'seed' attribute
    """

    @classmethod
    def _extract_ratio_from_input(cls, node_proto: NodeProto, graph_proto, default: float = 0.5) -> float:
        """
        Extract dropout ratio from input if provided (second input, index 1).

        In opset v12+, ratio can be provided as a constant input tensor instead
        of an attribute. This method extracts it from the graph initializers.

        Args:
            node_proto: ONNX node protocol buffer
            graph_proto: ONNX graph protocol buffer (for accessing initializers)
            default: Default ratio value if not provided

        Returns:
            Extracted ratio value or default if not provided

        Raises:
            ValueError: If ratio input is provided but cannot be converted to float
        """
        if len(node_proto.input) <= 1:
            return default

        # Check if ratio is provided as a constant input (opset v12+)
        is_valid, ratio_value, _ = validate_constant_input(node_proto, input_index=1, graph_proto=graph_proto)
        if is_valid and ratio_value is not None:
            try:
                return float(ratio_value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to extract ratio from input '{node_proto.input[1]}' "
                    f"for Dropout node '{node_proto.name}': {e}. "
                    f"Expected a numeric value, got {type(ratio_value).__name__}: {ratio_value}"
                ) from e
        return default

    @classmethod
    def _extract_training_mode_from_input(cls, node_proto: NodeProto, graph_proto, default: bool = False) -> bool:
        """
        Extract training mode from input if provided (third input, index 2).

        In opset v12+, training_mode can be provided as a constant input tensor
        instead of inferred from graph context. This method extracts it from
        the graph initializers.

        Args:
            node_proto: ONNX node protocol buffer
            graph_proto: ONNX graph protocol buffer (for accessing initializers)
            default: Default training mode value if not provided

        Returns:
            Extracted training mode value or default if not provided

        Raises:
            ValueError: If training_mode input is provided but cannot be converted to bool
        """
        if len(node_proto.input) <= 2:
            return default

        # Check if training_mode is provided as a constant input (opset v12+)
        is_valid, training_mode_value, _ = validate_constant_input(node_proto, input_index=2, graph_proto=graph_proto)
        if is_valid and training_mode_value is not None:
            try:
                # Convert various types to bool (ONNX may use int/float for boolean)
                if isinstance(training_mode_value, bool):
                    return training_mode_value
                elif isinstance(training_mode_value, (int, float)):
                    return bool(training_mode_value)
                else:
                    return bool(training_mode_value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to extract training_mode from input '{node_proto.input[2]}' "
                    f"for Dropout node '{node_proto.name}': {e}. "
                    f"Expected a boolean or numeric value, got {type(training_mode_value).__name__}: {training_mode_value}"
                ) from e
        return default

    @classmethod
    def _create_dropout_node(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        ratio: float,
        training: bool,
        seed: int,
        node_index: int,
    ) -> List:
        """
        Create DropoutNode or IdentityNode based on optimization opportunities.

        Applies optimizations to avoid unnecessary computation:
        1. Inference mode (training=False) → IdentityNode (dropout is no-op in inference)
        2. Zero dropout (ratio=0) → IdentityNode (no elements dropped)

        These optimizations reduce graph complexity and improve performance.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            ratio: Dropout probability (0.0 to 1.0)
            training: Training mode flag
            seed: Random seed for reproducibility
            node_index: Index of node in graph

        Returns:
            List containing either DropoutNode or IdentityNode
        """
        node_name = node_proto.name if node_proto.name else f"Dropout_{node_index}"
        data_input = node_proto.input[0]
        output_name = node_proto.output[0]

        # Build input/output dicts for identity optimization check
        input_dict, output_dict = build_input_output_dicts(
            node_proto, input_tensors, output_tensors, input_names=[data_input] if data_input else None
        )

        # Optimization 1: Inference mode → Identity (dropout has no effect)
        # In inference, dropout should pass input unchanged, which is what Identity does
        if not training:
            return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]

        # Optimization 2: Zero dropout → Identity (no elements dropped)
        # When ratio=0, dropout is effectively a no-op even in training mode
        if ratio == 0.0:
            return [IdentityNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]

        # Normal case: create DropoutNode with actual dropout behavior
        # Build full input dict (may include ratio/training_mode inputs in opset v12+)
        dropout_input_dict, _ = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        return [
            DropoutNode.create(
                name=node_name, inputs=dropout_input_dict, outputs=output_dict, p=ratio, training=training, seed=seed
            )
        ]

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
    ) -> List:
        """
        Dropout converter with version-specific dispatch.

        - Opset v1-v6: Uses `is_test` and `ratio` attributes
        - Opset v7-v11: Uses `ratio` attribute, training mode from graph context
        - Opset v12+: Uses `ratio` and `training_mode` as optional inputs, `seed` attribute
        """
        if opset < 7:
            # v1-v6: is_test and ratio attributes
            ratio = attrs.get("ratio", 0.5)
            is_test = attrs.get("is_test", 0)
            seed = attrs.get("seed", 0)
            training = is_test == 0  # is_test=0 means training, is_test=1 means inference
        elif opset < 12:
            # v7-v11: ratio attribute, training mode from graph context
            ratio = attrs.get("ratio", 0.5)
            seed = attrs.get("seed", 0)
            # Default to training mode (graph context would determine this in full implementation)
            # TODO: Extract from graph context if available
            training = True
        else:
            # v12+: ratio and training_mode as optional inputs
            seed = attrs.get("seed", 0)
            ratio = cls._extract_ratio_from_input(node_proto, graph_proto, default=0.5)
            training = cls._extract_training_mode_from_input(node_proto, graph_proto, default=False)

        # Create node (with optimizations)
        return cls._create_dropout_node(node_proto, input_tensors, output_tensors, ratio, training, seed, node_index)
