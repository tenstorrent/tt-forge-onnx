"""
ONNX Activation operation converters (Relu, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyRelu, Dropout).
"""
from typing import List, Dict, Any, Optional, Tuple
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.activation import (
    ReluNode, SigmoidNode, TanhNode, SoftmaxNode, LogSoftmaxNode, LeakyReluNode, DropoutNode
)
from forge.transpiler.ir.operations.other import IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input


class ReluConverter(OnnxOpConverter):
    """Converter for ONNX Relu operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Relu opset v1+: No version differences in converter logic.
        
        Note: The 'consumed_inputs' attribute in v1 is a legacy optimization
        attribute and should be ignored. Type constraints and function body
        (v14+) are handled by ONNX, not the converter.
        """
        node_name = node_proto.name if node_proto.name else f"Relu_{node_index}"
        # Ignore consumed_inputs attribute (legacy, deprecated)
        return [ReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class SigmoidConverter(OnnxOpConverter):
    """Converter for ONNX Sigmoid operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Sigmoid opset v1+: No version differences in converter logic.
        
        Note: The 'consumed_inputs' attribute in v1 is a legacy optimization
        attribute and should be ignored. Type constraints (bfloat16 in v13+)
        are handled by ONNX, not the converter.
        """
        node_name = node_proto.name if node_proto.name else f"Sigmoid_{node_index}"
        # Ignore consumed_inputs attribute (legacy, deprecated)
        return [SigmoidNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class TanhConverter(OnnxOpConverter):
    """Converter for ONNX Tanh operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Tanh opset v1+: No version differences in converter logic.
        
        Note: The 'consumed_inputs' attribute in v1 is a legacy optimization
        attribute and should be ignored. Type constraints (bfloat16 in v13+)
        are handled by ONNX, not the converter.
        """
        node_name = node_proto.name if node_proto.name else f"Tanh_{node_index}"
        # Ignore consumed_inputs attribute (legacy, deprecated)
        return [TanhNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]


class SoftmaxConverter(OnnxOpConverter):
    """Converter for ONNX Softmax operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Softmax opset v1-v12: axis defaults to 1.
        
        Note: When axis is not provided in attrs, ONNX doesn't automatically set it,
        so we need to use the correct default based on opset version.
        """
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        # For opset v1-v12, default axis is 1
        axis = attrs.get('axis', 1)
        return [SoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Softmax opset v13+: axis defaults to -1.
        
        Note: When axis is not provided in attrs, ONNX doesn't automatically set it,
        so we need to use the correct default based on opset version.
        Type constraints and function body (v13+) are handled by ONNX, not the converter.
        """
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        # For opset v13+, default axis is -1
        axis = attrs.get('axis', -1)
        return [SoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]


class LogSoftmaxConverter(OnnxOpConverter):
    """Converter for ONNX LogSoftmax operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        LogSoftmax opset v1-v12: axis defaults to 1.
        
        Note: v1 and v11 have the same behavior (axis defaults to 1),
        so a single _impl_v1 handles both.
        """
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        # For opset v1-v12, default axis is 1
        axis = attrs.get('axis', 1)
        return [LogSoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        LogSoftmax opset v13+: axis defaults to -1.
        
        Note: When axis is not provided in attrs, ONNX doesn't automatically set it,
        so we need to use the correct default based on opset version.
        Type constraints and function body (v13+) are handled by ONNX, not the converter.
        """
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        # For opset v13+, default axis is -1
        axis = attrs.get('axis', -1)
        return [LogSoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]


class LeakyReluConverter(OnnxOpConverter):
    """Converter for ONNX LeakyRelu operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """LeakyRelu opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"LeakyRelu_{node_index}"
        alpha = attrs.get('alpha', 0.01)
        return [LeakyReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            negative_slope=alpha
        )]


class DropoutConverter(OnnxOpConverter):
    """Converter for ONNX Dropout operation with optimizations."""
    
    @classmethod
    def _extract_ratio_from_input(cls, node_proto: NodeProto, graph_proto, 
                                  default: float = 0.5) -> float:
        """
        Extract ratio from input if provided (second input, index 1).
        
        Returns:
            Extracted ratio value or default if not provided.
        
        Raises:
            ValueError: If ratio input is provided but cannot be converted to float.
            TypeError: If ratio input is provided but has invalid type.
        """
        if len(node_proto.input) <= 1:
            return default
        
        is_valid, ratio_value, _ = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
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
    def _extract_training_mode_from_input(cls, node_proto: NodeProto, graph_proto,
                                          default: bool = False) -> bool:
        """
        Extract training_mode from input if provided (third input, index 2).
        
        Returns:
            Extracted training mode value or default if not provided.
        
        Raises:
            ValueError: If training_mode input is provided but cannot be converted to bool.
            TypeError: If training_mode input is provided but has invalid type.
        """
        if len(node_proto.input) <= 2:
            return default
        
        is_valid, training_mode_value, _ = validate_constant_input(
            node_proto, input_index=2, graph_proto=graph_proto
        )
        if is_valid and training_mode_value is not None:
            try:
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
    def _create_dropout_node(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                             output_tensors: Dict[str, TensorInfo], ratio: float,
                             training: bool, seed: int, node_index: int) -> List:
        """
        Create DropoutNode or IdentityNode based on optimization opportunities.
        
        Optimizations:
        1. If training=False (inference mode) → IdentityNode
        2. If ratio=0 (no dropout) → IdentityNode
        
        Returns:
            List containing either DropoutNode or IdentityNode.
        """
        node_name = node_proto.name if node_proto.name else f"Dropout_{node_index}"
        data_input = node_proto.input[0]
        output_name = node_proto.output[0]
        
        # Optimization 1: Inference mode (training=False) → Identity
        if not training:
            return [IdentityNode.create(
                name=node_name,
                inputs=[data_input],
                outputs=[output_name],
                input_tensors={data_input: input_tensors[data_input]},
                output_tensors=output_tensors
            )]
        
        # Optimization 2: ratio=0 (no dropout) → Identity
        if ratio == 0.0:
            return [IdentityNode.create(
                name=node_name,
                inputs=[data_input],
                outputs=[output_name],
                input_tensors={data_input: input_tensors[data_input]},
                output_tensors=output_tensors
            )]
        
        # Normal case: create DropoutNode
        return [DropoutNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[output_name],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            p=ratio,
            training=training,
            seed=seed
        )]
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Dropout opset v1-v6: Uses `is_test` and `ratio` attributes.
        
        Attributes:
            - ratio: Dropout probability (default: 0.5)
            - is_test: If nonzero, run in inference mode (default: 0 = training)
            - consumed_inputs: Legacy attribute, ignored
        """
        # Extract attributes
        ratio = attrs.get('ratio', 0.5)
        is_test = attrs.get('is_test', 0)
        seed = attrs.get('seed', 0)
        
        # Convert is_test to training flag (is_test=0 means training, is_test=1 means inference)
        training = (is_test == 0)
        
        # Create node (with optimizations)
        return cls._create_dropout_node(
            node_proto, input_tensors, output_tensors, ratio, training, seed, node_index
        )
    
    @classmethod
    def _impl_v7(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Dropout opset v7-v10: Uses `ratio` attribute, training mode from graph context.
        
        Attributes:
            - ratio: Dropout probability (default: 0.5)
            - seed: Random seed (optional, default: 0)
        
        Note: Training mode is determined from graph context (not from attribute).
        For now, we default to training=True. In a full implementation, this would
        be determined from the graph's training flag.
        """
        # Extract attributes
        ratio = attrs.get('ratio', 0.5)
        seed = attrs.get('seed', 0)
        
        # Default to training mode (graph context would determine this in full implementation)
        # TODO: Extract from graph context if available
        training = True
        
        # Create node (with optimizations)
        return cls._create_dropout_node(
            node_proto, input_tensors, output_tensors, ratio, training, seed, node_index
        )
    
    @classmethod
    def _impl_v12(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Dropout opset v12+: Uses `ratio` and `training_mode` as optional inputs, `seed` attribute.
        
        Attributes:
            - seed: Random seed (optional, default: 0)
        
        Inputs:
            - data: Required input tensor
            - ratio: Optional input tensor (default: 0.5 if not provided)
            - training_mode: Optional input tensor (default: False if not provided)
        
        Note: Extracts ratio and training_mode from inputs if present (as constants),
        otherwise uses defaults. Dynamic inputs are not yet supported.
        """
        # Extract seed attribute
        seed = attrs.get('seed', 0)
        
        # Extract ratio and training_mode from inputs
        ratio = cls._extract_ratio_from_input(node_proto, graph_proto, default=0.5)
        training = cls._extract_training_mode_from_input(node_proto, graph_proto, default=False)
        
        # Create node (with optimizations)
        return cls._create_dropout_node(
            node_proto, input_tensors, output_tensors, ratio, training, seed, node_index
        )

