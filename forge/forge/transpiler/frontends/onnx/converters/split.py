"""
ONNX Split operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.shape import SplitNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input


class SplitConverter(OnnxOpConverter):
    """Converter for ONNX Split operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Split opset v1-v12: split as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Split_{node_index}"
        
        # Get split attributes
        split_sizes = attrs.get('split', None)  # ONNX attribute name
        axis = attrs.get('axis', 0)  # Dimension along which to split
        
        # Create one SplitNode representing the entire split operation
        split_node = SplitNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),  # All outputs
            input_tensors=input_tensors,
            output_tensors=output_tensors,  # All outputs
            split_sizes=split_sizes,
            dim=axis
        )
        
        return [split_node]
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Split opset v13+: split as input tensor (second input).
        If split is not provided as input, divides evenly.
        """
        node_name = node_proto.name if node_proto.name else f"Split_{node_index}"
        axis = attrs.get('axis', 0)  # Dimension along which to split
        
        # Validate and extract split sizes from constant input (second input, optional)
        is_valid, split_sizes, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Convert split_sizes to list if it's a tuple or scalar
        if split_sizes is not None:
            if isinstance(split_sizes, (list, tuple)):
                split_sizes = [int(x) for x in split_sizes]
            else:
                split_sizes = [int(split_sizes)]
        
        # If split_sizes not found, will divide evenly (handled by SplitNode)
        
        # Create one SplitNode representing the entire split operation
        split_node = SplitNode.create(
            name=node_name,
            inputs=[node_proto.input[0]],  # Only data input, split is embedded
            outputs=list(node_proto.output),  # All outputs
            input_tensors=input_tensors,
            output_tensors=output_tensors,  # All outputs
            split_sizes=split_sizes,
            dim=axis
        )
        
        return [split_node]

