"""
ONNX BatchNormalization operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from loguru import logger
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.normalization import BatchNormalizationNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter


class BatchNormalizationConverter(OnnxOpConverter):
    """Converter for ONNX BatchNormalization operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        BatchNormalization opset v1-v5: training_mode not supported, always inference mode.
        """
        return cls._convert_batchnorm_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v6(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        BatchNormalization opset v6-v8: is_test deprecated, always inference mode.
        """
        return cls._convert_batchnorm_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v9(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        BatchNormalization opset v9+: training_mode attribute introduced.
        """
        return cls._convert_batchnorm_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v14(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        BatchNormalization opset v14+: training_mode attribute (same as v9).
        """
        return cls._convert_batchnorm_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v15(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        BatchNormalization opset v15+: training_mode attribute (same as v14).
        """
        return cls._convert_batchnorm_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _convert_batchnorm_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                                output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                                node_index: int, graph_proto=None) -> List:
        """
        Common implementation for BatchNormalization conversion.
        """
        node_name = node_proto.name if node_proto.name else f"BatchNormalization_{node_index}"
        
        # Extract attributes
        eps = attrs.get('epsilon', 1e-5)
        momentum = attrs.get('momentum', 0.9)
        
        # Check training_mode (opset >= 9)
        training_mode = attrs.get('training_mode', 0)
        if training_mode != 0:
            logger.warning(
                f"BatchNormalization {node_name} has training_mode=1, but Forge only supports inference mode. "
                f"Ignoring training_mode."
            )
        
        # ONNX BatchNorm takes 5 inputs: [X, scale, B, mean, var]
        # PyTorch/TIR BatchNorm also takes 5 inputs (same structure)
        return [BatchNormalizationNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            eps=eps,
            momentum=momentum
        )]

