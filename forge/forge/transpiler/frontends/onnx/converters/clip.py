"""
ONNX Clip operation converter with opset version support.
"""
from typing import List, Dict, Any, Optional, Tuple
from onnx import NodeProto
from forge.transpiler.ir.types import TensorInfo
from forge.transpiler.ir.operations.other import ClipNode, IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input


class ClipConverter(OnnxOpConverter):
    """Converter for ONNX Clip operation with opset version support."""
    
    @classmethod
    def _extract_min_max_from_inputs(cls, node_proto: NodeProto, graph_proto) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract min and max from optional input tensors (v11+).
        
        In ONNX v11+, inputs are:
        - input[0] = data
        - input[1] = min (optional)
        - input[2] = max (optional)
        
        However, some models may use named inputs where input[1] could be "max" if only max is provided.
        We check input names to determine which is which.
        
        Returns:
            Tuple of (min_val, max_val, error_message)
            - min_val: Extracted min value (None if not provided)
            - max_val: Extracted max value (None if not provided)
            - error_message: Error message if extraction failed (None if success)
        """
        min_val = None
        max_val = None
        
        # Check input names to determine which is min and which is max
        # ONNX spec: input[1] = min (optional), input[2] = max (optional)
        # But some models may put max in input[1] if only max is provided
        input_names = list(node_proto.input)
        
        # Extract from input[1] if present
        if len(input_names) > 1:
            input1_name = input_names[1].lower()
            is_valid, val, error_msg = validate_constant_input(
                node_proto, input_index=1, graph_proto=graph_proto
            )
            if not is_valid:
                return None, None, error_msg or "Failed to extract value from input[1]"
            
            if val is not None:
                if hasattr(val, 'item'):
                    val = val.item()
                val = float(val)
                
                # Determine if input[1] is min or max based on name
                if 'min' in input1_name:
                    min_val = val
                elif 'max' in input1_name:
                    max_val = val
                else:
                    # Default: assume input[1] is min (ONNX spec)
                    min_val = val
        
        # Extract from input[2] if present
        if len(input_names) > 2:
            input2_name = input_names[2].lower()
            is_valid, val, error_msg = validate_constant_input(
                node_proto, input_index=2, graph_proto=graph_proto
            )
            if not is_valid:
                return None, None, error_msg or "Failed to extract value from input[2]"
            
            if val is not None:
                if hasattr(val, 'item'):
                    val = val.item()
                val = float(val)
                
                # Determine if input[2] is min or max based on name
                if 'min' in input2_name:
                    min_val = val
                elif 'max' in input2_name:
                    max_val = val
                else:
                    # Default: assume input[2] is max (ONNX spec)
                    max_val = val
        
        return min_val, max_val, None
    
    @classmethod
    def _process_clip(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                     output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                     node_index: int, graph_proto=None, opset_version: int = 1) -> List:
        """
        Common processing logic for Clip operation.
        
        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            attrs: Node attributes
            node_index: Node index
            graph_proto: ONNX graph proto
            opset_version: Opset version
            
        Returns:
            List of TIR nodes (ClipNode)
        """
        node_name = node_proto.name or f"Clip_{node_index}"
        
        min_val = None
        max_val = None
        
        # v1-v6: min and max are attributes
        if opset_version < 11:
            min_val = attrs.get('min', None)
            max_val = attrs.get('max', None)
            
            # Convert to float if present
            if min_val is not None:
                min_val = float(min_val)
            if max_val is not None:
                max_val = float(max_val)
            
            # v6 has explicit defaults: min = -3.402823e+38, max = 3.402823e+38
            # If not provided, use defaults (don't treat as None)
            if opset_version >= 6:
                if min_val is None:
                    min_val = -3.402823e+38  # Default min for v6
                if max_val is None:
                    max_val = 3.402823e+38  # Default max for v6
        
        # v11+: min and max are optional input tensors
        else:
            min_val, max_val, error_msg = cls._extract_min_max_from_inputs(
                node_proto, graph_proto
            )
            
            if error_msg:
                raise ValueError(
                    f"Clip node '{node_name}': {error_msg}"
                )
        
        # If both min and max are None:
        # - For v6+: use defaults (already set above for v6)
        # - For v11+: return IdentityNode (no clipping)
        if min_val is None and max_val is None:
            if opset_version >= 11:
                # v11+: no limits means no clipping (IdentityNode)
                data_input = node_proto.input[0]
                if data_input not in input_tensors:
                    raise ValueError(
                        f"Clip node '{node_name}': data input '{data_input}' not found in input_tensors"
                    )
                identity_input_tensors = {data_input: input_tensors[data_input]}
                return [IdentityNode.create(
                    name=node_name,
                    inputs=[data_input],
                    outputs=[node_proto.output[0]],
                    input_tensors=identity_input_tensors,
                    output_tensors=output_tensors
                )]
            # For v6, defaults are already set above, so continue to create ClipNode
        
        # ClipNode always takes only the data input
        # min/max are passed as attributes (extracted from input tensors in v11+)
        data_input = node_proto.input[0]
        if data_input not in input_tensors:
            raise ValueError(
                f"Clip node '{node_name}': data input '{data_input}' not found in input_tensors"
            )
        clip_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ClipNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input
            outputs=[node_proto.output[0]],
            input_tensors=clip_input_tensors,
            output_tensors=output_tensors,
            min_val=min_val,
            max_val=max_val
        )]
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Clip opset v1-v5: min and max as attributes.
        """
        return cls._process_clip(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, 1)
    
    @classmethod
    def _impl_v6(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Clip opset v6-v10: min and max as attributes (same as v1, but with explicit defaults).
        """
        return cls._process_clip(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, 6)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Clip opset v11+: min and max as optional input tensors (scalars).
        """
        return cls._process_clip(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, 11)
    
    # All versions v11+ use the same implementation
    _impl_v12 = _impl_v13 = _impl_v21 = _impl_v23 = _impl_v24 = _impl_v25 = _impl_v11

