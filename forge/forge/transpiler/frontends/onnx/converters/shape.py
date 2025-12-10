"""
ONNX Shape operation converters (Transpose, Cast, Flatten).
"""
from typing import List, Dict, Any, Tuple
from onnx import NodeProto
import onnx
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.ir.operations.shape import TransposeNode, ReshapeNode
from forge.transpiler.ir.operations.other import CastNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input, handle_validation_error


class TransposeConverter(OnnxOpConverter):
    """Converter for ONNX Transpose operation."""
    
    @classmethod
    def _normalize_perm(cls, perm: Any, input_rank: int) -> List[int]:
        """Normalize permutation to positive integers, handling negative indices."""
        if perm is None:
            return list(range(input_rank - 1, -1, -1))
        return [idx + input_rank if idx < 0 else idx for idx in map(int, perm)]
    
    @classmethod
    def _validate_perm(cls, perm: List[int], input_rank: int) -> None:
        """Validate that perm is a valid permutation of [0, 1, ..., n-1]."""
        if len(perm) != input_rank:
            raise ValueError(f"Permutation length ({len(perm)}) must equal input rank ({input_rank}). Perm: {perm}")
        
        if set(perm) != set(range(input_rank)):
            raise ValueError(f"Permutation must be a permutation of [0, 1, ..., {input_rank-1}]. Got: {perm}")
    
    @classmethod
    def _is_identity_perm(cls, perm: List[int]) -> bool:
        """Check if permutation is identity (no change)."""
        return all(i == idx for i, idx in enumerate(perm))
    
    @classmethod
    def _decompose_permutation(cls, perm: List[int]) -> List[Tuple[int, int]]:
        """
        Decompose a permutation into a sequence of two-dimension swaps.
        
        Uses a greedy algorithm that tracks the current state after each swap.
        For each position i, if the current value at i is not the target value,
        find where the target value is and swap it into place.
        
        Args:
            perm: Target permutation (list of integers)
            
        Returns:
            List of (dim0, dim1) tuples representing swap operations in the
            current coordinate system (each swap is applied to the result of previous swaps)
        """
        swap_sequence = []
        # Work with a mutable copy representing current state
        # current[i] = which original dimension is currently at position i
        current = list(range(len(perm)))
        target = list(perm)
        
        # For each position, bring the correct value into place
        for i in range(len(target)):
            # What value should be at position i in the final result?
            target_value = target[i]
            
            # What value is currently at position i?
            current_value = current[i]
            
            if current_value != target_value:
                # Find where target_value currently is
                j = current.index(target_value)
                if i != j:
                    # Swap positions i and j
                    swap_sequence.append((i, j))
                    # Update current state to reflect the swap
                    current[i], current[j] = current[j], current[i]
        
        return swap_sequence
    
    @classmethod
    def _convert_permutation_to_swaps(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                                      output_tensors: Dict[str, TensorInfo], perm: List[int],
                                      node_index: int) -> List:
        """Convert ONNX permutation to a sequence of TransposeNode operations."""
        from forge.transpiler.ir.operations.other import IdentityNode
        
        # Check if identity permutation or empty swap sequence
        swap_sequence = cls._decompose_permutation(perm) if not cls._is_identity_perm(perm) else []
        if not swap_sequence:
            node_name = node_proto.name or f"Transpose_{node_index}"
            return [IdentityNode.create(
                name=node_name,
                inputs=list(node_proto.input),
                outputs=[node_proto.output[0]],
                input_tensors=input_tensors,
                output_tensors=output_tensors
            )]
        
        # Create transpose nodes for each swap
        nodes = []
        input_info = list(input_tensors.values())[0]
        current_inputs = list(node_proto.input)
        current_input_tensors = input_tensors.copy()
        current_shape = list(input_info.shape) if input_info.shape else None
        base_name = node_proto.name or f"Transpose_{node_index}"
        onnx_dtype = getattr(input_info, 'onnx_dtype', None)
        
        for swap_idx, (dim0, dim1) in enumerate(swap_sequence):
            is_last = swap_idx == len(swap_sequence) - 1
            
            if is_last:
                node_outputs = [node_proto.output[0]]
                node_output_tensors = output_tensors.copy()
            else:
                intermediate_name = f"{base_name}_intermediate_{swap_idx}"
                node_outputs = [intermediate_name]
                if current_shape is not None:
                    new_shape = list(current_shape)
                    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
                    intermediate_shape = tuple(new_shape)
                else:
                    intermediate_shape = None
                node_output_tensors = {
                    intermediate_name: TensorInfo(name=intermediate_name, shape=intermediate_shape, onnx_dtype=onnx_dtype)
                }
            
            nodes.append(TransposeNode.create(
                name=f"{base_name}_swap_{swap_idx}",
                inputs=current_inputs,
                outputs=node_outputs,
                input_tensors=current_input_tensors,
                output_tensors=node_output_tensors,
                dim0=dim0,
                dim1=dim1
            ))
            
            if not is_last:
                current_inputs = node_outputs
                current_input_tensors = node_output_tensors
                if current_shape is not None:
                    current_shape[dim0], current_shape[dim1] = current_shape[dim1], current_shape[dim0]
        
        return nodes
    
    @classmethod
    def _get_input_rank(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                        output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any]) -> int:
        """Determine input rank from available information."""
        input_info = list(input_tensors.values())[0]
        if input_info.shape:
            return len(input_info.shape)
        
        output_info = list(output_tensors.values())[0]
        if output_info.shape:
            return len(output_info.shape)
        
        perm = attrs.get('perm')
        if perm:
            return len(perm)
        
        raise ValueError(
            f"Cannot determine input rank for Transpose node '{node_proto.name}'. "
            f"Input shape, output shape, and perm attribute are all unknown."
        )
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Transpose opset v1+: No version differences.
        
        ONNX Transpose uses perm=[0,2,1,3] which can swap multiple dimensions.
        PyTorch transpose only swaps two dimensions, so we decompose the permutation
        into a series of two-dimension swaps.
        
        If perm is omitted, defaults to (n-1, ..., 0) (reverse all dimensions).
        """
        input_rank = cls._get_input_rank(node_proto, input_tensors, output_tensors, attrs)
        perm = cls._normalize_perm(attrs.get('perm', None), input_rank)
        cls._validate_perm(perm, input_rank)
        return cls._convert_permutation_to_swaps(
            node_proto, input_tensors, output_tensors, perm, node_index
        )
    
    # All versions use the same implementation (no functional differences)
    _impl_v13 = _impl_v21 = _impl_v23 = _impl_v24 = _impl_v25 = _impl_v1


class CastConverter(OnnxOpConverter):
    """Converter for ONNX Cast operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Cast opset v1-v8: to as attribute."""
        node_name = node_proto.name if node_proto.name else f"Cast_{node_index}"
        to_dtype = attrs.get('to', None)
        torch_dtype = None
        if to_dtype:
            # Convert ONNX dtype string to torch dtype
            if isinstance(to_dtype, str):
                # Map string dtype to ONNX enum
                dtype_map = {
                    'float32': onnx.TensorProto.FLOAT,
                    'float64': onnx.TensorProto.DOUBLE,
                    'int32': onnx.TensorProto.INT32,
                    'int64': onnx.TensorProto.INT64,
                    'bool': onnx.TensorProto.BOOL,
                }
                onnx_dtype = dtype_map.get(to_dtype, onnx.TensorProto.FLOAT)
            else:
                onnx_dtype = to_dtype
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
        
        return [CastNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dtype=torch_dtype
        )]
    
    @classmethod
    def _impl_v9(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Cast opset v9+: to as attribute (same as v1)."""
        return cls._impl_v1(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Cast opset v13+: to as optional input tensor."""
        node_name = node_proto.name if node_proto.name else f"Cast_{node_index}"
        
        # Validate and extract dtype from constant input (second input, optional) or attribute
        is_valid, to_dtype, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Convert to int if it's a numpy scalar
        if to_dtype is not None:
            try:
                to_dtype = int(to_dtype)
            except (ValueError, TypeError):
                pass
        
        # Fallback to attribute
        if to_dtype is None:
            to_dtype = attrs.get('to', None)
        
        torch_dtype = None
        if to_dtype:
            if isinstance(to_dtype, str):
                dtype_map = {
                    'float32': onnx.TensorProto.FLOAT,
                    'float64': onnx.TensorProto.DOUBLE,
                    'int32': onnx.TensorProto.INT32,
                    'int64': onnx.TensorProto.INT64,
                    'bool': onnx.TensorProto.BOOL,
                }
                onnx_dtype = dtype_map.get(to_dtype, onnx.TensorProto.FLOAT)
            else:
                onnx_dtype = to_dtype
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
        
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [CastNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, dtype is embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dtype=torch_dtype
        )]


class FlattenConverter(OnnxOpConverter):
    """
    Converter for ONNX Flatten operation.
    
    Flatten converts a tensor into a 2D matrix by flattening dimensions.
    This converter maps Flatten to ReshapeNode with a calculated 2D shape.
    
    Key differences across versions:
    - v1-v9: axis range is [0, R] (non-negative only)
    - v11+: axis range is [-r, r] (negative values supported)
    """
    
    @classmethod
    def _calculate_flatten_shape(cls, input_shape: Tuple, axis: int, node_name: str = None) -> Tuple[int, int]:
        """
        Calculate the 2D output shape for Flatten operation.
        
        Args:
            input_shape: Input tensor shape
            axis: Flatten axis (already normalized to non-negative)
            node_name: Optional node name for error messages
            
        Returns:
            Tuple of (outer_dim, inner_dim) for 2D output
            
        Raises:
            ValueError: If input shape is invalid or contains None dimensions
        """
        if not input_shape:
            node_info = f" in node '{node_name}'" if node_name else ""
            raise ValueError(
                f"Flatten{node_info}: Cannot flatten tensor with unknown or empty shape. "
                f"Input shape must be known and non-empty."
            )
        
        rank = len(input_shape)
        
        # Validate that axis is within valid range
        if axis < 0 or axis > rank:
            node_info = f" in node '{node_name}'" if node_name else ""
            raise ValueError(
                f"Flatten{node_info}: Invalid axis={axis} for input rank={rank}. "
                f"Axis must be in range [0, {rank}]."
            )
        
        # Check for None dimensions and validate all dimensions are positive integers
        for i, dim in enumerate(input_shape):
            if dim is None:
                node_info = f" in node '{node_name}'" if node_name else ""
                raise ValueError(
                    f"Flatten{node_info}: Cannot calculate output shape: dimension {i} is unknown (None). "
                    f"All input dimensions must be known for Flatten operation. "
                    f"Input shape: {input_shape}"
                )
            if not isinstance(dim, int):
                node_info = f" in node '{node_name}'" if node_name else ""
                raise ValueError(
                    f"Flatten{node_info}: Invalid dimension type at index {i}: expected int, got {type(dim).__name__}. "
                    f"Input shape: {input_shape}"
                )
            if dim <= 0:
                node_info = f" in node '{node_name}'" if node_name else ""
                raise ValueError(
                    f"Flatten{node_info}: Invalid dimension value at index {i}: {dim}. "
                    f"All dimensions must be positive integers. Input shape: {input_shape}"
                )
        
        # Optimized calculation: use slicing to avoid loops where possible
        # Calculate outer dimension: product of dimensions [0:axis)
        if axis == 0:
            outer_dim = 1
        else:
            outer_dims = input_shape[:axis]
            outer_dim = 1
            for dim in outer_dims:
                outer_dim *= dim
        
        # Calculate inner dimension: product of dimensions [axis:]
        if axis >= rank:
            inner_dim = 1
        else:
            inner_dims = input_shape[axis:]
            inner_dim = 1
            for dim in inner_dims:
                inner_dim *= dim
        
        # Validate that both dimensions are positive
        if outer_dim <= 0 or inner_dim <= 0:
            node_info = f" in node '{node_name}'" if node_name else ""
            raise ValueError(
                f"Flatten{node_info}: Calculated invalid output shape ({outer_dim}, {inner_dim}). "
                f"This should not happen with valid input shape {input_shape} and axis={axis}."
            )
        
        return (outer_dim, inner_dim)
    
    @classmethod
    def _normalize_axis(cls, axis: int, input_rank: int, opset: int, node_name: str = None) -> int:
        """
        Normalize and validate axis based on opset version.
        
        Args:
            axis: Axis value (may be negative)
            input_rank: Rank of input tensor
            opset: Opset version
            node_name: Optional node name for error messages
            
        Returns:
            Normalized axis (non-negative)
            
        Raises:
            ValueError: If axis is out of valid range for the opset version
        """
        if input_rank <= 0:
            node_info = f" in node '{node_name}'" if node_name else ""
            raise ValueError(
                f"Flatten{node_info}: Invalid input rank={input_rank}. "
                f"Input tensor must have rank >= 1."
            )
        
        # v1-v9: axis must be in range [0, R] (non-negative only)
        if opset < 11:
            if axis < 0:
                node_info = f" in node '{node_name}'" if node_name else ""
                raise ValueError(
                    f"Flatten{node_info} (opset {opset}): axis must be non-negative. "
                    f"Got axis={axis}, but negative axis is only supported in opset 11+. "
                    f"Valid range: [0, {input_rank}]"
                )
            if axis > input_rank:
                node_info = f" in node '{node_name}'" if node_name else ""
                raise ValueError(
                    f"Flatten{node_info} (opset {opset}): axis={axis} is out of range [0, {input_rank}]. "
                    f"Input rank: {input_rank}"
                )
            return axis
        
        # v11+: axis can be in range [-r, r] (negative values supported)
        if axis < -input_rank or axis > input_rank:
            node_info = f" in node '{node_name}'" if node_name else ""
            raise ValueError(
                f"Flatten{node_info} (opset {opset}): axis={axis} is out of range "
                f"[-{input_rank}, {input_rank}]. Input rank: {input_rank}"
            )
        
        # Normalize negative axis to positive
        if axis < 0:
            axis = axis + input_rank
        
        return axis
    
    @classmethod
    def _validate_inputs(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                         node_name: str) -> Tuple[str, TensorInfo]:
        """
        Validate and extract input tensor information.
        
        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            node_name: Node name for error messages
            
        Returns:
            Tuple of (input_name, input_info)
            
        Raises:
            ValueError: If input validation fails
        """
        if not node_proto.input:
            raise ValueError(
                f"Flatten node '{node_name}': No input provided. "
                f"Flatten requires exactly one input tensor."
            )
        
        if len(node_proto.input) > 1:
            raise ValueError(
                f"Flatten node '{node_name}': Too many inputs ({len(node_proto.input)}). "
                f"Flatten requires exactly one input tensor."
            )
        
        data_input = node_proto.input[0]
        input_info = input_tensors.get(data_input)
        
        if input_info is None:
            raise ValueError(
                f"Flatten node '{node_name}': Input tensor '{data_input}' not found in input_tensors. "
                f"Available inputs: {list(input_tensors.keys())}"
            )
        
        return data_input, input_info
    
    @classmethod
    def _convert_flatten_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                              output_tensors: Dict[str, TensorInfo], axis: int,
                              node_index: int) -> List:
        """
        Common implementation for Flatten conversion across all opset versions.
        
        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            axis: Normalized axis (non-negative)
            node_index: Node index for naming
            
        Returns:
            List containing ReshapeNode
            
        Raises:
            ValueError: If input validation or shape calculation fails
        """
        node_name = node_proto.name if node_proto.name else f"Flatten_{node_index}"
        
        # Validate outputs
        if not node_proto.output:
            raise ValueError(
                f"Flatten node '{node_name}': No output provided. "
                f"Flatten requires exactly one output tensor."
            )
        
        output_name = node_proto.output[0]
        
        # Validate and get input info
        data_input, input_info = cls._validate_inputs(node_proto, input_tensors, node_name)
        
        input_shape = input_info.shape if input_info.shape else ()
        
        # Validate input shape is not empty
        if not input_shape:
            raise ValueError(
                f"Flatten node '{node_name}': Input tensor '{data_input}' has unknown or empty shape. "
                f"Flatten requires a tensor with known shape."
            )
        
        # Calculate 2D output shape (will raise ValueError if shape is invalid)
        output_shape = cls._calculate_flatten_shape(input_shape, axis, node_name)
        
        # Create ReshapeNode to perform the flattening
        return [ReshapeNode.create(
            name=node_name,
            inputs=[data_input],
            outputs=[output_name],
            input_tensors={data_input: input_info},
            output_tensors=output_tensors,
            shape=output_shape
        )]
    
    @classmethod
    def _extract_and_validate_axis(cls, attrs: Dict[str, Any], input_rank: int, 
                                   opset: int, node_name: str) -> int:
        """
        Extract and validate axis attribute.
        
        Args:
            attrs: Node attributes
            input_rank: Input tensor rank
            opset: Opset version
            node_name: Node name for error messages
            
        Returns:
            Normalized axis (non-negative)
            
        Raises:
            ValueError: If axis extraction or validation fails
        """
        # Extract axis attribute (default is 1)
        axis_value = attrs.get('axis', 1)
        
        # Validate axis is an integer
        try:
            axis = int(axis_value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Flatten node '{node_name}': Invalid axis attribute value '{axis_value}'. "
                f"Expected integer, got {type(axis_value).__name__}: {e}"
            )
        
        # Normalize and validate axis based on opset version
        return cls._normalize_axis(axis, input_rank, opset, node_name)
    
    @classmethod
    def _get_input_rank(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                       node_name: str) -> int:
        """
        Get and validate input tensor rank.
        
        Args:
            node_proto: ONNX node proto
            input_tensors: Input tensor info dict
            node_name: Node name for error messages
            
        Returns:
            Input tensor rank
            
        Raises:
            ValueError: If input rank cannot be determined or is invalid
        """
        # Validate inputs first
        data_input, input_info = cls._validate_inputs(node_proto, input_tensors, node_name)
        
        input_shape = input_info.shape if input_info.shape else ()
        
        if not input_shape:
            raise ValueError(
                f"Flatten node '{node_name}': Cannot determine input rank: "
                f"input tensor '{data_input}' has unknown or empty shape."
            )
        
        input_rank = len(input_shape)
        
        if input_rank == 0:
            raise ValueError(
                f"Flatten node '{node_name}': Input tensor '{data_input}' must have rank >= 1, "
                f"got rank 0 (scalar). Input shape: {input_shape}"
            )
        
        return input_rank
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Flatten opset v1-v9: axis range [0, R] (non-negative only).
        """
        node_name = node_proto.name if node_proto.name else f"Flatten_{node_index}"
        
        # Get and validate input rank
        input_rank = cls._get_input_rank(node_proto, input_tensors, node_name)
        
        # Extract and validate axis (v1-v9: non-negative only)
        axis = cls._extract_and_validate_axis(attrs, input_rank, opset=1, node_name=node_name)
        
        return cls._convert_flatten_impl(node_proto, input_tensors, output_tensors, 
                                         axis, node_index)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Flatten opset v11+: axis range [-r, r] (negative values supported).
        """
        node_name = node_proto.name if node_proto.name else f"Flatten_{node_index}"
        
        # Get and validate input rank
        input_rank = cls._get_input_rank(node_proto, input_tensors, node_name)
        
        # Extract and validate axis (v11+: supports negative)
        axis = cls._extract_and_validate_axis(attrs, input_rank, opset=11, node_name=node_name)
        
        return cls._convert_flatten_impl(node_proto, input_tensors, output_tensors, 
                                         axis, node_index)
    
    # All versions >= 11 use the same implementation
    _impl_v13 = _impl_v21 = _impl_v23 = _impl_v24 = _impl_v25 = _impl_v11

