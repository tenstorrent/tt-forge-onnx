"""
ONNX Gemm (General Matrix Multiplication) operation converter.

Decomposes Gemm into: Transpose (if needed) -> MatMul -> Mul (alpha) -> Mul (beta*C) -> Add (C)
"""
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from onnx import NodeProto
import torch
from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
import onnx
from forge.transpiler.ir.operations.arithmetic import AddNode, MulNode, MatMulNode
from forge.transpiler.ir.operations.shape import TransposeNode
from forge.transpiler.ir.operations.other import FullNode, IdentityNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.converters.validation import validate_constant_input, handle_validation_error
from forge.transpiler.frontends.onnx.converters.utils import torch_dtype_to_onnx_dtype


class GemmConverter(OnnxOpConverter):
    """
    Converter for ONNX Gemm operation.
    
    Decomposes Gemm into basic TIR operations:
    - TransposeNode (if transA or transB is set)
    - MatMulNode (for matrix multiplication)
    - MulNode (for alpha and beta scaling)
    - AddNode (for adding C)
    
    Formula: Y = alpha * A' * B' + beta * C
    where A' = transpose(A) if transA else A
          B' = transpose(B) if transB else B
    """
    
    @classmethod
    def get_converter(cls, opset: int):
        """
        Get converter with opset captured.
        
        Returns a wrapper function that includes opset in the call.
        """
        base_converter = super().get_converter(opset)
        
        def converter_with_opset(node_proto, input_tensors, output_tensors, attrs, 
                                  node_index, graph_proto=None):
            return base_converter(node_proto, input_tensors, output_tensors, attrs,
                                 node_index, graph_proto, opset=opset)
        
        return converter_with_opset
    
    @classmethod
    def _create_constant_scalar(cls, name: str, value: float, torch_dtype: torch.dtype,
                                output_tensors: Dict[str, TensorInfo]) -> Tuple[str, TensorInfo]:
        """
        Create a constant scalar tensor for alpha/beta.
        
        Args:
            name: Name for the constant tensor
            value: Scalar value
            torch_dtype: PyTorch data type for the tensor
            output_tensors: Dictionary to add the output tensor info
            
        Returns:
            Tuple of (output_name, TensorInfo)
        """
        # Convert torch dtype to ONNX dtype
        onnx_dtype = torch_dtype_to_onnx_dtype(torch_dtype)
        
        # Create a scalar tensor (0D) with the value
        # PyTorch supports 0D tensors and broadcasting with them
        output_tensors[name] = TensorInfo(
            name=name,
            shape=(),  # Scalar (0D tensor)
            onnx_dtype=onnx_dtype
        )
        return name, output_tensors[name]
    
    @classmethod
    def _get_input_dtype(cls, input_tensors: Dict[str, TensorInfo]) -> torch.dtype:
        """Get the torch dtype from the first input tensor."""
        if not input_tensors:
            return torch.float32  # Default
        
        first_tensor = list(input_tensors.values())[0]
        if hasattr(first_tensor, 'torch_dtype') and first_tensor.torch_dtype is not None:
            return first_tensor.torch_dtype
        elif hasattr(first_tensor, 'onnx_dtype') and first_tensor.onnx_dtype is not None:
            return onnx_dtype_to_torch_dtype(first_tensor.onnx_dtype)
        else:
            return torch.float32  # Default
    
    @classmethod
    def _validate_inputs(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                        opset: int) -> None:
        """
        Validate Gemm inputs based on opset version.
        
        Raises:
            ValueError: If inputs are invalid
        """
        input_names = list(node_proto.input)
        
        # Check minimum inputs
        if len(input_names) < 2:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Expected at least 2 inputs (A, B), "
                f"got {len(input_names)}"
            )
        
        # Check A and B exist
        if input_names[0] not in input_tensors:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Input 'A' ('{input_names[0]}') not found in input_tensors"
            )
        if input_names[1] not in input_tensors:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Input 'B' ('{input_names[1]}') not found in input_tensors"
            )
        
        # Check C based on opset version
        if opset < 9:
            # Opset 1-8: C is required
            if len(input_names) < 3:
                raise ValueError(
                    f"Gemm node '{node_proto.name}': Opset {opset} requires 3 inputs (A, B, C), "
                    f"got {len(input_names)}"
                )
            if input_names[2] not in input_tensors:
                raise ValueError(
                    f"Gemm node '{node_proto.name}': Input 'C' ('{input_names[2]}') not found in input_tensors"
                )
        # Opset 9+: C is optional, no validation needed
    
    @classmethod
    def _validate_shapes(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                         attrs: Dict[str, Any], opset: int) -> None:
        """
        Validate input shapes for Gemm operation.
        
        Raises:
            ValueError: If shapes are invalid
        """
        input_names = list(node_proto.input)
        transA = attrs.get('transA', 0)
        transB = attrs.get('transB', 0)
        
        tensor_a = input_tensors[input_names[0]]
        tensor_b = input_tensors[input_names[1]]
        
        if tensor_a.shape is None or tensor_b.shape is None:
            logger.warning(
                f"Gemm node '{node_proto.name}': Cannot validate shapes - "
                f"one or both shapes are unknown. A: {tensor_a.shape}, B: {tensor_b.shape}"
            )
            return
        
        shape_a = tensor_a.shape
        shape_b = tensor_b.shape
        
        # Validate A shape
        if len(shape_a) < 2:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Input A must have at least 2 dimensions, "
                f"got shape {shape_a}"
            )
        
        # Validate B shape
        if len(shape_b) < 2:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Input B must have at least 2 dimensions, "
                f"got shape {shape_b}"
            )
        
        # Handle batched vs non-batched shapes
        # For batched: (batch, ..., M, K) or (batch, ..., K, M) if transA
        # For non-batched: (M, K) or (K, M) if transA
        is_batched = len(shape_a) > 2
        
        if is_batched:
            # Batched operation: last two dims are the matrix dimensions
            # Check batch dimensions match
            batch_dims_a = shape_a[:-2]
            batch_dims_b = shape_b[:-2]
            if batch_dims_a != batch_dims_b:
                raise ValueError(
                    f"Gemm node '{node_proto.name}': Batch dimensions must match. "
                    f"A has batch dims {batch_dims_a}, B has batch dims {batch_dims_b}"
                )
            
            # Get matrix dimensions (last two dims)
            if transA:
                # A is (..., K, M), effective is (..., M, K)
                m, k_a = shape_a[-1], shape_a[-2]
            else:
                # A is (..., M, K)
                m, k_a = shape_a[-2], shape_a[-1]
            
            if transB:
                # B is (..., N, K), effective is (..., K, N)
                k_b, n = shape_b[-1], shape_b[-2]
            else:
                # B is (..., K, N)
                k_b, n = shape_b[-2], shape_b[-1]
        else:
            # Non-batched: 2D matrices
            if transA:
                # A is (K, M), effective is (M, K)
                m, k_a = shape_a[1], shape_a[0]
            else:
                # A is (M, K)
                m, k_a = shape_a[0], shape_a[1]
            
            if transB:
                # B is (N, K), effective is (K, N)
                k_b, n = shape_b[1], shape_b[0]
            else:
                # B is (K, N)
                k_b, n = shape_b[0], shape_b[1]
        
        # Validate K dimension matches
        if k_a != k_b:
            raise ValueError(
                f"Gemm node '{node_proto.name}': Inner dimension mismatch. "
                f"A has K={k_a} (shape {shape_a}, transA={transA}), "
                f"B has K={k_b} (shape {shape_b}, transB={transB}). "
                f"K dimensions must match for matrix multiplication."
            )
        
        # Validate C shape if provided
        if len(input_names) >= 3 and input_names[2] in input_tensors:
            tensor_c = input_tensors[input_names[2]]
            if tensor_c.shape is not None:
                shape_c = tensor_c.shape
                # C should be broadcastable to output shape
                # For batched: (batch, ..., M, N) or broadcastable
                # For non-batched: (M, N) or broadcastable
                if is_batched:
                    # Check if batch dimensions are compatible
                    if len(shape_c) >= 2:
                        c_m, c_n = shape_c[-2], shape_c[-1]
                        # Last two dims should be broadcastable to (M, N)
                        if c_m != 1 and c_m != m and c_n != 1 and c_n != n:
                            if c_m != m or c_n != n:
                                logger.warning(
                                    f"Gemm node '{node_proto.name}': C shape {shape_c} may not be "
                                    f"broadcastable to output shape. Broadcasting will be attempted at runtime."
                                )
                else:
                    # Non-batched: C should be broadcastable to (M, N)
                    if len(shape_c) >= 2:
                        c_m, c_n = shape_c[-2], shape_c[-1]
                        if c_m != 1 and c_m != m and c_n != 1 and c_n != n:
                            if c_m != m or c_n != n:
                                logger.warning(
                                    f"Gemm node '{node_proto.name}': C shape {shape_c} may not be "
                                    f"broadcastable to ({m}, {n}). Broadcasting will be attempted at runtime."
                                )
    
    @classmethod
    def _create_transpose_if_needed(cls, node_proto: NodeProto, input_name: str,
                                    input_tensors: Dict[str, TensorInfo],
                                    output_tensors: Dict[str, TensorInfo],
                                    should_transpose: bool, base_name: str,
                                    node_index: int, transpose_idx: int) -> Tuple[str, List]:
        """
        Create a TransposeNode if needed, otherwise return IdentityNode.
        
        Args:
            node_proto: Original ONNX node
            input_name: Name of input tensor
            input_tensors: Input tensor info dict
            output_tensors: Output tensor info dict
            should_transpose: Whether to transpose
            base_name: Base name for the node
            node_index: Original node index
            transpose_idx: Index for this transpose (0 for A, 1 for B)
            
        Returns:
            Tuple of (output_name, list_of_nodes)
        """
        if not should_transpose:
            # No transpose needed, return identity
            return input_name, []
        
        # Create transpose node (transpose last two dimensions)
        transpose_name = f"{base_name}_transpose_{transpose_idx}"
        intermediate_name = f"{base_name}_transposed_{transpose_idx}"
        
        input_info = input_tensors[input_name]
        input_shape = input_info.shape
        
        # Calculate output shape (swap last two dims)
        if input_shape and len(input_shape) >= 2:
            output_shape = list(input_shape)
            output_shape[-2], output_shape[-1] = output_shape[-1], output_shape[-2]
            output_shape = tuple(output_shape)
        else:
            output_shape = None
        
        onnx_dtype = getattr(input_info, 'onnx_dtype', None)
        if onnx_dtype is None:
            onnx_dtype = onnx.TensorProto.FLOAT
        output_tensors[intermediate_name] = TensorInfo(
            name=intermediate_name,
            shape=output_shape,
            onnx_dtype=onnx_dtype
        )
        
        transpose_node = TransposeNode.create(
            name=transpose_name,
            inputs=[input_name],
            outputs=[intermediate_name],
            input_tensors={input_name: input_tensors[input_name]},
            output_tensors={intermediate_name: output_tensors[intermediate_name]},
            dim0=-2,  # Second to last dimension
            dim1=-1   # Last dimension
        )
        
        return intermediate_name, [transpose_node]
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None, opset: int = 1) -> List:
        """
        Convert Gemm opset v1.
        
        Key differences:
        - C input is required (3 inputs)
        - broadcast attribute controls broadcasting behavior
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=1)
    
    @classmethod
    def _impl_v6(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None, opset: int = 6) -> List:
        """
        Convert Gemm opset v6.
        
        Key differences from v1:
        - broadcast attribute removed, automatic unidirectional broadcasting
        - C input still required (3 inputs)
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=6)
    
    @classmethod
    def _impl_v7(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None, opset: int = 7) -> List:
        """
        Convert Gemm opset v7.
        
        Same as v6, improved documentation.
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=7)
    
    @classmethod
    def _impl_v9(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None, opset: int = 9) -> List:
        """
        Convert Gemm opset v9.
        
        Key differences:
        - C input is now optional (2-3 inputs)
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=9)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None, opset: int = 11) -> List:
        """
        Convert Gemm opset v11.
        
        Key differences:
        - Extended type support (integer types)
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=11)
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None, opset: int = 13) -> List:
        """
        Convert Gemm opset v13.
        
        Key differences:
        - Extended type support (bfloat16)
        """
        return cls._convert_gemm_impl(node_proto, input_tensors, output_tensors, attrs,
                                      node_index, graph_proto, opset=13)
    
    @classmethod
    def _convert_gemm_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                            output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                            node_index: int, graph_proto=None, opset: int = 1) -> List:
        """
        Core implementation for Gemm conversion.
        
        Decomposes: Y = alpha * A' * B' + beta * C
        into: Transpose (if needed) -> MatMul -> Mul (alpha) -> Mul (beta*C) -> Add (C)
        """
        # Extract attributes with defaults
        alpha = float(attrs.get('alpha', 1.0))
        beta = float(attrs.get('beta', 1.0))
        transA = int(attrs.get('transA', 0))
        transB = int(attrs.get('transB', 0))
        
        # Handle broadcast attribute for v1 (deprecated in v6+)
        if opset < 6:
            broadcast = int(attrs.get('broadcast', 0))
            if broadcast != 0:
                logger.debug(
                    f"Gemm node '{node_proto.name}': Using broadcast=1 (opset {opset}). "
                    f"This attribute is deprecated in opset 6+."
                )
        else:
            # Opset 6+: automatic unidirectional broadcasting
            if 'broadcast' in attrs:
                logger.warning(
                    f"Gemm node '{node_proto.name}': 'broadcast' attribute is not supported "
                    f"in opset {opset} (removed in opset 6+). It will be ignored."
                )
        
        # Validate inputs
        cls._validate_inputs(node_proto, input_tensors, opset)
        
        # Validate shapes
        cls._validate_shapes(node_proto, input_tensors, attrs, opset)
        
        # Get input names
        input_names = list(node_proto.input)
        input_a = input_names[0]
        input_b = input_names[1]
        input_c = input_names[2] if len(input_names) >= 3 else None
        
        # Check if C is provided and valid
        has_c = input_c is not None and input_c in input_tensors and input_c != ""
        
        # Get base name
        base_name = node_proto.name if node_proto.name else f"Gemm_{node_index}"
        
        # Get dtype from inputs
        dtype = cls._get_input_dtype(input_tensors)
        
        # List to collect all nodes
        nodes = []
        current_input_a = input_a
        current_input_b = input_b
        
        # Step 1: Transpose A if needed
        if transA:
            output_a, transpose_a_nodes = cls._create_transpose_if_needed(
                node_proto, input_a, input_tensors, output_tensors,
                should_transpose=True, base_name=base_name,
                node_index=node_index, transpose_idx=0
            )
            nodes.extend(transpose_a_nodes)
            current_input_a = output_a
        
        # Step 2: Transpose B if needed
        if transB:
            output_b, transpose_b_nodes = cls._create_transpose_if_needed(
                node_proto, input_b, input_tensors, output_tensors,
                should_transpose=True, base_name=base_name,
                node_index=node_index, transpose_idx=1
            )
            nodes.extend(transpose_b_nodes)
            current_input_b = output_b
        
        # Step 3: Matrix multiplication (A' * B')
        matmul_output = f"{base_name}_matmul"
        
        # Calculate output shape for matmul
        tensor_a_info = input_tensors[input_a]
        tensor_b_info = input_tensors[input_b]
        
        matmul_output_shape = None
        if tensor_a_info.shape and tensor_b_info.shape:
            shape_a = tensor_a_info.shape
            shape_b = tensor_b_info.shape
            
            # Get effective shapes
            if transA:
                m, k = shape_a[1], shape_a[0]
            else:
                m, k = shape_a[0], shape_a[1]
            
            if transB:
                n = shape_b[0]
            else:
                n = shape_b[1]
            
            # Handle batch dimensions (if any)
            if len(shape_a) > 2:
                # Batched matmul
                batch_dims = shape_a[:-2]
                matmul_output_shape = batch_dims + (m, n)
            else:
                # 2D matmul
                matmul_output_shape = (m, n)
        
        onnx_dtype_matmul = getattr(tensor_a_info, 'onnx_dtype', None)
        if onnx_dtype_matmul is None:
            onnx_dtype_matmul = onnx.TensorProto.FLOAT
        output_tensors[matmul_output] = TensorInfo(
            name=matmul_output,
            shape=matmul_output_shape,
            onnx_dtype=onnx_dtype_matmul
        )
        
        matmul_node = MatMulNode.create(
            name=f"{base_name}_matmul",
            inputs=[current_input_a, current_input_b],
            outputs=[matmul_output],
            input_tensors={
                current_input_a: input_tensors.get(current_input_a, input_tensors[input_a]),
                current_input_b: input_tensors.get(current_input_b, input_tensors[input_b])
            },
            output_tensors={matmul_output: output_tensors[matmul_output]}
        )
        nodes.append(matmul_node)
        current_output = matmul_output
        
        # Step 4: Multiply by alpha (if alpha != 1.0)
        if alpha != 1.0:
            alpha_const_name = f"{base_name}_alpha"
            alpha_output = f"{base_name}_alpha_mul"
            
            # Create constant for alpha
            cls._create_constant_scalar(alpha_const_name, alpha, dtype, output_tensors)
            
            # Create FullNode for alpha constant
            alpha_const_node = FullNode.create(
                name=alpha_const_name,
                inputs=[],
                outputs=[alpha_const_name],
                input_tensors={},
                output_tensors={alpha_const_name: output_tensors[alpha_const_name]},
                shape=(),
                fill_value=alpha,
                dtype=dtype
            )
            nodes.append(alpha_const_node)
            
            # Create MulNode for alpha multiplication
            onnx_dtype_alpha = getattr(tensor_a_info, 'onnx_dtype', None)
            if onnx_dtype_alpha is None:
                onnx_dtype_alpha = onnx.TensorProto.FLOAT
            output_tensors[alpha_output] = TensorInfo(
                name=alpha_output,
                shape=matmul_output_shape,
                onnx_dtype=onnx_dtype_alpha
            )
            
            alpha_mul_node = MulNode.create(
                name=f"{base_name}_alpha_mul",
                inputs=[alpha_const_name, current_output],
                outputs=[alpha_output],
                input_tensors={
                    alpha_const_name: output_tensors[alpha_const_name],
                    current_output: output_tensors[current_output]
                },
                output_tensors={alpha_output: output_tensors[alpha_output]}
            )
            nodes.append(alpha_mul_node)
            current_output = alpha_output
        
        # Step 5: Handle C (beta * C + result)
        if has_c and beta != 0.0:
            # Multiply C by beta
            if beta != 1.0:
                beta_const_name = f"{base_name}_beta"
                beta_c_output = f"{base_name}_beta_c"
                
                # Create constant for beta
                cls._create_constant_scalar(beta_const_name, beta, dtype, output_tensors)
                
                # Create FullNode for beta constant
                beta_const_node = FullNode.create(
                    name=beta_const_name,
                    inputs=[],
                    outputs=[beta_const_name],
                    input_tensors={},
                    output_tensors={beta_const_name: output_tensors[beta_const_name]},
                    shape=(),
                    fill_value=beta,
                    dtype=dtype
                )
                nodes.append(beta_const_node)
                
                # Create MulNode for beta * C
                onnx_dtype_beta_c = getattr(input_tensors[input_c], 'onnx_dtype', None)
                if onnx_dtype_beta_c is None:
                    onnx_dtype_beta_c = onnx.TensorProto.FLOAT
                output_tensors[beta_c_output] = TensorInfo(
                    name=beta_c_output,
                    shape=input_tensors[input_c].shape,
                    onnx_dtype=onnx_dtype_beta_c
                )
                
                beta_c_mul_node = MulNode.create(
                    name=f"{base_name}_beta_c_mul",
                    inputs=[beta_const_name, input_c],
                    outputs=[beta_c_output],
                    input_tensors={
                        beta_const_name: output_tensors[beta_const_name],
                        input_c: input_tensors[input_c]
                    },
                    output_tensors={beta_c_output: output_tensors[beta_c_output]}
                )
                nodes.append(beta_c_mul_node)
                c_input_for_add = beta_c_output
            else:
                c_input_for_add = input_c
            
            # Add C to the result
            final_output = node_proto.output[0]
            onnx_dtype_final = getattr(tensor_a_info, 'onnx_dtype', None)
            if onnx_dtype_final is None:
                onnx_dtype_final = onnx.TensorProto.FLOAT
            output_tensors[final_output] = TensorInfo(
                name=final_output,
                shape=matmul_output_shape,  # Output shape matches matmul result
                onnx_dtype=onnx_dtype_final
            )
            
            add_node = AddNode.create(
                name=f"{base_name}_add",
                inputs=[current_output, c_input_for_add],
                outputs=[final_output],
                input_tensors={
                    current_output: output_tensors[current_output],
                    c_input_for_add: output_tensors.get(c_input_for_add, input_tensors.get(c_input_for_add))
                },
                output_tensors={final_output: output_tensors[final_output]}
            )
            nodes.append(add_node)
        else:
            # No C or beta=0, final output is current_output
            final_output = node_proto.output[0]
            if current_output != final_output:
                # Create identity node to rename output
                output_tensors[final_output] = output_tensors[current_output]
                output_tensors[final_output].name = final_output
                
                identity_node = IdentityNode.create(
                    name=f"{base_name}_identity",
                    inputs=[current_output],
                    outputs=[final_output],
                    input_tensors={current_output: output_tensors[current_output]},
                    output_tensors={final_output: output_tensors[final_output]}
                )
                nodes.append(identity_node)
        
        return nodes

