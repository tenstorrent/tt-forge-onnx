"""
ONNX transpiler engine for converting ONNX models to Forge graphs.
"""
import onnx
from onnx import numpy_helper, shape_inference, checker
import torch
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
from typing import List as ListType

from forge.transpiler.ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.converters.utils import torch_dtype_to_onnx_dtype
from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.frontends.onnx.converters import (
    extract_attributes,
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
)
from forge.transpiler.frontends.onnx.converters.pad import PadConverter
from forge.transpiler.frontends.onnx.converters.split import SplitConverter
from forge.transpiler.frontends.onnx.converters.squeeze import SqueezeConverter
from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter
from forge.transpiler.frontends.onnx.converters.unsqueeze import UnsqueezeConverter
from forge.transpiler.frontends.onnx.converters.concat import ConcatConverter
from forge.transpiler.frontends.onnx.converters.clip import ClipConverter
from forge.transpiler.frontends.onnx.converters.conv import ConvConverter
from forge.transpiler.frontends.onnx.converters.batchnorm import BatchNormalizationConverter
from forge.transpiler.frontends.onnx.converters.arithmetic import BinaryArithmeticConverter, MatMulConverter
from forge.transpiler.frontends.onnx.converters.gemm import GemmConverter
from forge.transpiler.frontends.onnx.converters.activation import ReluConverter, SigmoidConverter, TanhConverter, SoftmaxConverter, LogSoftmaxConverter, LeakyReluConverter, DropoutConverter
from forge.transpiler.frontends.onnx.converters.reduction import ReduceSumConverter, ReduceMeanConverter, ReduceMaxConverter
from forge.transpiler.frontends.onnx.converters.pooling import MaxPoolConverter, AveragePoolConverter, GlobalAveragePoolConverter
from forge.transpiler.frontends.onnx.converters.shape import TransposeConverter, CastConverter, FlattenConverter
from forge.transpiler.frontends.onnx.converters.constant import ConstantConverter
from forge.transpiler.frontends.onnx.utils.naming import sanitize_name, ensure_unique_name


class ONNXToForgeTranspiler:
    """Main transpiler class for converting ONNX models to Forge graphs."""
    def __init__(self, debug: bool = False, freeze_params: bool = False, validate_model: bool = True):
        """
        Initialize the transpiler.
        
        Args:
            debug: Enable debug mode (compare outputs with ONNXRuntime)
            freeze_params: If True, all initializers become constants (non-trainable).
                          If False, uses heuristics to distinguish parameters from constants.
            validate_model: If True, validate ONNX model before conversion (default: True)
        """
        self.debug = debug
        self.freeze_params = freeze_params
        self.validate_model = validate_model
        self.onnx_model = None  # Store original model for debug mode
        if debug:
            try:
                import onnxruntime
            except ImportError:
                logger.warning("onnxruntime not available. Debug mode requires: pip install onnxruntime")
                self.debug = False
        
        # Opset version (will be set during transpile)
        self.opset = 1  # Default opset
        
        # Dictionary mapping ONNX op types to converter methods
        # Will be built dynamically based on opset version
        self._op_converters = {}
        
        # Track node names for uniqueness
        self._node_names = set()
        
        # Name sanitization system for clean Python variable names
        self._output_name_map: Dict[str, str] = {}  # Original ONNX name -> sanitized name
        self._used_output_names: set = set()  # Track used sanitized names for uniqueness
        self._op_type_counters: Dict[str, int] = {}  # Counter per operation type (for ForgeWriter-style names)
    
    def _generate_clean_output_name(self, op_type: str) -> str:
        """
        Generate a clean Python variable name for an output, following ForgeWriter pattern.
        
        This generates a NEW name based on operation type (e.g., "conv2d_0", "relu_1"),
        completely ignoring the original ONNX output name. This ensures consistent,
        readable variable names in generated code.
        
        Note: This is different from sanitize_name() which cleans an existing name.
        - sanitize_name(): Cleans an existing name (removes invalid chars)
        - _generate_clean_output_name(): Generates a new name based on op_type
        
        Args:
            op_type: Operation type (e.g., "Conv2d", "Relu")
            
        Returns:
            Clean variable name (e.g., "conv2d_0", "relu_1")
        """
        from forge.transpiler.frontends.onnx.utils.naming import generate_clean_variable_name, ensure_unique_name
        
        # Get or initialize counter for this operation type
        if op_type not in self._op_type_counters:
            self._op_type_counters[op_type] = 0
        
        # Generate base name based on op_type and counter
        base_name = generate_clean_variable_name(op_type, self._op_type_counters[op_type])
        
        # Ensure uniqueness across all outputs
        clean_name = ensure_unique_name(base_name, self._used_output_names)
        self._used_output_names.add(clean_name)
        
        # Increment counter for next time
        self._op_type_counters[op_type] += 1
        
        return clean_name

    def _is_constant(self, name: str, tensor: torch.Tensor) -> bool:
        """
        Determine if an initializer should be treated as a constant (non-trainable)
        vs a parameter (trainable).
        
        Heuristics based on TVM's approach:
        - Constants: name contains "constant", or
        - Constants: not weight/bias and (int/bool dtype or scalar shape), or
        - Parameters: weights, biases, and other trainable tensors
        
        Args:
            name: Name of the initializer
            tensor: The tensor value
            
        Returns:
            True if constant, False if parameter
        """
        name_lower = name.lower()
        
        # Check if name contains "constant"
        if "constant" in name_lower:
            return True
        
        # Check if it's a scalar (shape length 0)
        if len(tensor.shape) == 0:
            return True
        
        # Check if it's not weight/bias and has int/bool dtype
        if "weight" not in name_lower and "bias" not in name_lower:
            dtype_str = str(tensor.dtype).lower()
            if "int" in dtype_str or "bool" in dtype_str:
                return True
        
        # Otherwise, it's a parameter (trainable)
        return False

    def _get_tensor_info(self, value_info_map, name):
        """
        Retrieves shape and dtype and wraps it in a TensorInfo object.
        """
        if name not in value_info_map:
            # Handle the case where the name is an input that is not defined in value_info 
            return TensorInfo(name, None, onnx.TensorProto.UNDEFINED)
        
        vi = value_info_map[name]
        tensor_type = vi.type.tensor_type
        
        onnx_dtype = tensor_type.elem_type
        shape = None

        if tensor_type.HasField("shape"):
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param) # Represents dynamic dimension name
                else:
                    shape.append(None) # Represents unknown dynamic dimension
            shape = tuple(shape)
        
        return TensorInfo(name, shape, onnx_dtype)

    def _get_opset_version(self, onnx_model: onnx.ModelProto) -> int:
        """
        Extract opset version from ONNX model.
        
        Returns:
            Opset version (defaults to 1 if not found)
        """
        try:
            opset_in_model = 1
            if onnx_model.opset_import:
                # Find ai.onnx opset (default domain)
                for opset_identifier in onnx_model.opset_import:
                    # As per ONNX spec, default domain is "ai.onnx" or ""
                    if str(opset_identifier.domain) in ["ai.onnx", ""]:
                        opset_in_model = opset_identifier.version
                        break
            return opset_in_model
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not extract opset version from model: {e}. Defaulting to opset 1.")
            return 1
    
    def _build_convert_map(self, opset: int) -> Dict[str, callable]:
        """
        Build converter map based on opset version.
        All operations now use versioned converter classes following TVM pattern.
        """
        convert_map = {
            # Arithmetic operations
            "Add": BinaryArithmeticConverter.get_converter(opset),
            "Sub": BinaryArithmeticConverter.get_converter(opset),
            "Mul": BinaryArithmeticConverter.get_converter(opset),
            "Div": BinaryArithmeticConverter.get_converter(opset),
            "MatMul": MatMulConverter.get_converter(opset),
            "Gemm": GemmConverter.get_converter(opset),
            
            # Activation operations
            "Relu": ReluConverter.get_converter(opset),
            "Sigmoid": SigmoidConverter.get_converter(opset),
            "Tanh": TanhConverter.get_converter(opset),
            "Softmax": SoftmaxConverter.get_converter(opset),
            "LogSoftmax": LogSoftmaxConverter.get_converter(opset),
            "LeakyRelu": LeakyReluConverter.get_converter(opset),
            "Dropout": DropoutConverter.get_converter(opset),
            
            # Reduction operations
            "ReduceSum": ReduceSumConverter.get_converter(opset),
            "ReduceMean": ReduceMeanConverter.get_converter(opset),
            "ReduceMax": ReduceMaxConverter.get_converter(opset),
            
            # Pooling operations
            "MaxPool": MaxPoolConverter.get_converter(opset),
            "AveragePool": AveragePoolConverter.get_converter(opset),
            "GlobalAveragePool": GlobalAveragePoolConverter.get_converter(opset),
            
            # Shape operations
            "Transpose": TransposeConverter.get_converter(opset),
            "Cast": CastConverter.get_converter(opset),
            "Flatten": FlattenConverter.get_converter(opset),
            "Pad": PadConverter.get_converter(opset),
            "Split": SplitConverter.get_converter(opset),
            "Squeeze": SqueezeConverter.get_converter(opset),
            "Reshape": ReshapeConverter.get_converter(opset),
            "Unsqueeze": UnsqueezeConverter.get_converter(opset),
            "Concat": ConcatConverter.get_converter(opset),
            "Clip": ClipConverter.get_converter(opset),
            "Conv": ConvConverter.get_converter(opset),
            "BatchNormalization": BatchNormalizationConverter.get_converter(opset),
            "Constant": ConstantConverter.get_converter(opset),
        }
        
        return convert_map
    
    def transpile(self, onnx_model: onnx.ModelProto) -> TIRGraph:
        """Transpile an ONNX model to a TIR graph."""
        logger.info("Starting Transpilation with Shape Inference...")
        
        # Store original model for debug mode and converter access
        self.onnx_model = onnx_model
        
        # 0. Model Validation (optional)
        if self.validate_model:
            try:
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model validation passed")
            except onnx.checker.ValidationError as e:
                logger.error(f"ONNX model validation failed: {e}")
                raise ValueError(f"Invalid ONNX model: {e}") from e
            except Exception as e:
                logger.warning(f"ONNX model validation encountered an error: {e}. Proceeding anyway.")
        
        # Extract opset version from model
        self.opset = self._get_opset_version(onnx_model)
        logger.info(f"ONNX Opset Version: {self.opset}")
        
        # Build converter map based on opset version
        self._op_converters = self._build_convert_map(self.opset)
        
        # Reset node names tracking for this transpilation
        self._node_names = set()
        
        # Reset name sanitization system
        self._output_name_map = {}
        self._used_output_names = set()
        self._op_type_counters = {}
        
        # 1. Perform Shape Inference
        try:
            inferred_model = shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            logger.error(f"Shape inference failed: {e}. Proceeding without inferred shapes.")
            inferred_model = onnx_model

        # 2. Remove Initializers from Graph Inputs
        inferred_model = remove_initializers_from_input(inferred_model)

        graph_proto = inferred_model.graph
        # Store graph_proto for converter methods to access initializers
        self.graph_proto = graph_proto
        tir_graph = TIRGraph(name=graph_proto.name, frontend_model=onnx_model if self.debug else None, debug_mode=self.debug)

        # 3. Create map of all value infos (including model inputs/outputs)
        # This map contains all tensor names mapped to their full ONNX metadata (shape, type)
        value_info_map = {vi.name: vi for vi in graph_proto.value_info}
        value_info_map.update({vi.name: vi for vi in graph_proto.input})
        value_info_map.update({vi.name: vi for vi in graph_proto.output})

        # 4. Process Initializers (Weights/Parameters vs Constants)
        for initializer in graph_proto.initializer:
            np_array = numpy_helper.to_array(initializer)
            
            # Use the new utility to correctly determine PyTorch dtype
            onnx_dtype = initializer.data_type
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
            
            torch_tensor = torch.from_numpy(np_array).to(torch_dtype)
            
            # Distinguish between parameters (trainable) and constants (non-trainable)
            if self.freeze_params:
                # If freeze_params is True, all initializers become constants
                tir_graph.constants[initializer.name] = torch_tensor
            else:
                # Use heuristics to determine if constant or parameter
                if self._is_constant(initializer.name, torch_tensor):
                    tir_graph.constants[initializer.name] = torch_tensor
                else:
                    tir_graph.params[initializer.name] = torch_tensor

        # 5. Process remaining Graph Inputs (Actual model inputs)
        # Use utility function to get input names (original ONNX names)
        original_inputs = get_inputs_names(graph_proto)
        tir_graph.original_inputs = original_inputs
        
        # Sanitize input names (for code generation) - optimized batch processing
        # Pre-compute sets for faster lookups
        params_and_constants = set(tir_graph.params.keys()) | set(tir_graph.constants.keys())
        sanitized_inputs = []
        for original_input in original_inputs:
            # Inputs that are not parameters/constants need sanitization
            if original_input not in params_and_constants:
                clean_name = self._output_name_map.get(original_input)
                if clean_name is None:
                    # Generate clean name for input (use "input" as base)
                    clean_name = ensure_unique_name(
                        sanitize_name(original_input) or f"input_{len(sanitized_inputs)}",
                        self._used_output_names
                    )
                    self._output_name_map[original_input] = clean_name
                    self._used_output_names.add(clean_name)
                    tir_graph.original_to_sanitized[original_input] = clean_name
                    tir_graph.sanitized_to_original[clean_name] = original_input
                sanitized_inputs.append(clean_name)
            else:
                # Parameters and constants keep original names (for parameter loading)
                sanitized_inputs.append(original_input)
                # Still add to mapping for consistency (identity mapping)
                if original_input not in tir_graph.original_to_sanitized:
                    tir_graph.original_to_sanitized[original_input] = original_input
                    tir_graph.sanitized_to_original[original_input] = original_input
        
        tir_graph.inputs = sanitized_inputs

        # 6. Early Validation: Check for unsupported operations BEFORE conversion
        unsupported_ops = []  # Collect unsupported operations with details
        
        # First pass: Identify all unsupported operations and collect details
        for i, node_proto in enumerate(graph_proto.node):
            op_type = node_proto.op_type
            converter_method = self._op_converters.get(op_type, None)
            
            if converter_method is None:
                # Unsupported operation found - collect details
                node_name = node_proto.name if node_proto.name else f"{op_type}_{i}"
                
                # Get input tensor info (shapes and dtypes)
                input_tensors = {}
                for name in node_proto.input:
                    input_tensors[name] = self._get_tensor_info(value_info_map, name)
                
                # Extract attributes
                attrs = extract_attributes(node_proto)
                
                # Build input details string
                input_details = []
                for input_name in node_proto.input:
                    if input_name in input_tensors:
                        tensor_info = input_tensors[input_name]
                        shape_str = str(tensor_info.shape) if tensor_info.shape else "unknown"
                        dtype_str = str(tensor_info.torch_dtype) if tensor_info.torch_dtype else "unknown"
                        input_details.append(f"{input_name}: shape={shape_str}, dtype={dtype_str}")
                    else:
                        input_details.append(f"{input_name}: unknown")
                
                unsupported_ops.append({
                    'op_type': op_type,
                    'node_name': node_name,
                    'node_index': i,
                    'input_details': input_details,
                    'attrs': attrs
                })
        
        # Report unsupported operations upfront with full details
        if unsupported_ops:
            unsupported_types = sorted(set([op['op_type'] for op in unsupported_ops]))
            error_msg = (
                f"Found {len(unsupported_ops)} unsupported ONNX operation(s) in the model:\n"
                f"  Unsupported operation types: {', '.join(unsupported_types)}\n"
                f"  Total unsupported nodes: {len(unsupported_ops)}\n"
                f"  Details:\n"
            )
            for op in unsupported_ops:
                attrs_str = ", ".join([f"{k}={v}" for k, v in op['attrs'].items()]) if op['attrs'] else "none"
                error_msg += (
                    f"    - {op['op_type']} (node: {op['node_name']}, index: {op['node_index']})\n"
                    f"      Inputs: {', '.join(op['input_details'])}\n"
                    f"      Attributes: {attrs_str}\n"
                )
            logger.error(error_msg)
        else:
            logger.info("All ONNX operations are supported. Proceeding with conversion.")
        
        # 7. Process Nodes - Convert supported operations with input validation
        unsupported_op_types = set([op['op_type'] for op in unsupported_ops]) if unsupported_ops else set()
        invalid_nodes = []  # Collect nodes that fail validation
        
        for i, node_proto in enumerate(graph_proto.node):
            op_type = node_proto.op_type
            
            # Skip unsupported operations (already identified in first pass)
            if op_type in unsupported_op_types:
                continue
            
            # --- Input Tensor Metadata: Now uses TensorInfo ---
            input_tensors = {}
            for name in node_proto.input:
                tensor_info = self._get_tensor_info(value_info_map, name)
                # If tensor not found in value_info_map, check if it's a parameter or constant
                if tensor_info.shape is None and tensor_info.onnx_dtype == onnx.TensorProto.UNDEFINED:
                    if name in tir_graph.params:
                        # Parameter found - create TensorInfo with shape from the tensor
                        param_tensor = tir_graph.params[name]
                        param_shape = tuple(param_tensor.shape) if param_tensor.shape else None
                        param_onnx_dtype = torch_dtype_to_onnx_dtype(param_tensor.dtype)
                        tensor_info = TensorInfo(name, param_shape, param_onnx_dtype)
                    elif name in tir_graph.constants:
                        # Constant found - create TensorInfo with shape from the tensor
                        const_tensor = tir_graph.constants[name]
                        const_shape = tuple(const_tensor.shape) if const_tensor.shape else None
                        const_onnx_dtype = torch_dtype_to_onnx_dtype(const_tensor.dtype)
                        tensor_info = TensorInfo(name, const_shape, const_onnx_dtype)
                input_tensors[name] = tensor_info

            # --- Output Tensor Metadata: Now uses TensorInfo ---
            output_tensors = {}
            for name in node_proto.output:
                output_tensors[name] = self._get_tensor_info(value_info_map, name)

            # Use enhanced attribute extraction
            attrs = extract_attributes(node_proto)

            # Basic output validation: Check that node has at least one output
            if len(node_proto.output) == 0:
                logger.warning(
                    f"Skipping {op_type} node '{node_proto.name or f'{op_type}_{i}'}' "
                    f"at index {i}: No outputs provided"
                )
                invalid_nodes.append({
                    'op_type': op_type,
                    'node_name': node_proto.name or f"{op_type}_{i}",
                    'node_index': i,
                    'reason': 'No outputs provided'
                })
                continue

            # Get converter method for this op type (already validated in first pass)
            converter_method = self._op_converters[op_type]
            
            try:
                # Special handling for Constant nodes
                if op_type == "Constant":
                    # Constant nodes don't create TIR nodes, they just store values
                    tir_nodes = converter_method(node_proto, input_tensors, output_tensors, attrs, i, self.graph_proto)
                    
                    # Extract constant value from node_proto (set by converter)
                    if hasattr(node_proto, '_forge_constant_value'):
                        output_name = node_proto.output[0]
                        tir_graph.constants[output_name] = node_proto._forge_constant_value
                        logger.debug(f"Stored constant '{output_name}' from Constant node")
                    else:
                        logger.warning(
                            f"Constant node '{node_proto.name or f'Constant_{i}'}' "
                            f"at index {i} did not set constant value"
                        )
                        invalid_nodes.append({
                            'op_type': op_type,
                            'node_name': node_proto.name or f"{op_type}_{i}",
                            'node_index': i,
                            'reason': 'Constant value not extracted'
                        })
                    continue
                
                # Use converter method to create TIR node(s)
                # All converters now follow the same pattern and need graph_proto
                tir_nodes = converter_method(node_proto, input_tensors, output_tensors, attrs, i, self.graph_proto)
                
                # Validate that converter returned at least one node
                if not tir_nodes or len(tir_nodes) == 0:
                    logger.warning(
                        f"Skipping {op_type} node '{node_proto.name or f'{op_type}_{i}'}' "
                        f"at index {i}: Converter returned no nodes"
                    )
                    invalid_nodes.append({
                        'op_type': op_type,
                        'node_name': node_proto.name or f"{op_type}_{i}",
                        'node_index': i,
                        'reason': 'Converter returned no nodes'
                    })
                    continue
                
                # Add all nodes returned by converter (may be multiple for multi-node conversions)
                for tir_node in tir_nodes:
                    # Validate TIR node before adding
                    # Allow FullNode to have no inputs (it creates constants)
                    if not tir_node.inputs and tir_node.op_type != "Full":
                        logger.warning(
                            f"Skipping TIR node '{tir_node.name}' from {op_type}: No inputs"
                        )
                        continue
                    
                    if not tir_node.outputs:
                        logger.warning(
                            f"Skipping TIR node '{tir_node.name}' from {op_type}: No outputs"
                        )
                        continue
                    
                    # Ensure node name is unique and properly sanitized
                    original_name = tir_node.name
                    tir_node.name = ensure_unique_name(
                        sanitize_name(tir_node.name),
                        self._node_names
                    )
                    self._node_names.add(tir_node.name)
                    
                    if original_name != tir_node.name:
                        logger.debug(
                            f"Renamed node from '{original_name}' to '{tir_node.name}' "
                            f"for uniqueness/sanitization"
                        )
                    
                    # Store original outputs before sanitization (for debug/comparison)
                    tir_node.original_outputs = tir_node.outputs.copy()
                    
                    # Sanitize output names (generate clean Python variable names)
                    # Batch process: generate all sanitized names first, then update
                    sanitized_outputs = []
                    for original_output in tir_node.outputs:
                        # Use get() to avoid repeated lookups
                        clean_name = self._output_name_map.get(original_output)
                        if clean_name is None:
                            # Generate clean name for this output (based on op_type, ignoring original name)
                            clean_name = self._generate_clean_output_name(tir_node.op_type)
                            self._output_name_map[original_output] = clean_name
                            tir_graph.original_to_sanitized[original_output] = clean_name
                            tir_graph.sanitized_to_original[clean_name] = original_output
                        sanitized_outputs.append(clean_name)
                    
                    # Update node outputs to use sanitized names
                    tir_node.outputs = sanitized_outputs
                    
                    # Update node inputs to use sanitized names (batch process with list comprehension)
                    # More efficient: single pass with get() lookups
                    sanitized_inputs = [
                        self._output_name_map.get(original_input, original_input)
                        for original_input in tir_node.inputs
                    ]
                    tir_node.inputs = sanitized_inputs
                    
                    # Set source layer information from original ONNX node name
                    # This is used for debugging and tracking in generated Forge modules
                    if node_proto.name:
                        tir_node.src_layer = node_proto.name
                    
                    tir_graph.add_node(tir_node)
                    
                    # Store ONNX node proto for debug mode
                    if self.debug:
                        # For multi-node conversions, map all TIR nodes to the original ONNX node
                        tir_graph.node_proto_map[tir_node.name] = node_proto
                        # Build reverse mapping: ONNX node name -> list of TIR node names
                        # Only map nodes that have names (required for ONNX Runtime validation)
                        if node_proto.name:
                            tir_graph.onnx_node_to_tir_nodes[node_proto.name].append(tir_node.name)
                        
            except ValueError as e:
                # Re-raise ValueError exceptions (e.g., unsupported features, missing required attributes)
                # These are intentional errors that should propagate to the caller
                raise
            except Exception as e:
                # Graceful handling: Log error and continue with other nodes
                logger.error(
                    f"Failed to convert {op_type} node '{node_proto.name or f'{op_type}_{i}'}' "
                    f"at index {i}: {str(e)}"
                )
                invalid_nodes.append({
                    'op_type': op_type,
                    'node_name': node_proto.name or f"{op_type}_{i}",
                    'node_index': i,
                    'reason': f'Conversion failed: {str(e)}'
                })
                continue
        
        # Report invalid nodes if any
        if invalid_nodes:
            logger.warning(
                f"Encountered {len(invalid_nodes)} node(s) that failed validation or conversion:\n" +
                "\n".join([f"  - {node['op_type']} (node: {node['node_name']}, index: {node['node_index']}): {node['reason']}"
                          for node in invalid_nodes])
            )

        # 8. Process Graph Outputs
        # Use utility function to get output names (original ONNX names)
        original_outputs = get_outputs_names(graph_proto)
        tir_graph.original_outputs = original_outputs
        
        # Sanitize output names (for code generation) - optimized with list comprehension
        # Output names should already be in _output_name_map from node processing
        sanitized_outputs = []
        for original_output in original_outputs:
            clean_name = self._output_name_map.get(original_output)
            if clean_name is None:
                # Fallback: sanitize if not already mapped (shouldn't happen normally)
                logger.warning(f"Output '{original_output}' not found in name mapping, sanitizing on-the-fly")
                clean_name = ensure_unique_name(
                    sanitize_name(original_output) or f"output_{len(sanitized_outputs)}",
                    self._used_output_names
                )
                self._output_name_map[original_output] = clean_name
                self._used_output_names.add(clean_name)
                tir_graph.original_to_sanitized[original_output] = clean_name
                tir_graph.sanitized_to_original[clean_name] = original_output
            sanitized_outputs.append(clean_name)
        
        tir_graph.outputs = sanitized_outputs

        # 9. Compute activation dependencies for memory management
        tir_graph.compute_activation_dependencies()
        
        logger.info("Transpilation completed successfully.")
        
        # Debug mode is already set in TIRGraph constructor

        return tir_graph

