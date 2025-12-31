# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX transpiler engine for converting ONNX models to Forge graphs.

This module provides the main ONNXToForgeTranspiler class, which handles the complete
conversion pipeline from ONNX models to TIRGraph representation. It manages model validation,
shape inference, initializer processing, node conversion, and graph construction.
"""
import onnx
from onnx import numpy_helper, shape_inference
import torch
from loguru import logger
from typing import Dict, Any
from collections import OrderedDict

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.frontends.onnx.utils.attributes import extract_attributes
from forge.transpiler.frontends.onnx.utils.onnx_graph import (
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
    torch_dtype_to_onnx_dtype,
)
from forge.transpiler.frontends.onnx.converters.pad import PadConverter
from forge.transpiler.frontends.onnx.converters.split import SplitConverter
from forge.transpiler.frontends.onnx.converters.squeeze import SqueezeConverter
from forge.transpiler.frontends.onnx.converters.reshape import ReshapeConverter
from forge.transpiler.frontends.onnx.converters.unsqueeze import UnsqueezeConverter
from forge.transpiler.frontends.onnx.converters.concat import ConcatConverter
from forge.transpiler.frontends.onnx.converters.clip import ClipConverter
from forge.transpiler.frontends.onnx.converters.conv import ConvConverter
from forge.transpiler.frontends.onnx.converters.arithmetic import BinaryArithmeticConverter
from forge.transpiler.frontends.onnx.converters.gemm import GemmConverter
from forge.transpiler.frontends.onnx.converters.activation import (
    ReluConverter,
    SigmoidConverter,
    TanhConverter,
    SoftmaxConverter,
    LogSoftmaxConverter,
    LeakyReluConverter,
    DropoutConverter,
)
from forge.transpiler.frontends.onnx.converters.reduction import (
    ReduceSumConverter,
    ReduceMeanConverter,
    ReduceMaxConverter,
)
from forge.transpiler.frontends.onnx.converters.pooling import MaxPoolConverter, AveragePoolConverter
from forge.transpiler.frontends.onnx.converters.shape import TransposeConverter, CastConverter, FlattenConverter
from forge.transpiler.frontends.onnx.converters.constant import ConstantConverter
from forge.transpiler.frontends.onnx.converters.converter_result import ConverterResult, is_constant_result
from forge.transpiler.frontends.onnx.utils.naming import sanitize_name, ensure_unique_name
from forge.transpiler.frontends.onnx.utils.exceptions import UnsupportedOperationError, ONNXModelValidationError
from forge.transpiler.core.exceptions import ConversionError


class ONNXToForgeTranspiler:
    """
    Main transpiler class for converting ONNX models to Forge graphs.

    This class orchestrates the complete conversion process from ONNX ModelProto to TIRGraph,
    including model validation, shape inference, parameter/constant distinction, and node
    conversion using opset-specific converters.

    Attributes:
        debug: Whether debug mode is enabled (compares outputs with ONNXRuntime)
        freeze_params: If True, all initializers become constants (non-trainable)
        validate_model: Whether to validate ONNX model before conversion
        onnx_model: Original ONNX model (stored for debug mode)
        opset: ONNX opset version extracted from model
        _op_converters: Dictionary mapping ONNX op types to converter methods
    """

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
        self.onnx_model = None

        # Check if onnxruntime is available for debug mode
        # Debug mode requires onnxruntime to compare TIR outputs with ONNX Runtime outputs
        if debug:
            try:
                import onnxruntime
            except ImportError:
                logger.warning("onnxruntime not available. Debug mode requires: pip install onnxruntime")
                self.debug = False

        # Initialize opset version (will be extracted from model during transpile)
        self.opset = 1
        # Converter map will be built based on opset version during transpile
        self._op_converters = {}
        # Initialize state tracking for name generation and uniqueness
        self._reset_transpilation_state()

    def _reset_transpilation_state(self) -> None:
        """
        Reset all state variables for a new transpilation.

        This method should be called at the start of each transpile() call
        to ensure clean state. It initializes node name tracking, operation
        type counters, and sanitized name tracking.

        Note: Name mappings (original <-> sanitized) are stored in TIRGraph,
        not in the engine, to avoid redundancy.
        """
        self._unique_node_names: set = set()
        self._generated_sanitized_names: set = set()
        self._op_type_name_counters: Dict[str, int] = {}

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

        # Initialize counter for this operation type if not exists
        # Counter ensures unique names: conv2d_0, conv2d_1, etc.
        if op_type not in self._op_type_name_counters:
            self._op_type_name_counters[op_type] = 0

        # Generate base name using operation type and counter (e.g., "conv2d_0")
        base_name = generate_clean_variable_name(op_type, self._op_type_name_counters[op_type])
        # Ensure uniqueness across all generated names (may append suffix if collision)
        clean_name = ensure_unique_name(base_name, self._generated_sanitized_names)
        # Track this name to prevent future collisions
        self._generated_sanitized_names.add(clean_name)
        # Increment counter for next node of this type
        self._op_type_name_counters[op_type] += 1

        return clean_name

    def _handle_constant_result(self, const_result, tir_graph, op_type, node_proto, node_index, invalid_nodes):
        """
        Handle a ConstantResult from a converter.

        Constant nodes don't create TIR nodes, they just store values in the graph's
        constants dictionary.

        Args:
            const_result: ConstantResult instance containing output name and value
            tir_graph: TIRGraph to update
            op_type: Operation type (for logging)
            node_proto: ONNX node proto (for logging)
            node_index: Index of the node (for logging)
            invalid_nodes: List to append invalid nodes to (unused for constants)
        """
        tir_graph.constants[const_result.output_name] = const_result.value
        logger.debug(f"Stored constant '{const_result.output_name}' from Constant node")

    def _handle_tir_nodes_result(self, tir_nodes, tir_graph, op_type, node_proto, node_index, invalid_nodes):
        """
        Handle a list of TIR nodes from a converter.

        Validates nodes, sanitizes names, updates name mappings, and adds nodes to the graph.
        Supports multi-output operations and nodes that produce multiple TIR nodes.

        Args:
            tir_nodes: List of TIR nodes returned by converter
            tir_graph: TIRGraph to update (contains name mappings)
            op_type: Operation type (for logging)
            node_proto: ONNX node proto (for debug mode and source layer tracking)
            node_index: Index of the node (for logging)
            invalid_nodes: List to append invalid nodes to
        """
        # Validate that converter returned at least one node
        if not tir_nodes or len(tir_nodes) == 0:
            logger.warning(
                f"Skipping {op_type} node '{node_proto.name or f'{op_type}_{node_index}'}' "
                f"at index {node_index}: Converter returned no nodes"
            )
            invalid_nodes.append(
                {
                    "op_type": op_type,
                    "node_name": node_proto.name or f"{op_type}_{node_index}",
                    "node_index": node_index,
                    "reason": "Converter returned no nodes",
                }
            )
            return

        # Process each TIR node returned by the converter
        # Some converters may return multiple nodes (e.g., Gemm decomposes into multiple ops)
        for tir_node in tir_nodes:
            # Validate node has inputs (FullNode is exception - creates constant tensor)
            if not tir_node.inputs and tir_node.op_type != "Full":
                logger.warning(f"Skipping TIR node '{tir_node.name}' from {op_type}: No inputs")
                continue

            # Validate node has outputs
            if not tir_node.outputs:
                logger.warning(f"Skipping TIR node '{tir_node.name}' from {op_type}: No outputs")
                continue

            # Sanitize and ensure uniqueness of node name
            # Node names must be valid Python identifiers and unique within the graph
            original_name = tir_node.name
            tir_node.name = ensure_unique_name(sanitize_name(tir_node.name), self._unique_node_names)
            self._unique_node_names.add(tir_node.name)

            if original_name != tir_node.name:
                logger.debug(
                    f"Renamed node from '{original_name}' to '{tir_node.name}' " f"for uniqueness/sanitization"
                )

            # Store original output names before sanitization
            # This is used for debug mode to map back to ONNX outputs
            tir_node.original_outputs = list(tir_node.outputs.keys())

            # Sanitize output names: convert ONNX names to clean Python variable names
            # Output names are generated based on operation type, not original ONNX names
            # This ensures readable, consistent variable names in generated code
            sanitized_outputs = OrderedDict()
            for original_output, tensor_info in tir_node.outputs.items():
                # Check if this output was already mapped (may happen with multi-output ops)
                clean_name = tir_graph.original_to_sanitized.get(original_output)
                if clean_name is None:
                    # Generate new clean name based on operation type (e.g., "conv2d_0")
                    # This ignores the original ONNX name for consistency
                    clean_name = self._generate_clean_output_name(tir_node.op_type)
                    # Store bidirectional mapping in TIRGraph (single source of truth)
                    tir_graph.original_to_sanitized[original_output] = clean_name
                    tir_graph.sanitized_to_original[clean_name] = original_output
                # Update TensorInfo name to match sanitized name
                if tensor_info.name != clean_name:
                    tensor_info.name = clean_name
                sanitized_outputs[clean_name] = tensor_info

            # Update node outputs to use sanitized names
            tir_node.outputs = sanitized_outputs

            # Sanitize input names: map ONNX names to sanitized names
            # Parameters and constants keep their original names (not in mappings),
            # so .get() falls back to original name for them
            sanitized_inputs = OrderedDict()
            for original_input, tensor_info in tir_node.inputs.items():
                # Look up sanitized name, or use original if not found (params/constants)
                clean_input = tir_graph.original_to_sanitized.get(original_input, original_input)
                # Update TensorInfo name to match sanitized name
                if tensor_info.name != clean_input:
                    tensor_info.name = clean_input
                sanitized_inputs[clean_input] = tensor_info
            tir_node.inputs = sanitized_inputs

            # Store original ONNX node name as source layer for debugging/tracing
            if node_proto.name:
                tir_node.src_layer = node_proto.name

            # Add node to graph (this also validates graph structure)
            tir_graph.add_node(tir_node)

            # Store mapping for debug mode: allows comparing TIR outputs with ONNX Runtime
            if self.debug:
                # Map TIR node name to original ONNX node proto
                tir_graph.frontend_node_map[tir_node.name] = node_proto
                # Track which TIR nodes came from which ONNX node (one-to-many possible)
                if node_proto.name:
                    tir_graph.frontend_node_to_tir_nodes[node_proto.name].append(tir_node.name)

    def _is_constant(self, name: str, tensor: torch.Tensor) -> bool:
        """
        Determine if an initializer should be treated as a constant (non-trainable)
        vs a parameter (trainable).

        Uses heuristics based on TVM's approach:
        - Constants: name contains "constant", or scalar shape, or
          (not weight/bias and int/bool dtype)
        - Parameters: weights, biases, and other trainable tensors

        Args:
            name: Name of the initializer
            tensor: The tensor value

        Returns:
            True if constant, False if parameter
        """
        name_lower = name.lower()

        # Heuristic 1: Explicit constant naming
        if "constant" in name_lower:
            return True

        # Heuristic 2: Scalar tensors are typically constants (not trainable)
        if len(tensor.shape) == 0:
            return True

        # Heuristic 3: Non-weight/bias tensors with integer/bool dtype are constants
        # Integer and boolean tensors are typically used for indices, masks, etc.
        # and are not trainable parameters
        if "weight" not in name_lower and "bias" not in name_lower:
            dtype_str = str(tensor.dtype).lower()
            if "int" in dtype_str or "bool" in dtype_str:
                return True

        # Default: treat as parameter (trainable)
        return False

    def _get_tensor_info(self, value_info_map, name):
        """
        Retrieve shape and dtype from value_info_map and wrap in a TensorInfo object.

        Args:
            value_info_map: Dictionary mapping tensor names to ONNX ValueInfoProto
            name: Tensor name to look up

        Returns:
            TensorInfo object with shape and dtype information.
            Returns TensorInfo with None shape and UNDEFINED dtype if name not found.
        """
        # Return default TensorInfo if tensor not found in value_info_map
        # This can happen for intermediate values not explicitly tracked
        if name not in value_info_map:
            return TensorInfo(name, None, onnx.TensorProto.UNDEFINED)

        # Extract tensor type information from ONNX ValueInfoProto
        vi = value_info_map[name]
        tensor_type = vi.type.tensor_type

        # Extract data type (element type)
        onnx_dtype = tensor_type.elem_type
        shape = None

        # Extract shape information if available
        # ONNX shapes can contain:
        # - dim_value: Fixed dimension size (integer)
        # - dim_param: Dynamic dimension (symbolic name, e.g., "batch_size")
        # - Neither: Unknown dimension (None)
        if tensor_type.HasField("shape"):
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    # Fixed dimension size
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    # Dynamic dimension (symbolic name)
                    shape.append(dim.dim_param)
                else:
                    # Unknown dimension
                    shape.append(None)
            shape = tuple(shape)

        return TensorInfo(name, shape, onnx_dtype)

    def _validate_onnx_model(self, onnx_model: onnx.ModelProto) -> None:
        """
        Validate the ONNX model using ONNX checker.

        This method performs comprehensive validation of the ONNX model structure,
        including schema validation, type checking, and graph consistency.

        Args:
            onnx_model: ONNX ModelProto to validate

        Raises:
            ONNXModelValidationError: If model validation fails for any reason.
                This includes:
                - Schema validation errors (invalid node attributes, types, etc.)
                - Graph structure errors (invalid connections, cycles, etc.)
                - Type inference errors (incompatible types, missing shapes, etc.)
                - Any other unexpected errors during validation
        """
        try:
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")

        except onnx.checker.ValidationError as e:
            error_msg = (
                f"ONNX model validation failed: {str(e)}\n"
                f"This indicates the model does not conform to ONNX specification.\n"
                f"Common causes:\n"
                f"  - Invalid node attributes or types\n"
                f"  - Graph structure inconsistencies\n"
                f"  - Type inference failures\n"
                f"  - Missing required fields\n"
                f"\n"
                f"Please verify your ONNX model is valid using: onnx.checker.check_model(model)"
            )

            model_info = self._extract_model_info(onnx_model)

            logger.error(error_msg)
            raise ONNXModelValidationError(error_msg, validation_error=e, model_info=model_info) from e

        except Exception as e:
            error_msg = (
                f"ONNX model validation encountered an unexpected error: {str(e)}\n"
                f"This may indicate:\n"
                f"  - Corrupted or malformed ONNX model file\n"
                f"  - Missing ONNX dependencies or version incompatibility\n"
                f"  - Internal ONNX checker error\n"
                f"\n"
                f"Model validation is required when validate_model=True. "
                f"Please fix the model or disable validation (not recommended)."
            )

            model_info = self._extract_model_info(onnx_model)

            logger.error(error_msg, exc_info=True)
            raise ONNXModelValidationError(error_msg, validation_error=e, model_info=model_info) from e

    def _extract_model_info(self, onnx_model: onnx.ModelProto) -> Dict[str, Any]:
        """
        Extract metadata from ONNX model for error reporting.

        Args:
            onnx_model: ONNX ModelProto

        Returns:
            Dictionary containing model metadata
        """
        try:
            graph = onnx_model.graph

            opset = None
            try:
                opset = self._get_opset_version(onnx_model)
            except Exception:
                if onnx_model.opset_import:
                    opset = onnx_model.opset_import[0].version if onnx_model.opset_import else None

            return {
                "opset": opset,
                "inputs": len(graph.input) if graph.input else 0,
                "outputs": len(graph.output) if graph.output else 0,
                "nodes": len(graph.node) if graph.node else 0,
                "initializers": len(graph.initializer) if graph.initializer else 0,
                "ir_version": getattr(onnx_model, "ir_version", None),
            }
        except Exception:
            return {}

    def _get_opset_version(self, onnx_model: onnx.ModelProto) -> int:
        """
        Extract opset version from ONNX model.

        Returns:
            Opset version (defaults to 1 if not found)
        """
        try:
            opset_in_model = 1
            if onnx_model.opset_import:
                for opset_identifier in onnx_model.opset_import:
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

        All operations use versioned converter classes following TVM pattern.
        Each converter is bound to the specific opset version.

        Args:
            opset: ONNX opset version

        Returns:
            Dictionary mapping ONNX operation types to converter functions
        """
        convert_map = {
            # Arithmetic operations
            "Add": BinaryArithmeticConverter.get_converter(opset),
            "Sub": BinaryArithmeticConverter.get_converter(opset),
            "Mul": BinaryArithmeticConverter.get_converter(opset),
            "Div": BinaryArithmeticConverter.get_converter(opset),
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
            "Constant": ConstantConverter.get_converter(opset),
        }

        return convert_map

    def transpile(self, onnx_model: onnx.ModelProto) -> TIRGraph:
        """
        Transpile an ONNX model to a TIR graph.

        This is the main entry point for converting ONNX models to TIRGraph.
        The process includes:
        1. Model validation (if enabled)
        2. Shape inference
        3. Initializer processing (parameters vs constants)
        4. Node conversion using opset-specific converters
        5. Graph construction and name sanitization

        Args:
            onnx_model: ONNX ModelProto to transpile

        Returns:
            TIRGraph representing the converted model

        Raises:
            ONNXModelValidationError: If model validation fails
            UnsupportedOperationError: If unsupported operations are found
            ConversionError: If node conversion fails
        """
        logger.info("Starting Transpilation with Shape Inference...")

        # Store original model for debug mode (needed for ONNX Runtime comparison)
        self.onnx_model = onnx_model

        # Step 1: Validate ONNX model structure and schema (if enabled)
        # This catches errors early before attempting conversion
        if self.validate_model:
            self._validate_onnx_model(onnx_model)

        # Step 2: Extract opset version from model
        # Opset version determines which converter logic to use for each operation
        self.opset = self._get_opset_version(onnx_model)
        logger.info(f"ONNX Opset Version: {self.opset}")

        # Step 3: Build converter map for this opset version
        # Each converter is bound to the specific opset version
        self._op_converters = self._build_convert_map(self.opset)
        # Reset state tracking for name generation and uniqueness
        self._reset_transpilation_state()

        # Step 4: Run shape inference to determine tensor shapes throughout the graph
        # Shape inference fills in missing shape information and validates shape compatibility
        try:
            inferred_model = shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            # If shape inference fails, proceed with original model
            # Some models may have incomplete shape information, which is acceptable
            logger.error(f"Shape inference failed: {e}. Proceeding without inferred shapes.")
            inferred_model = onnx_model

        # Step 5: Remove initializers from input list
        # ONNX models may list initializers as inputs, but they're actually graph parameters/constants
        # This cleanup ensures inputs only contain actual model inputs
        inferred_model = remove_initializers_from_input(inferred_model)

        # Extract graph proto for processing
        graph_proto = inferred_model.graph
        self.graph_proto = graph_proto

        # Step 6: Create TIRGraph to hold the converted graph
        # Store original model in graph if debug mode enabled (for ONNX Runtime comparison)
        tir_graph = TIRGraph(
            name=graph_proto.name,
            framework="onnx",
            frontend_model=onnx_model if self.debug else None,
            debug_mode=self.debug,
        )

        # Step 7: Build value_info_map: maps tensor names to their shape/dtype information
        # This includes intermediate values (value_info), inputs, and outputs
        # Used later to provide shape/dtype info to converters
        value_info_map = {vi.name: vi for vi in graph_proto.value_info}
        value_info_map.update({vi.name: vi for vi in graph_proto.input})
        value_info_map.update({vi.name: vi for vi in graph_proto.output})

        # Step 8: Process initializers (weights, biases, constants)
        # Initializers are pre-computed tensor values stored in the ONNX model
        # They can be either parameters (trainable) or constants (non-trainable)
        for initializer in graph_proto.initializer:
            # Convert ONNX tensor to NumPy array, then to PyTorch tensor
            np_array = numpy_helper.to_array(initializer)
            onnx_dtype = initializer.data_type
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
            torch_tensor = torch.from_numpy(np_array).to(torch_dtype)

            # Classify as constant or parameter based on freeze_params flag or heuristics
            if self.freeze_params:
                # If freeze_params=True, treat all initializers as constants (non-trainable)
                # This is useful for inference-only models
                tir_graph.constants[initializer.name] = torch_tensor
            else:
                # Use heuristics to distinguish parameters from constants
                # Parameters go to params dict (trainable), constants go to constants dict (non-trainable)
                if self._is_constant(initializer.name, torch_tensor):
                    tir_graph.constants[initializer.name] = torch_tensor
                else:
                    tir_graph.params[initializer.name] = torch_tensor

        # Step 9: Process and sanitize input names
        # Inputs are the model's entry points (user-provided data)
        original_inputs = get_inputs_names(graph_proto)
        tir_graph.original_inputs = original_inputs

        # Sanitize input names to valid Python identifiers
        # Input names are preserved more closely than intermediate outputs (user may reference them)
        sanitized_inputs = []
        for original_input in original_inputs:
            # Check if already sanitized (shouldn't happen, but safe check)
            clean_name = tir_graph.original_to_sanitized.get(original_input)
            if clean_name is None:
                # Sanitize original name, or generate default if sanitization fails
                base_name = sanitize_name(original_input) or f"input_{len(sanitized_inputs)}"
                # Ensure uniqueness across all names (inputs, outputs, nodes)
                all_used_names = set(tir_graph.sanitized_to_original.keys()) | self._generated_sanitized_names
                clean_name = ensure_unique_name(base_name, all_used_names)
                # Store bidirectional mapping
                tir_graph.original_to_sanitized[original_input] = clean_name
                tir_graph.sanitized_to_original[clean_name] = original_input
                self._generated_sanitized_names.add(clean_name)
            sanitized_inputs.append(clean_name)

        tir_graph.inputs = sanitized_inputs

        # Step 10: Pre-scan nodes to check for unsupported operations
        # This provides better error messages by collecting all unsupported ops before conversion
        unsupported_ops = []

        for i, node_proto in enumerate(graph_proto.node):
            op_type = node_proto.op_type
            converter_method = self._op_converters.get(op_type, None)

            # Check if converter exists for this operation type
            if converter_method is None:
                node_name = node_proto.name if node_proto.name else f"{op_type}_{i}"

                # Collect input tensor information for error reporting
                input_tensors = OrderedDict()
                for name in node_proto.input:
                    input_tensors[name] = self._get_tensor_info(value_info_map, name)

                # Extract attributes for error reporting
                attrs = extract_attributes(node_proto)

                # Format input details for error message
                input_details = []
                for input_name in node_proto.input:
                    if input_name in input_tensors:
                        tensor_info = input_tensors[input_name]
                        shape_str = str(tensor_info.shape) if tensor_info.shape else "unknown"
                        dtype_str = str(tensor_info.torch_dtype) if tensor_info.torch_dtype else "unknown"
                        input_details.append(f"{input_name}: shape={shape_str}, dtype={dtype_str}")
                    else:
                        input_details.append(f"{input_name}: unknown")

                # Record unsupported operation details
                unsupported_ops.append(
                    {
                        "op_type": op_type,
                        "node_name": node_name,
                        "node_index": i,
                        "input_details": input_details,
                        "attrs": attrs,
                    }
                )

        # If unsupported operations found, raise error with detailed information
        if unsupported_ops:
            unsupported_types = sorted(set([op["op_type"] for op in unsupported_ops]))
            error_msg = (
                f"Found {len(unsupported_ops)} unsupported ONNX operation(s) in the model:\n"
                f"  Unsupported operation types: {', '.join(unsupported_types)}\n"
                f"  Total unsupported nodes: {len(unsupported_ops)}\n"
                f"  Details:\n"
            )
            for op in unsupported_ops:
                attrs_str = ", ".join([f"{k}={v}" for k, v in op["attrs"].items()]) if op["attrs"] else "none"
                error_msg += (
                    f"    - {op['op_type']} (node: {op['node_name']}, index: {op['node_index']})\n"
                    f"      Inputs: {', '.join(op['input_details'])}\n"
                    f"      Attributes: {attrs_str}\n"
                )

            logger.error(error_msg)
            raise UnsupportedOperationError(error_msg, unsupported_ops)
        else:
            logger.info("All ONNX operations are supported. Proceeding with conversion.")

        # Step 11: Convert each ONNX node to TIR nodes
        # Process nodes in order (ONNX graphs are typically topologically sorted)
        invalid_nodes = []

        for i, node_proto in enumerate(graph_proto.node):
            op_type = node_proto.op_type

            # Build input tensor information dictionary
            # Converters need shape/dtype info for validation and code generation
            input_tensors = OrderedDict()
            for name in node_proto.input:
                tensor_info = self._get_tensor_info(value_info_map, name)
                # If shape/dtype not found in value_info_map, try to get from params/constants
                # This handles cases where shape inference didn't populate value_info
                if tensor_info.shape is None and tensor_info.onnx_dtype == onnx.TensorProto.UNDEFINED:
                    if name in tir_graph.params:
                        # Extract shape/dtype from parameter tensor
                        param_tensor = tir_graph.params[name]
                        param_shape = tuple(param_tensor.shape) if param_tensor.shape else None
                        param_onnx_dtype = torch_dtype_to_onnx_dtype(param_tensor.dtype)
                        tensor_info = TensorInfo(name, param_shape, param_onnx_dtype)
                    elif name in tir_graph.constants:
                        # Extract shape/dtype from constant tensor
                        const_tensor = tir_graph.constants[name]
                        const_shape = tuple(const_tensor.shape) if const_tensor.shape else None
                        const_onnx_dtype = torch_dtype_to_onnx_dtype(const_tensor.dtype)
                        tensor_info = TensorInfo(name, const_shape, const_onnx_dtype)
                input_tensors[name] = tensor_info

            # Build output tensor information dictionary
            # Used by converters to determine output shapes/dtypes
            output_tensors = OrderedDict()
            for name in node_proto.output:
                output_tensors[name] = self._get_tensor_info(value_info_map, name)

            # Extract node attributes (operation-specific parameters)
            attrs = extract_attributes(node_proto)

            # Validate node has outputs (required for graph construction)
            if len(node_proto.output) == 0:
                logger.warning(
                    f"Skipping {op_type} node '{node_proto.name or f'{op_type}_{i}'}' "
                    f"at index {i}: No outputs provided"
                )
                invalid_nodes.append(
                    {
                        "op_type": op_type,
                        "node_name": node_proto.name or f"{op_type}_{i}",
                        "node_index": i,
                        "reason": "No outputs provided",
                    }
                )
                continue

            # Get converter for this operation type (already validated in pre-scan)
            converter_method = self._op_converters[op_type]

            # Call converter to convert ONNX node to TIR nodes
            try:
                converter_result: ConverterResult = converter_method(
                    node_proto, input_tensors, output_tensors, attrs, i, self.graph_proto
                )

                # Handle converter result based on type
                # Converters can return either ConstantResult or List[TIRNode]
                if is_constant_result(converter_result):
                    # Constant nodes: store value directly in graph.constants
                    self._handle_constant_result(converter_result, tir_graph, op_type, node_proto, i, invalid_nodes)
                else:
                    # Normal operations: add TIR nodes to graph with name sanitization
                    self._handle_tir_nodes_result(converter_result, tir_graph, op_type, node_proto, i, invalid_nodes)

            except ConversionError:
                # Re-raise conversion errors as-is (already properly formatted)
                raise
            except ValueError as e:
                # Wrap ValueError as ConversionError with context
                node_name = node_proto.name or f"{op_type}_{i}"
                error_msg = f"Failed to convert {op_type} node '{node_name}' " f"at index {i}: {str(e)}"
                logger.error(error_msg)
                raise ConversionError(op_type, node_name, str(e), node_index=i) from e
            except Exception as e:
                # Wrap unexpected errors as ConversionError with full context
                node_name = node_proto.name or f"{op_type}_{i}"
                error_msg = f"Unexpected error converting {op_type} node '{node_name}' " f"at index {i}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ConversionError(op_type, node_name, f"Unexpected error: {str(e)}", node_index=i) from e

        # Step 12: Report any invalid nodes that were skipped
        # Invalid nodes are those that failed validation but didn't cause conversion to fail
        if invalid_nodes:
            logger.warning(
                f"Encountered {len(invalid_nodes)} node(s) that failed validation or conversion:\n"
                + "\n".join(
                    [
                        f"  - {node['op_type']} (node: {node['node_name']}, index: {node['node_index']}): {node['reason']}"
                        for node in invalid_nodes
                    ]
                )
            )

        # Step 13: Process and sanitize output names
        # Outputs are the model's exit points (results returned to user)
        original_outputs = get_outputs_names(graph_proto)
        tir_graph.original_outputs = original_outputs

        # Sanitize output names to valid Python identifiers
        # Output names should already be in mappings from node conversion, but handle edge cases
        sanitized_outputs = []
        for original_output in original_outputs:
            # Look up sanitized name (should exist from node conversion)
            clean_name = tir_graph.original_to_sanitized.get(original_output)
            if clean_name is None:
                # Edge case: output not found in mapping (shouldn't happen normally)
                # This can occur if output comes from a constant or parameter directly
                logger.warning(f"Output '{original_output}' not found in name mapping, sanitizing on-the-fly")
                base_name = sanitize_name(original_output) or f"output_{len(sanitized_outputs)}"
                # Ensure uniqueness across all names
                all_used_names = set(tir_graph.sanitized_to_original.keys()) | self._generated_sanitized_names
                clean_name = ensure_unique_name(base_name, all_used_names)
                # Store bidirectional mapping
                tir_graph.original_to_sanitized[original_output] = clean_name
                tir_graph.sanitized_to_original[clean_name] = original_output
                self._generated_sanitized_names.add(clean_name)
            sanitized_outputs.append(clean_name)

        tir_graph.outputs = sanitized_outputs

        # Step 14: Compute activation dependencies for memory management
        # This determines which activations are still needed at each point in the graph
        # Used for garbage collection and memory optimization in code generation
        tir_graph.compute_activation_dependencies()

        logger.info("Transpilation completed successfully.")

        return tir_graph
