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
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.core.graph import TIRGraph
from forge.transpiler.frontends.onnx.utils.attributes import extract_attributes
from forge.transpiler.frontends.onnx.utils.conversion_logger import TranspilerLogger
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
from forge.transpiler.frontends.onnx.converters.expand import ExpandConverter
from forge.transpiler.frontends.onnx.converters.concat import ConcatConverter
from forge.transpiler.frontends.onnx.converters.clip import ClipConverter
from forge.transpiler.frontends.onnx.converters.conv import ConvConverter
from forge.transpiler.frontends.onnx.converters.elementwise_binary import (
    BinaryOpConverter,
    MatMulConverter,
    PowConverter,
)
from forge.transpiler.frontends.onnx.converters.elementwise_unary import UnaryOpConverter
from forge.transpiler.frontends.onnx.converters.gemm import GemmConverter
from forge.transpiler.frontends.onnx.converters.gather import GatherConverter
from forge.transpiler.frontends.onnx.converters.slice import SliceConverter
from forge.transpiler.frontends.onnx.converters.constantofshape import ConstantOfShapeConverter
from forge.transpiler.frontends.onnx.converters.shape import ShapeConverter
from forge.transpiler.frontends.onnx.converters.activation import (
    ReluConverter,
    SigmoidConverter,
    TanhConverter,
    SoftmaxConverter,
    LogSoftmaxConverter,
    LeakyReluConverter,
    DropoutConverter,
    SqrtConverter,
)
from forge.transpiler.frontends.onnx.converters.reduction import (
    ReduceSumConverter,
    ReduceMeanConverter,
    ReduceMaxConverter,
    ArgMaxConverter,
)
from forge.transpiler.frontends.onnx.converters.pooling import (
    MaxPoolConverter,
    AveragePoolConverter,
    GlobalAveragePoolConverter,
)
from forge.transpiler.frontends.onnx.converters.shape import TransposeConverter, CastConverter, FlattenConverter
from forge.transpiler.frontends.onnx.converters.constant import ConstantConverter
from forge.transpiler.frontends.onnx.converters.condition import WhereConverter
from forge.transpiler.frontends.onnx.converters.layernorm import LayerNormalizationConverter
from forge.transpiler.frontends.onnx.converters.trilu import TriluConverter
from forge.transpiler.frontends.onnx.converters.converter_result import ConverterResult, is_constant_result
from forge.transpiler.frontends.onnx.utils.naming import sanitize_name, ensure_unique_name
from forge.transpiler.utils.exceptions import (
    ConversionError,
    UnsupportedOperationError,
    ONNXModelValidationError,
)


class ONNXToForgeTranspiler:
    """
    Main transpiler class for converting ONNX models to Forge graphs.

    This class orchestrates the complete conversion process from ONNX ModelProto to TIRGraph,
    including model validation, shape inference, parameter/constant distinction, and node
    conversion using opset-specific converters.

    Attributes:
        debug: Whether debug mode is enabled.  When True, every TIR node is
            validated against ONNX Runtime during ``TIRGraph.run()``.  For each
            ONNX node the transpiler maps to one or more TIR nodes, the runner
            executes both the ONNX reference (via ``onnxruntime``) and the TIR
            subgraph and compares outputs element-wise for shape, dtype, and
            numerical values.  Any mismatch raises a ``DebugValidationError``
            immediately, pinpointing the exact node responsible.  Requires the
            ``onnxruntime`` package.
        freeze_params: If True, all initializers become constants (non-trainable)
        validate_model: Whether to validate ONNX model before conversion
        onnx_model: Original ONNX model (stored for debug mode)
        opset: ONNX opset version extracted from model
        _op_converters: Dictionary mapping ONNX op types to converter methods
    """

    def __init__(
        self,
        debug: bool = False,
        freeze_params: bool = False,
        validate_model: bool = True,
        resolve_dynamic_shapes: bool = False,
    ):
        """
        Initialize the transpiler.

        Args:
            debug: Enable per-node debug validation against ONNX Runtime.
                When True, every TIR node output is compared against the
                corresponding ONNX Runtime reference during ``TIRGraph.run()``.
                Checks cover shape, dtype, and numerical values (within
                tolerance).  A ``DebugValidationError`` is raised at the first
                mismatch, immediately identifying the faulty node.
                Requires ``onnxruntime``; automatically disabled if unavailable.
            freeze_params: If True, all initializers become constants (non-trainable).
                If False, uses heuristics to distinguish parameters from constants.
            validate_model: If True, validate ONNX model before conversion.
            resolve_dynamic_shapes: If True and *module_inputs* are supplied to
                :meth:`transpile`, execute the model once via ONNX Runtime with
                the provided inputs to replace every dynamic / symbolic dimension
                (e.g. batch size, sequence length) with a real integer value.
                The resolved shapes are written back into the model proto so that
                every subsequent converter receives fully concrete tensor shapes.
                Set to False when the model already has fully static shapes or
                when ONNX Runtime is unavailable.
        """
        self.debug = debug
        self.freeze_params = freeze_params
        self.validate_model = validate_model
        self.resolve_dynamic_shapes = resolve_dynamic_shapes
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
        from forge.transpiler.frontends.onnx.utils.naming import generate_clean_variable_name

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

    def _handle_constant_result(
        self, const_result, tir_graph, op_type, node_proto, node_index, invalid_nodes, log_lines=None
    ):
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
        # Sanitize output name (similar to TIR nodes)
        original_output_name = const_result.output_name
        clean_name = tir_graph.original_to_sanitized.get(original_output_name)
        if clean_name is None:
            # Generate sanitized name if not already mapped
            base_name = sanitize_name(original_output_name) or f"{op_type.lower()}_{node_index}"
            all_used_names = set(tir_graph.sanitized_to_original.keys()) | self._generated_sanitized_names
            clean_name = ensure_unique_name(base_name, all_used_names)
            # Store bidirectional mapping
            tir_graph.original_to_sanitized[original_output_name] = clean_name
            tir_graph.sanitized_to_original[clean_name] = original_output_name
            self._generated_sanitized_names.add(clean_name)

        # Store in computed_constants — these values come from ONNX graph nodes
        # (e.g. Constant op, ConstantOfShape), not from model.graph.initializer.
        # They must be saved to a PT file at code-gen time so that
        # process_framework_parameters() can load and set them at runtime.
        #
        # Normalize 0-d (scalar) tensors to 1-d so that Forge ops always receive
        # at least a 1-d tensor.  forge.op.* functions do not support 0-d tensors.
        value = const_result.value
        if value.ndim == 0:
            value = value.unsqueeze(0)

        tir_graph.computed_constants[clean_name] = value
        if original_output_name != clean_name:
            tir_graph.computed_constants[original_output_name] = value

        _const_log = TranspilerLogger.format_constant_result(original_output_name, clean_name, value)
        if log_lines is not None:
            log_lines.append(_const_log)
        else:
            logger.trace(_const_log)

    def _handle_tir_nodes_result(
        self, tir_nodes, tir_graph, op_type, node_proto, node_index, invalid_nodes, log_lines=None
    ):
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

        # Log how many TIR nodes this ONNX op produces
        _mapped_summary = TranspilerLogger.format_mapped_summary(op_type, tir_nodes)
        if log_lines is not None:
            log_lines.append(_mapped_summary)
        else:
            logger.trace(_mapped_summary)

        # Process each TIR node returned by the converter
        # Some converters may return multiple nodes (e.g., Gemm decomposes into multiple ops)
        _tir_node_idx = 0
        for tir_node in tir_nodes:
            # Validate node has inputs (FullNode and ConstantNode are exceptions - create constant tensors)
            if not tir_node.inputs and tir_node.op_type not in ("Full", "Constant"):
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
                _rename_msg = f"  (renamed: '{original_name}' -> '{tir_node.name}')"
                if log_lines is not None:
                    log_lines.append(_rename_msg)
                else:
                    logger.trace(_rename_msg)

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

            # Every FullNode must become a graph constant — FullNode.forge_op_name is None
            # and can never reach code generation.  Convert all FullNodes unconditionally
            # rather than relying on each converter to set the "should_be_constant" flag,
            # which is fragile and caused ConstantOfShape nodes to survive as live nodes.
            if tir_node.op_type == "Full":
                # Extract the constant value from the FullNode's attrs
                constant_value = tir_node.attrs.get("constant_value")
                if constant_value is None:
                    # If constant_value is not stored, execute the FullNode to get the value
                    constant_value = tir_node.eval({})[list(tir_node.outputs.keys())[0]]

                # Get both the sanitized output name and original output name
                sanitized_output_name = list(tir_node.outputs.keys())[0]
                original_output_name = (
                    tir_node.original_outputs[0] if tir_node.original_outputs else sanitized_output_name
                )

                # FullNode values are computed during transpilation — they are NOT sourced
                # from model.graph.initializer.  Store in computed_constants so that the
                # code generator knows to persist them to a PT file and load them at runtime.
                #
                # Normalize 0-d (scalar) tensors to 1-d: forge.op.* functions require
                # at least a 1-d tensor.
                if constant_value.ndim == 0:
                    constant_value = constant_value.unsqueeze(0)

                tir_graph.computed_constants[sanitized_output_name] = constant_value
                if original_output_name != sanitized_output_name:
                    tir_graph.computed_constants[original_output_name] = constant_value

                _full_const_msg = TranspilerLogger.format_full_const(
                    tir_node, original_output_name, sanitized_output_name, constant_value
                )
                if log_lines is not None:
                    log_lines.append(_full_const_msg)
                else:
                    logger.trace(_full_const_msg)

                # Skip adding this node to the graph - it's now a constant
                continue

            # Build TIR node detail (numbered inputs/outputs)
            _tir_node_idx += 1
            _tir_detail = TranspilerLogger.format_tir_node_detail(tir_node, _tir_node_idx)
            if log_lines is not None:
                log_lines.append(_tir_detail)
            else:
                logger.trace(_tir_detail)

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
                which = dim.WhichOneof("value")
                if which == "dim_value":
                    # Fixed dimension size (may be 0 for empty tensors — treat as concrete)
                    shape.append(dim.dim_value)
                elif which == "dim_param":
                    # Dynamic dimension (symbolic name)
                    shape.append(dim.dim_param)
                else:
                    # Unknown dimension
                    shape.append(None)
            shape = tuple(shape)

        return TensorInfo(name, shape, onnx_dtype)

    def _resolve_model_shapes_inplace(
        self,
        inferred_model: onnx.ModelProto,
        module_inputs,
    ) -> onnx.ModelProto:
        """
        Resolve all unknown / symbolic tensor dimensions in *inferred_model* by
        running the model once with concrete *module_inputs* via onnxruntime, then
        writing the actual integer values back into every ``dim`` entry
        (inputs, outputs, and all intermediate ``value_info`` tensors) of a deep
        copy of the model.

        After the shapes are patched a quick **verification pass** runs the same
        procedure with random tensors of matching dtype and shape to confirm that
        every patched dimension is consistent.  Any mismatch is logged as a
        warning but does not prevent the patched model from being used.

        Why inplace on a copy rather than a separate map
        -------------------------------------------------
        Writing shapes back into the model proto means every consumer of the
        model (converters, shape resolver, debug validator) automatically gets
        concrete shapes — no secondary lookup table is needed, and the existing
        ``_get_tensor_info`` path just works.

        Args:
            inferred_model: ONNX ModelProto after ``shape_inference.infer_shapes``
                and ``remove_initializers_from_input``.
            module_inputs: Concrete input tensors (``torch.Tensor`` or array-like).

        Returns:
            A deep-copied ModelProto whose every tensor shape uses concrete integer
            ``dim_value`` entries.  Returns the original *inferred_model* unchanged
            if onnxruntime is unavailable or execution fails.
        """
        try:
            import io
            import copy
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not available - skipping concrete shape resolution")
            return inferred_model

        def _run_and_collect(model_proto, np_inputs: dict) -> dict:
            """Serialize *model_proto*, run ORT, return {name: shape_tuple}."""
            # Expose all intermediate tensors as extra graph outputs so ORT
            # returns their shapes.
            probe = copy.deepcopy(model_proto)
            existing = {o.name for o in probe.graph.output}
            for vi in probe.graph.value_info:
                if vi.name not in existing:
                    extra = onnx.helper.ValueInfoProto()
                    extra.name = vi.name
                    probe.graph.output.append(extra)

            buf = io.BytesIO()
            onnx.save(probe, buf)
            buf.seek(0)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # suppress ORT verbose/info messages
            opts.inter_op_num_threads = 1  # no thread pool → no pthread_setaffinity_np errors
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(buf.read(), sess_options=opts)

            out_names = [o.name for o in sess.get_outputs()]
            results = sess.run(out_names, np_inputs)
            return {n: tuple(int(d) for d in r.shape) for n, r in zip(out_names, results)}

        def _to_numpy(t, dtype=None):
            """Convert a torch.Tensor / array to numpy, with optional dtype cast."""
            if isinstance(t, torch.Tensor):
                arr = t.detach().cpu().numpy()
            else:
                arr = np.array(t)
            return arr.astype(dtype) if dtype is not None else arr

        try:
            graph = inferred_model.graph

            # ── Build the concrete-input dict ────────────────────────────────
            ort_input_names = [inp.name for inp in graph.input]
            concrete_np: dict = {}
            for i, name in enumerate(ort_input_names):
                if i < len(module_inputs):
                    concrete_np[name] = _to_numpy(module_inputs[i])

            if not concrete_np:
                logger.warning("No concrete inputs available — skipping shape resolution")
                return inferred_model

            # ── Pass 1: run with concrete inputs ────────────────────────────
            concrete_shapes = _run_and_collect(inferred_model, concrete_np)
            if not concrete_shapes:
                return inferred_model

            # ── Patch shapes inplace on a deep copy ──────────────────────────
            patched_model = copy.deepcopy(inferred_model)
            patched_graph = patched_model.graph

            def _patch_tensor(vi, name):
                shape = concrete_shapes.get(name)
                if shape is None:
                    return 0
                t = vi.type.tensor_type
                if not t.HasField("shape"):
                    return 0
                dims = list(t.shape.dim)
                if len(dims) != len(shape):
                    return 0
                patched = 0
                for dim_proto, concrete_val in zip(dims, shape):
                    # Only replace unknown or symbolic dims; never overwrite a concrete value.
                    # Use WhichOneof so that a legitimate dim_value of 0 is not treated as unset.
                    if dim_proto.WhichOneof("value") != "dim_value":
                        dim_proto.dim_value = int(concrete_val)
                        dim_proto.ClearField("dim_param")
                        patched += 1
                return patched

            total_patched = 0
            for vi in patched_graph.input:
                total_patched += _patch_tensor(vi, vi.name)
            for vi in patched_graph.output:
                total_patched += _patch_tensor(vi, vi.name)
            for vi in patched_graph.value_info:
                total_patched += _patch_tensor(vi, vi.name)

            logger.trace(
                f"Concrete shape resolution: patched {total_patched} unknown dimensions "
                f"across {len(concrete_shapes)} tensors (inputs + outputs + intermediates)."
            )

            # ── Pass 2: verification with random inputs ───────────────────────
            # Build random tensors that match the concrete input shapes/dtypes.
            try:
                random_np: dict = {}
                for name, arr in concrete_np.items():
                    random_np[name] = np.random.randint(0, 128, size=arr.shape).astype(arr.dtype)

                verify_shapes = _run_and_collect(patched_model, random_np)
                mismatches = 0
                for name, expected in concrete_shapes.items():
                    got = verify_shapes.get(name)
                    if got is not None and got != expected:
                        logger.warning(
                            f"Shape verification mismatch for '{name}': "
                            f"expected {expected}, got {got} with random inputs. "
                            f"This tensor's shape may be value-dependent."
                        )
                        mismatches += 1
                if mismatches == 0:
                    logger.trace("Shape verification passed: all patched shapes are consistent.")
                else:
                    logger.warning(
                        f"Shape verification: {mismatches} tensor(s) have value-dependent shapes. "
                        f"Those shapes may be incorrect for inputs other than the ones provided."
                    )
            except Exception as verify_err:
                logger.trace(f"  shape verification pass failed (non-fatal): {verify_err}")

            return patched_model

        except Exception as e:
            logger.warning(f"Concrete shape resolution failed ({e}) — using original inferred model.")
            return inferred_model

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
            logger.trace("ONNX model validation passed")

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
        Extract metadata from an ONNX model for logging and error reporting.

        Args:
            onnx_model: ONNX ModelProto to inspect.

        Returns:
            Dictionary with the following keys:
                name          – graph name (str or "<unnamed>")
                opset         – ONNX opset version (int or None)
                nodes         – number of graph nodes
                inputs        – number of graph inputs
                outputs       – number of graph outputs
                initializers  – number of initializers (weights/constants)
                ir_version    – ONNX IR version (int or None)
                input_names   – list of input tensor names
                output_names  – list of output tensor names
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
                "name": graph.name or "<unnamed>",
                "opset": opset,
                "nodes": len(graph.node) if graph.node else 0,
                "inputs": len(graph.input) if graph.input else 0,
                "outputs": len(graph.output) if graph.output else 0,
                "initializers": len(graph.initializer) if graph.initializer else 0,
                "ir_version": getattr(onnx_model, "ir_version", None),
                "input_names": [vi.name for vi in graph.input],
                "output_names": [vi.name for vi in graph.output],
            }
        except Exception:
            return {}

    def _infer_shapes(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Run ONNX shape inference on the model.

        Shape inference propagates known tensor shapes and data types throughout
        the graph so that converters receive complete ``TensorInfo`` objects.
        Unlike the optional dynamic-shape resolution pass, this step is always
        performed and must succeed before transpilation can continue.

        Args:
            onnx_model: ONNX ModelProto to process.

        Returns:
            A new ModelProto with shape information populated.

        Raises:
            RuntimeError: If ONNX shape inference fails for any reason.
        """
        try:
            return shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            raise RuntimeError(
                f"ONNX shape inference failed — cannot proceed with transpilation.\n" f"Reason: {e}"
            ) from e

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
            # Binary operations (Arithmetic and Comparison)
            "Add": BinaryOpConverter.get_converter(opset),
            "Sub": BinaryOpConverter.get_converter(opset),
            "Mul": BinaryOpConverter.get_converter(opset),
            "Div": BinaryOpConverter.get_converter(opset),
            "Equal": BinaryOpConverter.get_converter(opset),
            "Greater": BinaryOpConverter.get_converter(opset),
            "Less": BinaryOpConverter.get_converter(opset),
            "GreaterOrEqual": BinaryOpConverter.get_converter(opset),
            "LessOrEqual": BinaryOpConverter.get_converter(opset),
            # Logical binary operations
            "And": BinaryOpConverter.get_converter(opset),
            "MatMul": MatMulConverter.get_converter(opset),
            "Gemm": GemmConverter.get_converter(opset),
            # Activation operations
            "Relu": ReluConverter.get_converter(opset),
            "Sigmoid": SigmoidConverter.get_converter(opset),
            "Tanh": TanhConverter.get_converter(opset),
            "Sqrt": SqrtConverter.get_converter(opset),
            "Pow": PowConverter.get_converter(opset),
            "Erf": UnaryOpConverter.get_converter(opset),
            # Logical unary operations
            "Not": UnaryOpConverter.get_converter(opset),
            "Softmax": SoftmaxConverter.get_converter(opset),
            "LogSoftmax": LogSoftmaxConverter.get_converter(opset),
            "LeakyRelu": LeakyReluConverter.get_converter(opset),
            "Dropout": DropoutConverter.get_converter(opset),
            # Reduction operations
            "ReduceSum": ReduceSumConverter.get_converter(opset),
            "ReduceMean": ReduceMeanConverter.get_converter(opset),
            "ReduceMax": ReduceMaxConverter.get_converter(opset),
            "ArgMax": ArgMaxConverter.get_converter(opset),
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
            "Expand": ExpandConverter.get_converter(opset),
            "Concat": ConcatConverter.get_converter(opset),
            "Clip": ClipConverter.get_converter(opset),
            "Conv": ConvConverter.get_converter(opset),
            "Constant": ConstantConverter.get_converter(opset),
            # Conditional operations
            "Where": WhereConverter.get_converter(opset),
            # Normalization operations
            "LayerNormalization": LayerNormalizationConverter.get_converter(opset),
            # Indexing operations
            "Gather": GatherConverter.get_converter(opset),
            "Slice": SliceConverter.get_converter(opset),
            # Creation operations
            "ConstantOfShape": ConstantOfShapeConverter.get_converter(opset),
            # Shape operations
            "Shape": ShapeConverter.get_converter(opset),
            # Triangular masking
            "Trilu": TriluConverter.get_converter(opset),
        }

        return convert_map

    def transpile(self, onnx_model: onnx.ModelProto, module_inputs: Optional[List[torch.Tensor]] = None) -> TIRGraph:
        """
        Transpile an ONNX model to a TIR graph.

        This is the main entry point for converting ONNX models to TIRGraph.
        The process includes:
        1. Model validation (if enabled)
        2. ONNX shape inference
        3. Dynamic shape resolution (when *module_inputs* are provided and
           ``self.resolve_dynamic_shapes`` is True): the inferred model is
           executed once via onnxruntime to replace every unknown/symbolic
           dimension with its real integer value, directly in a patched copy of
           the model proto.
        4. Initializer processing (parameters vs constants)
        5. Node conversion using opset-specific converters
        6. Graph construction and name sanitization

        Args:
            onnx_model: ONNX ModelProto to transpile.
            module_inputs: Optional list of concrete input tensors.  When
                provided and ``self.resolve_dynamic_shapes`` is True, all
                symbolic / unknown dimensions (inputs, outputs, and every
                intermediate tensor) are resolved to concrete integers by
                running the model once with onnxruntime.  The patched model is
                used for the rest of transpilation so that every converter
                receives fully concrete shapes.

        Returns:
            TIRGraph representing the converted model

        Raises:
            ONNXModelValidationError: If model validation fails
            UnsupportedOperationError: If unsupported operations are found
            ConversionError: If node conversion fails
        """
        _minfo = self._extract_model_info(onnx_model)
        logger.info(
            f"Starting ONNX → TIR Transpilation\n"
            f"  Model    : '{_minfo.get('name', '<unnamed>')}'\n"
            f"  IR ver   : {_minfo.get('ir_version')}  |  Opset: {_minfo.get('opset')}\n"
            f"  Nodes    : {_minfo.get('nodes')}  |  Initializers: {_minfo.get('initializers')}\n"
            f"  Inputs   : {_minfo.get('input_names', [])}\n"
            f"  Outputs  : {_minfo.get('output_names', [])}"
        )

        # Store original model for debug mode (needed for ONNX Runtime comparison)
        self.onnx_model = onnx_model

        # Step 1: Validate ONNX model structure and schema (if enabled)
        # This catches errors early before attempting conversion
        if self.validate_model:
            self._validate_onnx_model(onnx_model)

        # Step 2: Extract opset version from model
        # Opset version determines which converter logic to use for each operation
        self.opset = self._get_opset_version(onnx_model)

        # Step 3: Build converter map for this opset version
        # Each converter is bound to the specific opset version
        self._op_converters = self._build_convert_map(self.opset)
        # Reset state tracking for name generation and uniqueness
        self._reset_transpilation_state()

        # Step 4: Run shape inference to determine tensor shapes throughout the graph.
        # Delegates to _infer_shapes(), which raises RuntimeError on failure.
        inferred_model = self._infer_shapes(onnx_model)

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

        # Step 7: Resolve concrete shapes (when enabled and inputs are available).
        # ONNX shape_inference propagates ranks/types but cannot determine shapes
        # whose values depend on runtime tensors (e.g. Expand driven by
        # Shape → Gather → Where).  When concrete module_inputs are provided and
        # resolve_dynamic_shapes is True, we run the inferred model once via
        # onnxruntime to replace every unknown/symbolic dim with a real integer —
        # directly inside the model proto.  The rest of transpilation then uses
        # this patched model, so value_info_map automatically contains concrete shapes.
        if self.resolve_dynamic_shapes and module_inputs is not None:
            inferred_model = self._resolve_model_shapes_inplace(inferred_model, module_inputs)
            # Re-extract graph proto from the potentially-patched model.
            graph_proto = inferred_model.graph
            self.graph_proto = graph_proto

        # Step 7.5: Build value_info_map from the (possibly patched) model proto.
        # All shape entries are now concrete integers wherever they could be resolved.
        value_info_map = {vi.name: vi for vi in graph_proto.value_info}
        value_info_map.update({vi.name: vi for vi in graph_proto.input})
        value_info_map.update({vi.name: vi for vi in graph_proto.output})

        # Step 8: Process initializers (weights, biases, constants)
        # Initializers are pre-computed tensor values stored in the ONNX model
        # They can be either parameters (trainable) or constants (non-trainable)
        num_initializers = len(graph_proto.initializer)
        logger.trace(f"Processing {num_initializers} initializer(s) (freeze_params={self.freeze_params})")
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

        logger.trace(
            f"  Loaded {len(tir_graph.params)} param(s) and {len(tir_graph.constants)} constant(s) "
            f"from initializers."
        )

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
        logger.trace(f"  Graph inputs ({len(original_inputs)}): {original_inputs}")

        # Step 9.5: Pre-register graph output names in the mapping
        # This ensures that when nodes are added, their output names that match graph outputs
        # will use the pre-registered sanitized names, ensuring consistency
        original_outputs = get_outputs_names(graph_proto)
        for original_output in original_outputs:
            # Only register if not already in mapping (from inputs or previous processing)
            if original_output not in tir_graph.original_to_sanitized:
                # Sanitize original name, or generate default if sanitization fails
                base_name = sanitize_name(original_output) or f"output_{len(tir_graph.original_to_sanitized)}"
                # Ensure uniqueness across all names (inputs, outputs, nodes)
                all_used_names = set(tir_graph.sanitized_to_original.keys()) | self._generated_sanitized_names
                clean_name = ensure_unique_name(base_name, all_used_names)
                # Store bidirectional mapping
                tir_graph.original_to_sanitized[original_output] = clean_name
                tir_graph.sanitized_to_original[clean_name] = original_output
                self._generated_sanitized_names.add(clean_name)

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
            logger.trace("All ONNX operations are supported. Proceeding with conversion.")

        # Step 11: Convert each ONNX node to TIR nodes
        # Process nodes in order (ONNX graphs are typically topologically sorted)
        invalid_nodes = []
        total_nodes = len(graph_proto.node)
        _unique_op_types = sorted(set(n.op_type for n in graph_proto.node))
        logger.trace(
            f"Converting {total_nodes} ONNX node(s) → TIR, Op types ({len(_unique_op_types)}): {_unique_op_types}"
        )

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

            # Per-node ONNX node header — build via ConversionLogger, emit ONE combined trace after conversion
            node_name = node_proto.name or f"{op_type}_{i}"
            logger.trace(f"  [{i + 1}/{total_nodes}] {op_type} '{node_name}'")
            _onnx_section = TranspilerLogger.format_onnx_node_section(
                node_proto, input_tensors, output_tensors, attrs, i, total_nodes
            )

            # Collect result log lines, then emit ONE combined trace entry
            _log_lines = []

            # Call converter to convert ONNX node to TIR nodes
            # Pass tir_graph so converters can access constants and perform inline shape resolution
            try:
                converter_result: ConverterResult = converter_method(
                    node_proto, input_tensors, output_tensors, attrs, i, self.graph_proto, tir_graph=tir_graph
                )

                # Propagate concrete output shapes back into value_info_map so
                # that subsequent nodes in the loop see fully-resolved shapes
                # when they look up their inputs.  resolve_output_shapes()
                # (called in base.py) already wrote concrete dims into
                # output_tensors; we mirror those here so downstream converters
                # never have to re-resolve the same dimension.
                for _oname, _oinfo in output_tensors.items():
                    if (
                        _oname
                        and _oinfo.shape is not None
                        and all(isinstance(d, int) for d in _oinfo.shape)
                        and _oinfo.onnx_dtype != onnx.TensorProto.UNDEFINED
                    ):
                        value_info_map[_oname] = onnx.helper.make_tensor_value_info(
                            _oname,
                            _oinfo.onnx_dtype,
                            [int(d) for d in _oinfo.shape],
                        )

                # Handle converter result — handlers append their details to _log_lines
                if is_constant_result(converter_result):
                    self._handle_constant_result(
                        converter_result, tir_graph, op_type, node_proto, i, invalid_nodes, log_lines=_log_lines
                    )
                else:
                    self._handle_tir_nodes_result(
                        converter_result, tir_graph, op_type, node_proto, i, invalid_nodes, log_lines=_log_lines
                    )

                # ONE combined trace entry: ONNX header + result + closing separator
                TranspilerLogger.emit_node_trace(_onnx_section, _log_lines)

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
        logger.trace(f"  Graph outputs ({len(original_outputs)}): {original_outputs}")

        # Step 14: Compute activation dependencies for memory management
        # This determines which activations are still needed at each point in the graph
        # Used for garbage collection and memory optimization in code generation
        tir_graph.compute_activation_dependencies()

        _orig_inputs_done = [tir_graph.sanitized_to_original.get(i, i) for i in tir_graph.inputs]
        _orig_outputs_done = [tir_graph.sanitized_to_original.get(o, o) for o in tir_graph.outputs]
        logger.info(
            f"Transpilation Complete TIR nodes: {len(tir_graph.nodes)} Params: {len(tir_graph.params)} Constants: {len(tir_graph.constants)} (+ {len(tir_graph.computed_constants)} computed) Inputs: {_orig_inputs_done} Outputs: {_orig_outputs_done}"
        )

        return tir_graph
