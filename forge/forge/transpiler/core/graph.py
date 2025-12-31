# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Graph representation for the transpiler intermediate representation.
Framework-agnostic - works for all frontends.
"""
import torch
from loguru import logger
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from copy import deepcopy

from forge.transpiler.core.node import TIRNode
from forge.transpiler.utils.framework_conversion import convert_inputs_to_framework
from forge.transpiler.core.exceptions import ValidationError, DebugValidationError


class TIRGraph:
    """
    Computational graph in Transpiler Intermediate Representation (TIR).

    Represents a framework-agnostic intermediate graph between ML frameworks (e.g., ONNX)
    and Forge module graphs. Manages nodes, tensors, topology, and execution.

    Attributes:
        name: Graph name
        nodes: List of TIR nodes in the graph
        inputs: List of sanitized input names (for code generation)
        outputs: List of sanitized output names (for code generation)
        original_inputs: List of original frontend input names (for parameter loading)
        original_outputs: List of original frontend output names (for verification)
        original_to_sanitized: Mapping from original names to sanitized names
        sanitized_to_original: Reverse mapping from sanitized to original names
        params: Dictionary of trainable weights (uses original names)
        constants: Dictionary of non-trainable values (uses original names)
        producer_map: Maps output names to producing node names (uses sanitized names)
        consumer_map: Maps tensor names to list of consuming node names (uses sanitized names)
        needed_by: Activation dependency map for memory management
        framework: Framework name (currently only "onnx" supported)
        debug_mode: Whether debug mode is enabled
        frontend_model: Original frontend model for debug comparisons
        frontend_node_map: Maps TIR node names to frontend node representations
        frontend_node_to_tir_nodes: Maps frontend node names to list of TIR node names
    """

    def __init__(self, name: str, framework: str, frontend_model=None, debug_mode: bool = False):
        """
        Initialize a TIRGraph.

        Args:
            name: Graph name
            framework: Framework name (currently only "onnx" supported)
            frontend_model: Original frontend model for debug comparisons (optional)
            debug_mode: Enable debug mode for validation (default: False)

        Raises:
            ValueError: If framework is not "onnx"
        """
        self.name = name
        self.nodes: List[TIRNode] = []
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.original_inputs: List[str] = []
        self.original_outputs: List[str] = []
        self.original_to_sanitized: Dict[str, str] = {}
        self.sanitized_to_original: Dict[str, str] = {}
        self.params: Dict[str, torch.Tensor] = {}
        self.constants: Dict[str, torch.Tensor] = {}
        self.producer_map: Dict[str, str] = {}
        self.consumer_map: Dict[str, List[str]] = {}
        self.needed_by: Optional[Dict[str, set]] = None

        if framework != "onnx":
            raise ValueError(f"Unsupported framework: {framework}. " f"Currently only 'onnx' framework is supported.")
        self.framework: str = framework
        self.debug_mode: bool = debug_mode
        self.frontend_model = frontend_model
        self.frontend_node_map: Dict[str, Any] = {}
        self.frontend_node_to_tir_nodes: Dict[str, List[str]] = defaultdict(list)

    @property
    def initializers(self) -> Dict[str, torch.Tensor]:
        """
        Backward compatibility property combining params and constants.

        Returns:
            Combined dictionary of all initializers (params + constants)
        """
        combined = {}
        combined.update(self.params)
        combined.update(self.constants)
        return combined

    def add_node(self, node: TIRNode):
        """
        Add a node to the graph and update topology maps.

        Updates producer_map and consumer_map to maintain graph topology.
        Supports nodes with multiple outputs.

        Args:
            node: TIRNode to add to the graph
        """
        self.nodes.append(node)
        for output_name in node.outputs:
            self.producer_map[output_name] = node.name
        for in_name in node.inputs:
            if in_name not in self.consumer_map:
                self.consumer_map[in_name] = []
            self.consumer_map[in_name].append(node.name)

    def get_node_by_name(self, name: str) -> Optional[TIRNode]:
        """
        Get a node by its name.

        Args:
            name: Node name to search for

        Returns:
            TIRNode if found, None otherwise
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_topological_sort(self) -> List[TIRNode]:
        """
        Get nodes in topological order for execution.

        Uses Kahn's algorithm to sort nodes such that all dependencies
        of a node appear before the node itself. This ensures correct
        execution order where inputs are computed before nodes that use them.

        Algorithm:
        1. Calculate in-degree (number of dependencies) for each node
        2. Start with nodes that have no dependencies (in-degree = 0)
        3. Process nodes, decrementing in-degree of dependent nodes
        4. Add nodes to result when their in-degree reaches 0

        Returns:
            List of TIRNodes in topological order
        """
        # Step 1: Initialize in-degree counter for all nodes
        in_degree = {node.name: 0 for node in self.nodes}

        # Step 2: Calculate in-degree by counting dependencies
        # A node depends on another if it uses that node's output as input
        for node in self.nodes:
            for input_name in node.inputs:
                # Check if input is produced by another node (not a model input/initializer)
                if input_name in self.producer_map and self.producer_map[input_name] != node.name:
                    in_degree[node.name] += 1

        # Step 3: Initialize queue with nodes that have no dependencies
        queue = deque([node for node in self.nodes if in_degree[node.name] == 0])
        sorted_nodes = []

        # Step 4: Process nodes in topological order
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)

            # For each output of this node, decrement in-degree of dependent nodes
            for output_name in node.outputs:
                if output_name in self.consumer_map:
                    for consumer_name in self.consumer_map[output_name]:
                        if consumer_name in in_degree:
                            in_degree[consumer_name] -= 1
                            # When all dependencies are satisfied, add to queue
                            if in_degree[consumer_name] == 0:
                                consumer_node = self.get_node_by_name(consumer_name)
                                if consumer_node:
                                    queue.append(consumer_node)
        return sorted_nodes

    def compute_activation_dependencies(self):
        """
        Compute activation dependencies for memory management.

        Determines which activations are needed by which nodes, enabling
        garbage collection of unused activations during graph execution.
        Tracks all outputs from each node to properly handle multi-output operations.

        Algorithm:
        For each node, for each output tensor, mark which input tensors
        need that output. This creates a reverse dependency graph that tracks
        when an activation can be safely garbage collected.

        Example: If node B uses output of node A, then A's output is needed by B.
        When B finishes executing, A's output can potentially be deleted if
        no other nodes need it.

        Returns:
            Dictionary mapping tensor names to sets of tensor names that need them
        """
        needed_by = defaultdict(set)

        # Build reverse dependency graph: for each output, track which inputs need it
        # This allows us to know when an activation is no longer needed
        for node in self.nodes:
            for out_op_id in node.outputs:
                for in_op_id in node.inputs:
                    # Input tensor 'in_op_id' is needed by output tensor 'out_op_id'
                    # This means the producer of 'in_op_id' must keep it until 'out_op_id' is computed
                    needed_by[in_op_id].add(out_op_id)

        # Convert defaultdict to regular dict for serialization safety
        needed_by.default_factory = None
        self.needed_by = dict(needed_by)
        return self.needed_by

    def run(self, inputs: Dict[str, torch.Tensor], enable_gc: bool = True) -> Dict[str, torch.Tensor]:
        """
        Execute the graph with given inputs.

        Executes all nodes in topological order, managing tensor memory and optionally
        performing garbage collection of unused activations. In debug mode, validates
        outputs against the original frontend model.

        Args:
            inputs: Dictionary mapping input names to PyTorch tensors
            enable_gc: Enable activation garbage collection for memory optimization (default: True)

        Returns:
            Dictionary mapping output names (original names) to result tensors

        Raises:
            ValidationError: If debug validation fails
        """
        logger.info(f"Executing Graph: {self.name} (debug_mode={self.debug_mode})")
        # Initialize tensor memory with parameters, constants, and input tensors
        # Parameters and constants are persistent throughout execution
        tensor_memory = {}
        tensor_memory.update(self.params)
        tensor_memory.update(self.constants)
        tensor_memory.update(inputs)

        # Compute activation dependencies for garbage collection if enabled
        # This tracks which activations are still needed by future nodes
        if enable_gc and self.needed_by is None:
            self.compute_activation_dependencies()

        # Create a working copy of dependency map for tracking during execution
        # We'll decrement counts as nodes execute, marking tensors for deletion
        still_needed_by = deepcopy(self.needed_by) if enable_gc and self.needed_by else None

        # Prepare inputs for debug mode (convert to framework-specific format)
        # Uses framework information to convert inputs to the appropriate format
        # (e.g., numpy for ONNX, PyTorch tensors for PyTorch models, etc.)
        debug_inputs = None
        if self.debug_mode and self.frontend_model is not None:
            # Collect inputs in a dictionary for conversion
            input_dict = {}
            for input_name in self.inputs:
                if input_name in inputs:
                    input_dict[input_name] = inputs[input_name]

            # Convert to framework-specific format
            # Use self.inputs order to preserve input ordering
            debug_inputs = convert_inputs_to_framework(input_dict, self.framework, input_order=self.inputs)

        execution_plan = self.get_topological_sort()

        # Track which frontend nodes have been validated (to avoid duplicate validation)
        validated_frontend_nodes = set()

        # Track outputs from TIR nodes for each frontend node (for grouped validation)
        frontend_node_outputs = defaultdict(dict)  # frontend_node_name -> {output_name: tensor}

        # Track which TIR nodes have been executed (for each frontend node)
        frontend_node_executed_tir_nodes = defaultdict(set)  # frontend_node_name -> set of executed TIR node names

        # Log debug mode status (concise)
        if self.debug_mode:
            mapping_size = len(self.frontend_node_to_tir_nodes)
            if mapping_size > 0:
                logger.debug(f"Debug mode: {mapping_size} {self.framework.upper()} nodes mapped for validation")
            else:
                logger.warning(
                    f"Debug mode enabled but no {self.framework.upper()} node mappings found. "
                    "Validation will not be performed."
                )

        for node in execution_plan:
            node_inputs = {}
            ready = True
            for inp in node.inputs:
                if inp in tensor_memory:
                    node_inputs[inp] = tensor_memory[inp]
                else:
                    logger.error(f"Node {node.name} missing input: {inp}")
                    ready = False

            if ready:
                outputs = node.eval(node_inputs)
                # Store outputs in tensor_memory with sanitized names (for graph execution)
                tensor_memory.update(outputs)

                # Map outputs back to original names for debug/comparison (only if debug mode enabled)
                # outputs from node.eval() use sanitized names (node.outputs)
                # but debug validation expects original frontend names
                outputs_with_original_names = None
                if self.debug_mode and self.frontend_model is not None and debug_inputs is not None:
                    outputs_with_original_names = {}
                    # Use mapping from graph to convert sanitized names back to original names
                    # This is more reliable than index-based access since order is preserved during sanitization
                    for sanitized_name, tensor in outputs.items():
                        original_name = self.sanitized_to_original.get(sanitized_name, sanitized_name)
                        outputs_with_original_names[original_name] = tensor

                    # Collect outputs for frontend node validation
                    frontend_node_repr = self.frontend_node_map.get(node.name)
                    if frontend_node_repr and hasattr(frontend_node_repr, "name") and frontend_node_repr.name:
                        frontend_node_name = frontend_node_repr.name

                        # Add this node's outputs to the frontend node's output collection
                        frontend_node_outputs[frontend_node_name].update(outputs_with_original_names)

                        # Mark this TIR node as executed for the frontend node
                        frontend_node_executed_tir_nodes[frontend_node_name].add(node.name)

                        # Get all TIR nodes that belong to this frontend node
                        # Use direct access to defaultdict to ensure it's populated
                        tir_nodes_for_frontend = self.frontend_node_to_tir_nodes[frontend_node_name]

                        # Validate if all TIR nodes for this frontend node have been executed
                        # Note: If mapping wasn't built (debug not enabled during transpilation),
                        # tir_nodes_for_frontend will be empty, so validation won't trigger
                        executed_count = len(frontend_node_executed_tir_nodes[frontend_node_name])
                        total_count = len(tir_nodes_for_frontend)

                        # Log condition check for debugging (only if mapping exists)
                        if total_count > 0:
                            logger.debug(
                                f"TIR node {node.name} -> {self.framework.upper()} node {frontend_node_name}: "
                                f"executed={executed_count}/{total_count}, "
                                f"tir_nodes={tir_nodes_for_frontend}, "
                                f"already_validated={frontend_node_name in validated_frontend_nodes}"
                            )

                        if (
                            total_count > 0
                            and executed_count == total_count
                            and frontend_node_name not in validated_frontend_nodes
                        ):
                            logger.info(
                                f"[{frontend_node_name}] Starting validation against {len(tir_nodes_for_frontend)} TIR node(s): "
                                f"{', '.join(tir_nodes_for_frontend)}"
                            )
                            validated_frontend_nodes.add(frontend_node_name)

                            # Collect outputs from TIR nodes belonging to this frontend node
                            # Filter to only include outputs that match the frontend node's expected outputs
                            # This is important for operations like Transpose that create multiple intermediate nodes
                            all_collected_outputs = frontend_node_outputs[frontend_node_name]
                            # Get the frontend node representation to find expected output names
                            # Use the first TIR node to find the frontend node (all TIR nodes map to same frontend node)
                            frontend_node_repr = None
                            if tir_nodes_for_frontend:
                                frontend_node_repr = self.frontend_node_map.get(tir_nodes_for_frontend[0])

                            if frontend_node_repr and hasattr(frontend_node_repr, "output"):
                                # Only include outputs that match the frontend node's output names
                                # Note: all_collected_outputs already uses original names (from outputs_with_original_names)
                                expected_output_names = (
                                    list(frontend_node_repr.output) if hasattr(frontend_node_repr, "output") else []
                                )
                                collected_outputs = {
                                    name: tensor
                                    for name, tensor in all_collected_outputs.items()
                                    if name in expected_output_names
                                }
                            else:
                                # Fallback: use all collected outputs if we can't determine expected outputs
                                collected_outputs = all_collected_outputs

                            # If no outputs match, log warning but continue (may indicate a bug)
                            if not collected_outputs and all_collected_outputs:
                                logger.warning(
                                    f"Frontend node {frontend_node_name} expected outputs {expected_output_names if frontend_node_repr and hasattr(frontend_node_repr, 'output') else 'unknown'}, "
                                    f"but collected outputs have names: {list(all_collected_outputs.keys())}"
                                )

                            # Validate frontend node against collected TIR node outputs
                            # Framework-specific validation is handled by the frontend's debug validator
                            if self.framework == "onnx":
                                try:
                                    from ..frontends.onnx.debug.validator import debug_node_output

                                    debug_node_output(
                                        self.frontend_model,
                                        debug_inputs,
                                        collected_outputs,  # All outputs from TIR nodes for this frontend node
                                        frontend_node_repr,
                                    )
                                    logger.info(
                                        f"[{frontend_node_name}] ✓ Validated {self.framework.upper()} node against {len(tir_nodes_for_frontend)} TIR node(s): "
                                        f"{', '.join(tir_nodes_for_frontend)}"
                                    )
                                except DebugValidationError as e:
                                    # Debug validation errors (shape/value mismatches) should stop execution
                                    error_msg = (
                                        f"Debug validation failed for {self.framework.upper()} node {frontend_node_name} "
                                        f"(TIR nodes: {', '.join(tir_nodes_for_frontend)}): {e}. "
                                        f"Stopping execution in debug mode."
                                    )
                                    logger.error(error_msg)
                                    # Wrap in ValidationError for consistency
                                    raise ValidationError(
                                        error_msg,
                                        details={
                                            "frontend_node_name": frontend_node_name,
                                            "tir_nodes": tir_nodes_for_frontend,
                                            "framework": self.framework,
                                        },
                                    ) from e
                                except Exception as e:
                                    # Other exceptions (e.g., ONNX Runtime errors) should also stop execution in debug mode
                                    error_msg = (
                                        f"Debug comparison failed for {self.framework.upper()} node {frontend_node_name} "
                                        f"(TIR nodes: {', '.join(tir_nodes_for_frontend)}): {e}. "
                                        f"This may indicate an issue with {self.framework.upper()} runtime or model setup."
                                    )
                                    logger.error(error_msg, exc_info=True)
                                    raise ValidationError(
                                        error_msg,
                                        details={
                                            "frontend_node_name": frontend_node_name,
                                            "tir_nodes": tir_nodes_for_frontend,
                                            "framework": self.framework,
                                            "original_error": str(e),
                                        },
                                    ) from e
                            else:
                                raise NotImplementedError(
                                    f"Debug validation not yet implemented for framework: {self.framework}"
                                )

                # Garbage collection: free memory for activations no longer needed
                # After a node executes, check if its inputs are still needed by future nodes
                # If an input tensor's dependency count reaches 0, it can be safely deleted
                # Note: We process all outputs from each node to handle multi-output operations correctly
                if enable_gc and still_needed_by is not None:
                    for out_op_id in node.outputs:
                        for in_op_id in node.inputs:
                            if in_op_id in still_needed_by:
                                # This output no longer needs this input, decrement dependency count
                                still_needed_by[in_op_id].discard(out_op_id)
                                # If no more nodes need this input, delete it from memory
                                if len(still_needed_by[in_op_id]) == 0:
                                    # Safety check: never delete parameters, constants, or model inputs
                                    if (
                                        in_op_id in tensor_memory
                                        and in_op_id not in self.params
                                        and in_op_id not in self.constants
                                    ):
                                        del tensor_memory[in_op_id]
                                        logger.debug(f"GC: Deleted activation {in_op_id}")

        # Log validation summary if debug mode was enabled
        if self.debug_mode and len(validated_frontend_nodes) > 0:
            logger.info(
                f"Debug validation summary: Validated {len(validated_frontend_nodes)} {self.framework.upper()} node(s) "
                f"against their corresponding TIR nodes"
            )
        elif self.debug_mode and len(self.frontend_node_to_tir_nodes) > 0:
            logger.warning(
                f"Debug mode enabled but no {self.framework.upper()} nodes were validated. "
                f"Expected {len(self.frontend_node_to_tir_nodes)} {self.framework.upper()} nodes to be validated."
            )

        # Return results with original names for compatibility with comparison functions
        result = {}
        for sanitized_out_name in self.outputs:
            if sanitized_out_name in tensor_memory:
                # Map back to original name for comparison functions
                original_out_name = self.sanitized_to_original.get(sanitized_out_name, sanitized_out_name)
                result[original_out_name] = tensor_memory[sanitized_out_name]
            else:
                original_out_name = self.sanitized_to_original.get(sanitized_out_name, sanitized_out_name)
                logger.error(f"Graph output {original_out_name} (sanitized: {sanitized_out_name}) was not produced.")
        return result
