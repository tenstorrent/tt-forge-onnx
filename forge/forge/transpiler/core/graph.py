"""
Graph representation for the transpiler intermediate representation.
Framework-agnostic - works for all frontends.
"""
import torch
from loguru import logger
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from copy import deepcopy

from forge.transpiler.ir.nodes import TIRNode


class TIRGraph:
    """Represents a computational graph in Transpiler Intermediate Representation (TIR)."""
    def __init__(self, name: str, frontend_model=None, debug_mode: bool = False):
        self.name = name
        self.nodes: List[TIRNode] = []
        self.inputs: List[str] = []  # Sanitized input names (for code generation)
        self.outputs: List[str] = []  # Sanitized output names (for code generation)
        
        # Original names (for parameter loading, verification, etc.)
        self.original_inputs: List[str] = []  # Original ONNX input names
        self.original_outputs: List[str] = []  # Original ONNX output names
        
        # Name mappings: original -> sanitized
        self.original_to_sanitized: Dict[str, str] = {}  # Map original names to sanitized names
        self.sanitized_to_original: Dict[str, str] = {}  # Map sanitized names to original names (reverse)
        
        # Parameters vs Constants Distinction
        # Note: params and constants use ORIGINAL names (for parameter loading from ONNX)
        self.params: Dict[str, torch.Tensor] = {}      # Trainable weights (e.g., conv weights, biases)
        self.constants: Dict[str, torch.Tensor] = {}    # Non-trainable values (e.g., constant scalars, int tensors)
        
        # Topology info (uses sanitized names)
        self.producer_map: Dict[str, str] = {} 
        self.consumer_map: Dict[str, List[str]] = {}
        
        # Activation memory management
        self.needed_by: Optional[Dict[str, set]] = None
        
        # Debug mode
        self.debug_mode: bool = debug_mode
        self.frontend_model = frontend_model  # Store original model for debug comparisons
        self.node_proto_map: Dict[str, Any] = {}  # Map TIR node names to frontend node protos
        self.onnx_node_to_tir_nodes: Dict[str, List[str]] = defaultdict(list)  # Map ONNX node names to list of TIR node names
    
    @property
    def initializers(self) -> Dict[str, torch.Tensor]:
        """
        Backward compatibility property that combines params and constants.
        Returns a combined dictionary of all initializers.
        """
        combined = {}
        combined.update(self.params)
        combined.update(self.constants)
        return combined
        
    def add_node(self, node: TIRNode):
        """Add a node to the graph and update topology maps."""
        self.nodes.append(node)
        # Handle multiple outputs
        for output_name in node.outputs:
            self.producer_map[output_name] = node.name
        for in_name in node.inputs:
            if in_name not in self.consumer_map:
                self.consumer_map[in_name] = []
            self.consumer_map[in_name].append(node.name)

    def get_node_by_name(self, name: str) -> Optional[TIRNode]:
        """Get a node by its name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_topological_sort(self) -> List[TIRNode]:
        """Get nodes in topological order for execution."""
        in_degree = {node.name: 0 for node in self.nodes}
        
        # Calculate in-degree based on internal graph dependencies
        for node in self.nodes:
            for input_name in node.inputs:
                # Check if input is produced by another node (i.e., not a model input/initializer)
                if input_name in self.producer_map and self.producer_map[input_name] != node.name:
                    in_degree[node.name] += 1
        
        # Initialize queue with nodes that have zero in-degree
        queue = deque([node for node in self.nodes if in_degree[node.name] == 0])
        sorted_nodes = []
        
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            
            # Decrease in-degree of consumers
            for output_name in node.outputs:
                if output_name in self.consumer_map:
                    for consumer_name in self.consumer_map[output_name]:
                        if consumer_name in in_degree:
                            in_degree[consumer_name] -= 1
                            if in_degree[consumer_name] == 0:
                                consumer_node = self.get_node_by_name(consumer_name)
                                if consumer_node:
                                    queue.append(consumer_node)
        return sorted_nodes

    def compute_activation_dependencies(self):
        """
        Compute activation dependencies - which nodes need which activations.
        Used for memory management (garbage collection of unused activations).
        """
        needed_by = defaultdict (set)
        
        for node in self.nodes:
            out_op_id = node.outputs[0] if node.outputs else None
            if out_op_id:
                for in_op_id in node.inputs:
                    needed_by[in_op_id].add(out_op_id)
        
        needed_by.default_factory = None
        self.needed_by = dict(needed_by)
        return self.needed_by

    def run(self, inputs: Dict[str, torch.Tensor], enable_gc: bool = True) -> Dict[str, torch.Tensor]:
        """
        Execute the graph with given inputs.
        
        Args:
            inputs: Input tensors dictionary
            enable_gc: Enable activation garbage collection (memory optimization)
        """
        logger.info(f"Executing Graph: {self.name} (debug_mode={self.debug_mode})")
        tensor_memory = {}
        # Add both params and constants to tensor memory
        tensor_memory.update(self.params)
        tensor_memory.update(self.constants)
        tensor_memory.update(inputs)
        
        # Compute dependencies if not already computed and GC is enabled
        if enable_gc and self.needed_by is None:
            self.compute_activation_dependencies()
        
        still_needed_by = deepcopy(self.needed_by) if enable_gc and self.needed_by else None
        
        # Prepare inputs for debug mode (convert to numpy)
        debug_inputs = None
        if self.debug_mode and self.frontend_model is not None:
            import numpy as np
            debug_inputs = []
            for input_name in self.inputs:
                if input_name in inputs:
                    tensor = inputs[input_name]
                    if isinstance(tensor, torch.Tensor):
                        debug_inputs.append(tensor.detach().cpu().numpy())
                    else:
                        debug_inputs.append(np.array(tensor))
        
        execution_plan = self.get_topological_sort()
        
        # Track which ONNX nodes have been validated (to avoid duplicate validation)
        validated_onnx_nodes = set()
        
        # Track outputs from TIR nodes for each ONNX node (for grouped validation)
        onnx_node_outputs = defaultdict(dict)  # onnx_node_name -> {output_name: tensor}
        
        # Track which TIR nodes have been executed (for each ONNX node)
        onnx_node_executed_tir_nodes = defaultdict(set)  # onnx_node_name -> set of executed TIR node names
        
        # Log mapping status for debugging
        if self.debug_mode:
            mapping_size = len(self.onnx_node_to_tir_nodes)
            node_proto_map_size = len(self.node_proto_map)
            logger.info(
                f"Graph execution debug status: debug_mode={self.debug_mode}, "
                f"node_proto_map_size={node_proto_map_size}, "
                f"onnx_node_to_tir_nodes_size={mapping_size}"
            )
            if mapping_size > 0:
                logger.info(
                    f"ONNX node to TIR nodes mapping: {mapping_size} ONNX nodes mapped for validation"
                )
                # Log first few mappings for debugging
                sample_mappings = list(self.onnx_node_to_tir_nodes.items())[:3]
                for onnx_name, tir_nodes in sample_mappings:
                    logger.debug(f"  {onnx_name} -> {tir_nodes}")
            else:
                logger.warning(
                    "Debug mode enabled but onnx_node_to_tir_nodes mapping is empty. "
                    "This may indicate debug mode was not enabled during transpilation. "
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
                # but debug_node_output expects original ONNX names
                outputs_with_original_names = None
                if self.debug_mode and self.frontend_model is not None and debug_inputs is not None:
                    outputs_with_original_names = {}
                    if hasattr(node, 'original_outputs') and node.original_outputs:
                        # Use stored original outputs for mapping (more efficient and reliable)
                        for i, sanitized_name in enumerate(node.outputs):
                            if sanitized_name in outputs:
                                original_name = node.original_outputs[i] if i < len(node.original_outputs) else sanitized_name
                                outputs_with_original_names[original_name] = outputs[sanitized_name]
                    else:
                        # Fallback: use mapping from graph
                        for sanitized_name, tensor in outputs.items():
                            original_name = self.sanitized_to_original.get(sanitized_name, sanitized_name)
                            outputs_with_original_names[original_name] = tensor
                    
                    # Collect outputs for ONNX node validation
                    frontend_node = self.node_proto_map.get(node.name)
                    if frontend_node and frontend_node.name:
                        onnx_node_name = frontend_node.name
                        
                        # Add this node's outputs to the ONNX node's output collection
                        onnx_node_outputs[onnx_node_name].update(outputs_with_original_names)
                        
                        # Mark this TIR node as executed for the ONNX node
                        onnx_node_executed_tir_nodes[onnx_node_name].add(node.name)
                        
                        # Get all TIR nodes that belong to this ONNX node
                        # Use direct access to defaultdict to ensure it's populated
                        tir_nodes_for_onnx = self.onnx_node_to_tir_nodes[onnx_node_name]
                        
                        # Validate if all TIR nodes for this ONNX node have been executed
                        # Note: If mapping wasn't built (debug not enabled during transpilation),
                        # tir_nodes_for_onnx will be empty, so validation won't trigger
                        executed_count = len(onnx_node_executed_tir_nodes[onnx_node_name])
                        total_count = len(tir_nodes_for_onnx)
                        
                        # Log condition check for debugging (only if mapping exists)
                        if total_count > 0:
                            logger.debug(
                                f"TIR node {node.name} -> ONNX node {onnx_node_name}: "
                                f"executed={executed_count}/{total_count}, "
                                f"tir_nodes={tir_nodes_for_onnx}, "
                                f"already_validated={onnx_node_name in validated_onnx_nodes}"
                            )
                        
                        if total_count > 0 and executed_count == total_count and onnx_node_name not in validated_onnx_nodes:
                            logger.info(
                                f"[{onnx_node_name}] Starting validation against {len(tir_nodes_for_onnx)} TIR node(s): "
                                f"{', '.join(tir_nodes_for_onnx)}"
                            )
                            validated_onnx_nodes.add(onnx_node_name)
                            
                            # Collect all outputs from TIR nodes belonging to this ONNX node
                            collected_outputs = onnx_node_outputs[onnx_node_name]
                            
                            # Validate ONNX node proto against collected TIR node outputs
                            try:
                                from ..frontends.onnx.debug.validator import debug_node_output, DebugValidationError
                                debug_node_output(
                                    self.frontend_model,
                                    debug_inputs,
                                    collected_outputs,  # All outputs from TIR nodes for this ONNX node
                                    frontend_node
                                )
                                logger.info(
                                    f"[{onnx_node_name}] ✓ Validated ONNX node against {len(tir_nodes_for_onnx)} TIR node(s): "
                                    f"{', '.join(tir_nodes_for_onnx)}"
                                )
                            except DebugValidationError as e:
                                # Debug validation errors (shape/value mismatches) should stop execution
                                logger.error(
                                    f"Debug validation failed for ONNX node {onnx_node_name} "
                                    f"(TIR nodes: {', '.join(tir_nodes_for_onnx)}): {e}. "
                                    f"Stopping execution in debug mode."
                                )
                                raise  # Re-raise to stop execution
                            except Exception as e:
                                # Other exceptions (e.g., ONNX Runtime errors) are logged but don't stop execution
                                logger.error(
                                    f"Debug comparison failed for ONNX node {onnx_node_name} "
                                    f"(TIR nodes: {', '.join(tir_nodes_for_onnx)}): {e}. "
                                    f"This may indicate an issue with ONNX Runtime or model setup."
                                )
                
                # Garbage collection: remove activations no longer needed
                if enable_gc and still_needed_by is not None:
                    out_op_id = node.outputs[0] if node.outputs else None
                    if out_op_id:
                        for in_op_id in node.inputs:
                            if in_op_id in still_needed_by:
                                still_needed_by[in_op_id].discard(out_op_id)
                                if len(still_needed_by[in_op_id]) == 0:
                                    # Don't delete params, constants, or initializers
                                    if (in_op_id in tensor_memory and 
                                        in_op_id not in self.params and 
                                        in_op_id not in self.constants):
                                        del tensor_memory[in_op_id]
                                        logger.debug(f"GC: Deleted activation {in_op_id}")
        
        # Log validation summary if debug mode was enabled
        if self.debug_mode and len(validated_onnx_nodes) > 0:
            logger.info(
                f"Debug validation summary: Validated {len(validated_onnx_nodes)} ONNX node(s) "
                f"against their corresponding TIR nodes"
            )
        elif self.debug_mode and len(self.onnx_node_to_tir_nodes) > 0:
            logger.warning(
                f"Debug mode enabled but no ONNX nodes were validated. "
                f"Expected {len(self.onnx_node_to_tir_nodes)} ONNX nodes to be validated."
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

