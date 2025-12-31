# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TranspilerCodeGenerator for generating Python Forge module code from TIRGraph.
Matches ForgeWriter capabilities, currently supports only ONNX.
"""
from typing import Dict, List, Any, Tuple

from forge.transpiler.core.graph import TIRGraph
from forge.python_codegen import forge_df_from_str, pytorch_df_from_str


class TranspilerCodeGenerator:
    """
    Generate Python Forge module code from TIRGraph.

    Converts a TIRGraph into executable Python code that creates a ForgeModule.
    The generated code follows the same structure as ForgeWriter for consistency.
    Currently supports only ONNX frontend.
    """

    def __init__(
        self,
        tir_graph: TIRGraph,
        class_name: str,
        delete_inputs=True,
    ):
        """
        Initialize the code generator.

        Args:
            tir_graph: TIRGraph to convert to Python code
            class_name: Name for the generated ForgeModule class
            delete_inputs: If True, delete intermediate activations for memory optimization
        """
        self.tir_graph = tir_graph
        self.class_name = class_name
        self.delete_inputs = delete_inputs
        self.lines = []
        self.indent = 0
        self.param_names = []
        self.const_names = []

    def generate(self) -> str:
        """
        Generate complete Python module code.

        Returns:
            Complete Python code as a string
        """
        self.write_header()
        self.write_class_definition()
        self.write_forward()
        self.write_param_parser()
        return "\n".join(self.lines)

    def write_header(self):
        """Write Python imports and module header."""
        self.wl("import torch")
        self.wl("import forge")
        self.wl("import forge.op")
        self.wl("from forge import ForgeModule")
        self.wl("")
        self.wl("from loguru import logger")
        self.wl("")

    def write_class_definition(self):
        """
        Write ForgeModule class definition with __init__ method.

        Generates code to add parameters and constants to the module.
        Matches ForgeWriter.write_class_definition() format.
        """
        self.wl("")
        self.wl(f"class {self.class_name}(ForgeModule):")
        self.indent += 1
        self.wl("def __init__(self, name):")
        self.indent += 1
        self.wl("super().__init__(name)")

        # Generate parameter declarations (trainable weights)
        # Skip duplicates: same parameter may be referenced by multiple nodes
        for name, tensor in self.tir_graph.params.items():
            if name in self.param_names:
                continue
            self.param_names.append(name)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            # Convert PyTorch dtype to string format, then to Forge data format
            # This handles device-specific data format requirements
            dtype_str = pytorch_df_from_str(dtype, name)
            forge_df = forge_df_from_str(dtype_str, name, return_as_str=True)
            self.wl(
                f'self.add_parameter("{name}", '
                f"forge.Parameter(*{shape}, requires_grad=True, "
                f"dev_data_format={forge_df}))"
            )

        # Generate constant declarations (non-trainable values)
        # Constants don't need duplicate checking as they're typically unique
        for name, tensor in self.tir_graph.constants.items():
            self.const_names.append(name)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            dtype_str = pytorch_df_from_str(dtype, name)
            self.wl(f'self.add_constant("{name}", ' f"shape={shape}, dtype=torch.{dtype_str})")

        self.indent = 0
        self.wl("")

    def write_forward(self):
        """
        Write forward method with operation calls.

        Generates the forward() method that executes all operations in topological order.
        Matches ForgeWriter.write_forward() format.
        """
        self.indent = 1

        # Filter out parameters and constants from forward() arguments
        # Only actual model inputs (activations) should be forward() parameters
        # Parameters/constants are accessed via self.get_parameter()/self.get_constant()
        all_initializers = set(self.tir_graph.params.keys()) | set(self.tir_graph.constants.keys())
        forward_args = [inp for inp in self.tir_graph.inputs if inp not in all_initializers]

        activation_names = "".join([", " + name for name in forward_args])
        self.wl(f"def forward(self{activation_names}):")
        self.indent += 1

        # Execute nodes in topological order to respect data dependencies
        sorted_nodes = self.tir_graph.get_topological_sort()

        # Collect operation metadata for code generation
        ops_info = []
        for node in sorted_nodes:
            op_info = node.emit()
            ops_info.append(op_info)

        # Compute which intermediate activations can be deleted after each operation
        # This enables memory optimization by freeing unused tensors
        inputs_to_delete_map = self._compute_inputs_to_delete(ops_info, forward_args)

        for op_info in ops_info:
            input_names = self.get_op_input_names(op_info["input_names"])
            activation_names = "".join([", " + name for name in input_names])

            attrs = op_info.get("args", {})
            if len(attrs) == 0:
                arg_text = ""
            else:
                arg_text = "".join([", " + argument + "=" + value for argument, value in self._format_args(attrs)])

            output_name = op_info["output_name"]
            function_name = op_info["function_name"]
            node_name = op_info["node_name"]
            src_layer = op_info.get("src_layer")

            set_src_layer = ""
            if src_layer:
                set_src_layer = f'.set_src_layer("{src_layer}")'

            # Generate operation call: output = forge.op.OperationName("node_name", inputs..., args...)
            self.wl(f'{output_name} = {function_name}("{node_name}"{activation_names}{arg_text}){set_src_layer}')

            # Memory optimization: delete intermediate activations that are no longer needed
            # This reduces memory footprint during forward pass by freeing tensors immediately
            # after their last use, rather than waiting until end of forward()
            if self.delete_inputs:
                output_name_key = op_info["output_name"]
                if output_name_key in inputs_to_delete_map:
                    for name_to_del in inputs_to_delete_map[output_name_key]:
                        # Set tensor value to None to allow garbage collection
                        self.wl(f"{name_to_del}._value = None")

        outputs = self.tir_graph.outputs
        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)

        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")

    def _compute_inputs_to_delete(self, ops_info, forward_args):
        """
        Compute which inputs should be deleted for each operation.

        Implements reference counting to determine when intermediate activations
        are no longer needed and can be deleted for memory optimization.
        Matches TVM's delete_unneeded_outputs() logic.

        Algorithm:
        1. Count how many times each tensor is consumed (reference counting)
        2. Process operations in topological order, decrementing reference counts
        3. When a tensor's reference count reaches 0, mark it for deletion

        Only deletes intermediate activations (outputs of operations), not:
        - Model inputs (forward_args) - these are user-provided and shouldn't be deleted
        - Parameters/constants - these are persistent model state

        Args:
            ops_info: List of operation info dictionaries from node.emit() (in topological order)
            forward_args: List of model input names (for safety check)

        Returns:
            Dictionary mapping output_name -> list of input names to delete
        """
        consumers = {}
        op_outputs = set()

        # Step 1: Initialize reference counts for graph outputs (always needed)
        # Graph outputs must be preserved until the end, so they start with count=1
        for output_name in self.tir_graph.outputs:
            consumers[output_name] = 1

        # Step 2: Count total references to each tensor
        # For each operation, increment reference count for each input tensor
        for op_info in ops_info:
            output_name = op_info["output_name"]
            op_outputs.add(output_name)

            for input_name in op_info["input_names"]:
                if input_name not in consumers:
                    consumers[input_name] = 1
                else:
                    consumers[input_name] = consumers[input_name] + 1

        inputs_to_delete_map = {}

        # Step 3: Process operations in topological order and mark tensors for deletion
        # When an operation consumes a tensor, decrement its reference count
        # If count reaches 0, the tensor is no longer needed and can be deleted
        for op_info in ops_info:
            output_name = op_info["output_name"]
            inputs_to_delete = []

            for input_name in op_info["input_names"]:
                # Skip if input is not an output of another operation
                # (e.g., model inputs, parameters, constants)
                if input_name not in op_outputs:
                    continue

                # Safety check: never delete model inputs
                if input_name in forward_args:
                    continue

                # Decrement reference count as this operation consumes the tensor
                assert input_name in consumers, f"Input {input_name} not found in consumers dict"
                consumers[input_name] = consumers[input_name] - 1

                # If no more references, mark for deletion after this operation
                if consumers[input_name] == 0:
                    inputs_to_delete.append(input_name)

            if inputs_to_delete:
                inputs_to_delete_map[output_name] = inputs_to_delete

        return inputs_to_delete_map

    def write_param_parser(self):
        """
        Write parameter loading method.

        Generates code for loading parameters from framework models.
        Currently only supports ONNX. Matches ForgeWriter ONNX implementation.
        """
        self.indent = 1

        self.wl("def process_framework_parameters(self, model):")
        self.indent += 1
        self.wl("import onnx")
        self.wl("import onnx.numpy_helper")
        self.wl("import numpy as np")
        self.wl("weights = model.graph.initializer")
        self.wl("")
        self.wl("for weight in weights:")
        self.indent += 1
        self.wl("name = weight.name")
        self.wl("weight_numpy = onnx.numpy_helper.to_array(weight)")
        self.wl("tensor = torch.tensor(weight_numpy)")
        self.wl("")
        self.wl("if name in self._parameters:")
        self.indent += 1
        self.wl("tensor.requires_grad = True")
        self.wl("self.set_parameter(name, tensor)")
        self.indent -= 1
        self.wl("elif name in self._constants:")
        self.indent += 1
        # Handle scalar constants: reshape to (1, 1) for compatibility
        # ONNX scalar constants (shape=()) need to be reshaped for Forge compatibility
        # This generates: if torch.numel(tensor) == 1 and len(tensor.shape) == 0: ...
        self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
        self.indent += 1
        self.wl("tensor = tensor.reshape((1, 1))")
        self.indent -= 1
        self.wl("tensor.requires_grad = False")
        self.wl("self.set_constant(name, tensor)")
        self.indent -= 1
        self.wl("else:")
        self.indent += 1
        self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
        self.indent -= 1
        self.indent -= 1

        self.indent = 0
        self.wl("")

    def get_op_input_names(self, input_names: List[str]) -> List[str]:
        """
        Format input names for function call.

        Converts parameter/constant names to method calls, leaves activation names as-is.
        Matches ForgeWriter.get_op_input_names() format.

        Args:
            input_names: List of input tensor names

        Returns:
            List of formatted input names (strings ready for code generation)
        """
        formatted = []
        for name in input_names:
            # Parameters and constants are accessed via module methods
            # Activations are passed directly as function arguments
            if name in self.param_names:
                formatted.append('self.get_parameter("' + name + '")')
            elif name in self.const_names:
                formatted.append('self.get_constant("' + name + '")')
            else:
                # Activation tensor - use variable name directly
                formatted.append(name)
        return formatted

    def _format_args(self, attrs: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Format attributes as (argument, value) tuples for code generation.

        Converts Python values to properly formatted strings for code generation:
        - Strings: wrapped in quotes
        - Lists/Tuples: converted to string representation
        - Other types: converted via str()

        Args:
            attrs: Dictionary of attribute names to values

        Returns:
            List of (argument_name, formatted_value_string) tuples
        """
        formatted = []
        for k, v in attrs.items():
            if isinstance(v, str):
                # String values need quotes in generated code
                formatted.append((k, f'"{v}"'))
            elif isinstance(v, (list, tuple)):
                # Preserve tuple vs list distinction for code generation
                if isinstance(v, tuple):
                    formatted.append((k, str(v)))
                else:
                    formatted.append((k, str(list(v))))
            else:
                # Numeric types, bools, etc. - convert directly
                formatted.append((k, str(v)))
        return formatted

    def wl(self, line: str):
        """
        Write a line with current indentation.

        Args:
            line: Line of code to write (without indentation)
        """
        indent_str = "    " * self.indent
        self.lines.append(indent_str + line)
