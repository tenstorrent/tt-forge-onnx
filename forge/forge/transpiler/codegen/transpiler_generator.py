"""
Enhanced TranspilerCodeGenerator for generating Python Forge module code from TIRGraph.
Matches ForgeWriter capabilities, currently supports only ONNX.
"""
from typing import Dict, List, Any, Tuple

from forge.transpiler.core.graph import TIRGraph
from forge.python_codegen import forge_df_from_str, pytorch_df_from_str


class TranspilerCodeGenerator:
    """Generates Python Forge module code from TIRGraph. Currently supports only ONNX."""
    
    def __init__(
        self,
        tir_graph: TIRGraph,
        class_name: str,
        delete_inputs = True,
    ):
        self.tir_graph = tir_graph
        self.class_name = class_name
        self.delete_inputs = delete_inputs
        self.lines = []
        self.indent = 0
        self.param_names = []
        self.const_names = []
    
    def generate(self) -> str:
        """Generate complete Python module code."""
        self.write_header()
        self.write_class_definition()
        self.write_forward()
        self.write_param_parser()
        return "\n".join(self.lines)
    
    def write_header(self):
        """Write imports and module header."""
        self.wl("import torch")
        self.wl("import forge")
        self.wl("import forge.op")
        self.wl("from forge import ForgeModule")
        self.wl("")
        self.wl("from loguru import logger")
        self.wl("")
    
    def write_class_definition(self):
        """Write ForgeModule class definition. Matches ForgeWriter.write_class_definition()."""
        self.wl("")
        self.wl(f"class {self.class_name}(ForgeModule):")
        self.indent += 1
        self.wl("def __init__(self, name):")
        self.indent += 1
        self.wl("super().__init__(name)")
        
        # Add parameters (matches ForgeWriter format)
        for name, tensor in self.tir_graph.params.items():
            if name in self.param_names:
                continue
            self.param_names.append(name)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            # Convert dtype to string format expected by forge_df_from_str
            dtype_str = pytorch_df_from_str(dtype, name)
            forge_df = forge_df_from_str(dtype_str, name, return_as_str=True)
            self.wl(
                f'self.add_parameter("{name}", '
                f'forge.Parameter(*{shape}, requires_grad=True, '
                f'dev_data_format={forge_df}))'
            )
        
        # Add constants (matches ForgeWriter format)
        for name, tensor in self.tir_graph.constants.items():
            self.const_names.append(name)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            dtype_str = pytorch_df_from_str(dtype, name)
            self.wl(
                f'self.add_constant("{name}", '
                f'shape={shape}, dtype=torch.{dtype_str})'
            )
        
        self.indent = 0
        self.wl("")
    
    def write_forward(self):
        """Write forward method. Matches ForgeWriter.write_forward()."""
        self.indent = 1
        
        # Get forward arguments (exclude params and constants)
        all_initializers = set(self.tir_graph.params.keys()) | set(self.tir_graph.constants.keys())
        forward_args = [inp for inp in self.tir_graph.inputs if inp not in all_initializers]
        
        # Format activation names like ForgeWriter
        activation_names = "".join([", " + name for name in forward_args])
        self.wl(f"def forward(self{activation_names}):")
        self.indent += 1
        
        # Get nodes in topological order
        sorted_nodes = self.tir_graph.get_topological_sort()
        
        for node in sorted_nodes:
            op_info = node.emit()
            
            # Format input names (matches ForgeWriter.get_op_input_names())
            input_names = self.get_op_input_names(op_info['input_names'])
            activation_names = "".join([", " + name for name in input_names])
            
            # Format attributes
            attrs = op_info.get('args', {})
            if len(attrs) == 0:
                arg_text = ""
            else:
                arg_text = "".join([", " + argument + "=" + value for argument, value in self._format_args(attrs)])
            
            # Write operation call (matches ForgeWriter format)
            output_name = op_info['output_name']
            function_name = op_info['function_name']
            node_name = op_info['node_name']
            src_layer = op_info.get('src_layer')
            
            # Add source layer tracking (matches ForgeWriter)
            set_src_layer = ""
            if src_layer:
                set_src_layer = f'.set_src_layer("{src_layer}")'
            
            self.wl(
                f'{output_name} = {function_name}("{node_name}"{activation_names}{arg_text}){set_src_layer}'
            )
            
            # Memory optimization: delete inputs if needed (matches ForgeWriter)
            if self.delete_inputs:
                for input_name in op_info['input_names']:
                    if input_name not in all_initializers:
                        self.wl(f"{input_name}._value = None")
        
        # Write return statement (matches ForgeWriter)
        outputs = self.tir_graph.outputs
        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)
        
        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")
    
    def write_param_parser(self):
        """Write parameter loading method. Only supports ONNX currently. Matches ForgeWriter ONNX implementation."""
        self.indent = 1
        
        # Only ONNX is supported - matches ForgeWriter.write_param_parser() for ONNX
        self.wl("def process_framework_parameters(self, model):")
        self.indent += 1
        self.wl("import onnx")
        self.wl("import onnx.numpy_helper")
        self.wl("import numpy as np")
        self.wl("weights = model.graph.initializer")
        self.wl("")
        self.wl("# Onnx will convert bfloat16 to float32 in numpy with numpy_helper call")
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
        """Format input names for function call. Matches ForgeWriter.get_op_input_names()."""
        formatted = []
        for name in input_names:
            if name in self.param_names:
                formatted.append('self.get_parameter("' + name + '")')
            elif name in self.const_names:
                formatted.append('self.get_constant("' + name + '")')
            else:
                formatted.append(name)
        return formatted
    
    def _format_args(self, attrs: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Format attributes as (argument, value) tuples for code generation."""
        formatted = []
        for k, v in attrs.items():
            if isinstance(v, str):
                formatted.append((k, f'"{v}"'))
            elif isinstance(v, (list, tuple)):
                # Format lists/tuples properly
                if isinstance(v, tuple):
                    formatted.append((k, str(v)))
                else:
                    formatted.append((k, str(list(v))))
            else:
                formatted.append((k, str(v)))
        return formatted
    
    def wl(self, line: str):
        """Write line with current indentation."""
        indent_str = "    " * self.indent
        self.lines.append(indent_str + line)

