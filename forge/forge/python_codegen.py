# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
import numpy as np
import tensorflow as tf
from loguru import logger
import forge
from forge.tensor import forge_dataformat_to_pytorch_dtype

from typing import Any, Dict, List, Optional, Tuple, Union


def forge_df_from_str(df: Union[str, forge.DataFormat], name: str, return_as_str: bool = True):

    dtype_str_to_forge_df = {
        "bfp2": forge.DataFormat.Bfp2,
        "bfp2_b": forge.DataFormat.Bfp2_b,
        "bfp4": forge.DataFormat.Bfp4,
        "bfp4_b": forge.DataFormat.Bfp4_b,
        "bfp8": forge.DataFormat.Bfp8,
        "bfp8_b": forge.DataFormat.Bfp8_b,
        "float16": forge.DataFormat.Float16,
        "float16_b": forge.DataFormat.Float16_b,
        "bfloat16": forge.DataFormat.Float16_b,
        "float32": forge.DataFormat.Float32,
        "int8": forge.DataFormat.Int8,
        "int32": forge.DataFormat.Int32,
        "invalid": forge.DataFormat.Invalid,
        "lf8": forge.DataFormat.Lf8,
        "raw_uint16": forge.DataFormat.RawUInt16,
        "raw_uint32": forge.DataFormat.RawUInt32,
        "raw_uint8": forge.DataFormat.RawUInt8,
        "uint16": forge.DataFormat.UInt16,
    }

    if isinstance(df, str):
        df = df.lower()
        dev_data_format = dtype_str_to_forge_df.get(df, forge.DataFormat.Float32)
        if df not in dtype_str_to_forge_df:
            logger.warning(f"Invalid data format: {df} for constant/parameter '{name}', defaulting to float32")
        if return_as_str:
            dev_data_format = "forge." + str(dev_data_format)

    elif isinstance(df, forge.DataFormat):
        forge_df_to_dtype_str = {}
        for dtype_str, forge_df in dtype_str_to_forge_df.items():
            forge_df_to_dtype_str.setdefault(forge_df, dtype_str)
        dev_data_format = forge_df_to_dtype_str.get(df, "float32")
        if df not in forge_df_to_dtype_str:
            logger.warning(f"Invalid data format: {df} for constant/parameter '{name}', defaulting to float32")

    return dev_data_format


def pytorch_df_from_str(df: Union[str, torch.dtype], name: str, return_as_str: bool = True):

    dtype_str_to_torch_dtype = {
        "float16": torch.float16,
        "float16_b": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int32": torch.int32,
        "int16": torch.int16,
        "int64": torch.int64,
        "uint1": torch.bool,
    }

    if isinstance(df, str):
        df = df.lower()
        torch_dtype = dtype_str_to_torch_dtype.get(df, torch.float32)
        if df not in dtype_str_to_torch_dtype:
            logger.warning(f"Invalid data format: {df} for constant/parameter '{name}', defaulting to float32")
        if return_as_str:
            torch_dtype = str(torch_dtype)

    elif isinstance(df, torch.dtype):
        torch_dtype_to_dtype_str = {}
        for dtype_str, torch_dtype in dtype_str_to_torch_dtype.items():
            torch_dtype_to_dtype_str.setdefault(torch_dtype, dtype_str)
        torch_dtype = torch_dtype_to_dtype_str.get(df, "float32")
        if df not in torch_dtype_to_dtype_str:
            logger.warning(f"Invalid data format: {df} for constant/parameter '{name}', defaulting to float32")

    return torch_dtype


class PythonWriter:
    def __init__(self, module_name, module_directory="generated_modules", open_file=True):
        self.filename = module_name + ".py"

        self.module_directory = module_directory
        os.makedirs(self.module_directory, exist_ok=True)
        if open_file:
            self.file = open(os.path.join(self.module_directory, self.filename), "w")
        self.indent = 0
        self.module_name = module_name
        self.class_name = module_name.title().replace("_", "")

    def wl(self, text):
        indent = self.indent * "    "
        self.file.write(indent + text + "\n")

    def breakpoint(self):
        self.wl("import pdb; pdb.set_trace()")

    def close_file(self):
        self.file.close()

    def import_module_path(self):
        return self.module_directory + f".{self.module_name}"


class ForgeWriter(PythonWriter):
    incompatible_np_float_types = [
        tf.bfloat16,
    ]

    def __init__(
        self,
        module_name,
        framework,
        module_directory="generated_modules",
        contains_incompatible_np_floats=False,
        delete_inputs=True,
        df_override_dtype=None,
    ):
        super().__init__(module_name, module_directory)

        self.framework = framework
        self.param_names = []
        self.const_names = []
        self.num_submodels = 0
        self.contains_incompatible_np_floats = contains_incompatible_np_floats
        self.delete_inputs = delete_inputs
        self.dev = "TTDevice"
        self.df_override_dtype = df_override_dtype

    def write_header(self, include_pytest_imports=False):
        self.wl("import forge")
        self.wl("import forge.op")
        self.wl("from forge import ForgeModule")
        self.wl("")

        self.wl("from loguru import logger")

        self.wl("import torch")

        if include_pytest_imports:
            self.wl("")
            self.wl("from forge import Tensor, compile")
            self.wl("from forge.verify.verify import verify")
            self.wl("from forge.verify.value_checkers import AutomaticValueChecker")
            self.wl("from forge.verify.config import VerifyConfig")
            self.wl(
                "from forge.forge_property_utils import record_forge_op_name, record_op_model_names, record_forge_op_args, record_single_op_operands_info"
            )
            self.wl("import pytest")

        if self.framework == "tensorflow":
            self.wl("import tensorflow as tf")
            self.wl("from forge.tvm_utils import map_tf_dtype_to_pt")

        if self.framework == "jax":
            self.wl("import flax")
            self.wl("import numpy as np")
            self.wl("import jax.numpy as jnp")
            self.wl("from collections.abc import MutableMapping")
            self.wl("from transformers.modeling_flax_utils import FlaxPreTrainedModel")

        self.wl("\n")

    def write_class_definition(self, params, constants, class_name=None, num_submodels=0, is_submodel=False):
        if class_name is None:
            class_name = self.class_name
        self.num_submodels = num_submodels
        self.wl("")
        self.wl(f"class {class_name}(ForgeModule):")
        self.indent += 1
        self.wl("def __init__(self, name):")
        self.indent += 1
        self.wl(f"super().__init__(name)")
        if num_submodels > 0:
            self.wl(f"self.num_layers = {num_submodels}")
            self.wl("self.layers = []")
            self.wl("for i in range(self.num_layers):")
            self.indent += 1
            self.wl('self.layers.append(Submodel(f"layer_{i}"))')
            self.indent -= 1

        for param in params.values():
            name, shape, requires_grad, dtype = param
            if name in self.param_names:
                continue
            self.param_names.append(name)
            if is_submodel:
                self.wl(
                    f'self.add_parameter("{name}", forge.Parameter(*{shape}, requires_grad={requires_grad}, dev_data_format={forge_df_from_str(dtype, name)}), prepend_name=True)'
                )
            else:
                self.wl(
                    f'self.add_parameter("{name}", forge.Parameter(*{shape}, requires_grad={requires_grad}, dev_data_format={forge_df_from_str(dtype, name)}))'
                )

        for const in constants.values():
            name = const[0]
            shape = tuple(const[1])
            dtype = pytorch_df_from_str(const[2], name)
            self.const_names.append(name)
            if is_submodel:
                self.wl(f'self.add_constant("{name}", prepend_name=True, shape={shape}, dtype={dtype})')
            else:
                self.wl(f'self.add_constant("{name}", shape={shape}, dtype={dtype})')

        self.indent = 0
        self.wl("")

    def get_op_input_names(self, op):
        input_names = []
        for name in op.input_names:
            if name in self.param_names:
                input_names.append('self.get_parameter("' + name + '")')
            elif name in self.const_names:
                input_names.append('self.get_constant("' + name + '")')
            else:
                input_names.append(name)

        return input_names

    def write_forward(self, ops, inputs, outputs):
        self.indent = 1
        activation_names = "".join([", " + name for name in [inputs[key] for key in sorted(inputs)]])
        self.wl("def forward(self" + activation_names + "):")
        self.indent += 1

        for key in sorted(ops):
            input_names = self.get_op_input_names(ops[key])
            activation_names = "".join([", " + name for name in input_names])
            if ops[key].is_submodule_call:
                activation_names = activation_names.lstrip(", ")
            if len(ops[key].args) == 0:
                arg_text = ""
            else:
                arg_text = "".join([", " + argument + "=" + value for argument, value in ops[key].args])

            set_src_layer = ""
            if ops[key].src_layer:
                set_src_layer = f'.set_src_layer("{ops[key].src_layer}")'
            if ops[key].is_submodule_call:
                if len(ops[key].loop_with):
                    if len(ops[key].loop_with) + 1 == self.num_submodels:
                        loop_len = "self.num_layers"
                    else:
                        if ops[key].loop_start_index == 0:
                            loop_len = f"{len(ops[key].loop_with) + 1}"
                        else:
                            loop_len = f"{ops[key].loop_start_index}, {len(ops[key].loop_with) + ops[key].loop_start_index + 1}"

                    self.wl(f"for i in range({loop_len}):")  # +1 for current op
                    self.indent += 1
                    self.wl(
                        f"{ops[key].output_name} = {ops[key].function_name}({activation_names}{arg_text}){set_src_layer}"
                    )
                    self.indent -= 1
                else:
                    self.wl(
                        f"{ops[key].output_name} = {ops[key].function_name}({activation_names}{arg_text}){set_src_layer}"
                    )
            else:
                self.wl(
                    f'{ops[key].output_name} = {ops[key].function_name}("{ops[key].node_name}"{activation_names}{arg_text}){set_src_layer}'
                )
                if self.delete_inputs:
                    for name_to_del in ops[key].inputs_to_delete:
                        self.wl(f"{name_to_del}._value = None")

        outputs = list(outputs.values())
        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)

        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")

    def write_param_parser(
        self, param_names, param_file_name, named_params_file_name=None, named_buffers_file_name=None
    ):
        self.indent = 1

        if self.framework == "pytorch" or self.framework == "paddle":
            # Case 1: No parameter or buffer files provided. Extract parameters and buffers
            # directly from the model.
            if not named_params_file_name and not named_buffers_file_name:
                self.wl(f"def process_framework_parameters(self, model):")
                self.indent += 1
                self.wl(f"named_parameters = dict(model.state_dict().items())")
                if param_file_name is not None:
                    self.wl(f'serialized_params = torch.load("{param_file_name}")')
                    self.wl(f"named_parameters.update(serialized_params)")
                self.wl("named_buffers = dict(model.named_buffers())")
                self.wl("named_parameters.update(named_buffers)")
            # Case 2: Parameter and buffer files provided. Load parameters and buffers from
            # the serialized files.

            elif named_params_file_name and named_buffers_file_name:
                self.wl(f"def process_framework_parameters(self):")
                self.indent += 1
                self.wl(f"named_parameters = torch.load('{named_params_file_name}')")
                if param_file_name is not None:
                    self.wl(f'serialized_params = torch.load("{param_file_name}")')
                    self.wl(f"named_parameters.update(serialized_params)")
                self.wl(f"named_buffers = torch.load('{named_buffers_file_name}')")
                self.wl("named_parameters.update(named_buffers)")
            else:
                assert False, "Invalid combination of param files (either both or none)"

            if len(param_names):
                self.wl("flattened_to_hierarchical_map = {")
                self.indent += 1
                for k, v in param_names.items():
                    self.wl(f"'{k}' : {v},")
                self.indent -= 1
                self.wl("}")
            self.wl("for name, torch_param in named_parameters.items():")
            self.indent += 1
            if self.framework == "paddle":
                self.wl("if hasattr(torch_param, 'name') and torch_param.name is not None:")
                self.indent += 1
                self.wl("name = torch_param.name")
                self.indent -= 1
                self.wl("tensor = torch.tensor(torch_param.data.numpy())")

            else:
                # Handle -inf and inf values
                self.wl("# Replace infinities with relevant numbers")
                self.wl("if torch.any(torch.isinf(torch_param)):")
                self.indent += 1
                self.wl(
                    "torch_param = torch.where(torch.isposinf(torch_param), torch.tensor(1e4, dtype=torch_param.dtype), torch_param)"
                )
                self.wl(
                    "torch_param = torch.where(torch.isneginf(torch_param), torch.tensor(-1e4, dtype=torch_param.dtype), torch_param)"
                )
                self.wl('logger.warning(f"Replacing -inf and inf values in tensor param: {name}")')
                self.indent -= 1

                self.wl("tensor = torch_param.data")

            if len(param_names):
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")

                # If name in parameter dictionary
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                # If name in constant dictionary
                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
                self.indent -= 1

                self.indent -= 1
                self.wl("else:")
                self.indent += 1
            self.wl("if name in self._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = torch.is_floating_point(tensor)")
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

        elif self.framework == "tensorflow":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"weights = model.weights")
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            if self.contains_incompatible_np_floats:
                self.wl(f"incompatible_np_float_types = {ForgeWriter.incompatible_np_float_types}")

            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.path")
            if self.contains_incompatible_np_floats:
                self.wl(
                    "# Some floating-point weights in the model havea a dtype that is incompatible with the .numpy() call."
                )
                self.wl("if weight.dtype in incompatible_np_float_types:")
                self.indent += 1
                self.wl("dtype = map_tf_dtype_to_pt(weight.dtype)")
                self.wl("weight = tf.cast(weight, tf.float32)")
                self.wl("tensor = torch.tensor(weight.numpy(), dtype=dtype)")
                self.indent -= 1
                self.wl("else:")
                self.indent += 1
                self.wl("tensor = torch.tensor(weight.numpy())")
                self.indent -= 1
            else:
                self.wl("tensor = torch.tensor(weight.numpy())")

            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")

            # If name in parameter dictionary
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
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
            self.indent -= 1

            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl(f"for name, torch_param in serialized_params.items():")
                self.indent += 1
                self.wl("tensor = torch_param.data")
                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
                self.indent -= 1
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
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
                self.indent -= 1
        elif self.framework == "tf_graphdef":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            self.wl(f'serialized_params = torch.load("{param_file_name}")')
            self.wl(f"for name, torch_param in serialized_params.items():")
            self.indent += 1
            self.wl("tensor = torch_param.data")
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
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
            self.indent -= 1

        elif self.framework == "onnx":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"import onnx")
            self.wl(f"import onnx.numpy_helper")
            self.wl(f"import numpy as np")
            self.wl(f"weights = model.graph.initializer")
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            # Onnx will convert bfloat16 to float32 in numpy with numpy_helper call
            self.wl("# Onnx will convert bfloat16 to float32 in numpy with numpy_helper call")

            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.name")
            self.wl("weight_numpy = onnx.numpy_helper.to_array(weight)")

            self.wl("tensor = torch.tensor(weight_numpy)")

            # onnx.numpy_helper.to_array() returns a float32 NumPy array even when the
            # ONNX model stores bfloat16 weights, because NumPy has no bfloat16 dtype.
            # When a lower-precision override is active we must cast the float32 tensor to
            # the requested dtype so the ForgeModule parameters match the compiled model.
            if self.df_override_dtype is not None:
                self.wl("if torch.is_floating_point(tensor):")
                self.indent += 1
                self.wl(f"tensor = tensor.to({str(self.df_override_dtype)})")
                self.indent -= 1

            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
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
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl(f"for name, torch_param in serialized_params.items():")
                self.indent += 1
                self.wl("tensor = torch_param.data")
                if self.df_override_dtype is not None:
                    self.wl("if torch.is_floating_point(tensor):")
                    self.indent += 1
                    self.wl(f"tensor = tensor.to({str(self.df_override_dtype)})")
                    self.indent -= 1
                self.wl("if name in flattened_to_hierarchical_map:")
                self.indent += 1
                self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
                self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
                self.indent += 1
                self.wl("tensor.requires_grad = True")
                self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
                self.indent -= 1

                self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
                self.indent += 1
                self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
                self.indent += 1
                self.wl("tensor = tensor.reshape((1, 1))")
                self.indent -= 1
                self.wl("tensor.requires_grad = False")
                self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
                self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
                self.indent -= 1
                self.indent -= 1

                self.wl("else:")
                self.indent += 1
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
                self.indent -= 1

        elif self.framework == "jax":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"def flatten_params(params, parent_key='', sep='.'):")
            self.indent += 1
            self.wl(f"items = []")
            self.wl(f"for key, val in params.items():")
            self.indent += 1
            self.wl(f"new_key = parent_key + sep + key if parent_key else key")
            self.wl(f"if isinstance(val, MutableMapping):")
            self.indent += 1
            self.wl(f"items.extend(flatten_params(val, new_key, sep=sep).items())")
            self.indent -= 1
            self.wl(f"else:")
            self.indent += 1
            self.wl(f"items.append((new_key, val))")
            self.indent -= 1
            self.indent -= 1
            self.wl(f"return dict(items)")
            self.indent -= 1
            self.wl(f"if isinstance(model, FlaxPreTrainedModel):")
            self.indent += 1
            self.wl(f"model_params = model.params")
            self.indent -= 1
            self.wl(f"elif isinstance(model, flax.linen.Module):")
            self.indent += 1
            self.wl("model_params = {}")
            self.wl("if hasattr(model, 'variables') and 'params' in model.variables:")
            self.indent += 1
            self.wl(f"model_params = model.variables['params']")
            self.indent -= 1
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl(f"for key, val in serialized_params.items():")
                self.indent += 1
                self.wl(f"model_params[key] = jnp.array(val.data.numpy())")
                self.indent -= 1
            self.wl(f"model_params = flatten_params(model_params)")
            self.wl(f"for key, value in model_params.items():")
            self.indent += 1
            self.wl(f"name = key")
            self.wl(f"tensor = torch.Tensor(np.array(value))")

            self.wl(f"if name in self._parameters:")
            self.indent += 1
            self.wl(f"tensor.requires_grad = True")
            self.wl(f"self.set_parameter(name, tensor)")
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
            self.wl(f"else:")
            self.indent += 1
            self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
            self.indent -= 1
            self.indent -= 1
            self.indent -= 1
        elif self.framework == "tflite":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("flattened_to_hierarchical_map = {")
            self.indent += 1
            for k, v in param_names.items():
                self.wl(f"'{k}' : {v},")
            self.indent -= 1
            self.wl("}")

            self.wl(f'serialized_params = torch.load("{param_file_name}")')
            self.wl(f"for name, torch_param in serialized_params.items():")
            self.indent += 1
            self.wl("tensor = torch_param.data")
            self.wl("if name in flattened_to_hierarchical_map:")
            self.indent += 1
            self.wl("layer_name, param_name = flattened_to_hierarchical_map[name]")
            self.wl("if param_name in self.get_submodules()[layer_name]._parameters:")
            self.indent += 1
            self.wl("tensor.requires_grad = True")
            self.wl("self.get_submodules()[layer_name].set_parameter(param_name, tensor)")
            self.indent -= 1

            self.wl("elif param_name in self.get_submodules()[layer_name]._constants:")
            self.indent += 1
            self.wl("if torch.numel(tensor) == 1 and len(tensor.shape) == 0:")
            self.indent += 1
            self.wl("tensor = tensor.reshape((1, 1))")
            self.indent -= 1
            self.wl("tensor.requires_grad = False")
            self.wl("self.get_submodules()[layer_name].set_constant(param_name, tensor)")
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
            self.wl('logger.warning(f"{name} not found in self._parameters and self._constants")')
            self.indent -= 1
            self.indent -= 1

            self.wl("else:")
            self.indent += 1
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
            self.indent -= 1
        else:
            assert False, "TODO: Add other framework param parsers"

    def write_pytest_function(
        self,
        forge_module_names: List[str],
        pytest_input_shapes_and_dtypes_list: List[List[Union[Tuple[Any, ...], str]]],
        markers: Optional[List[str]] = None,
        module_metadata: Optional[Dict[str, Any]] = None,
        pytest_metadata_list: Optional[List[Dict[str, Any]]] = None,
        use_ids_function: bool = False,
        exclude_record_property: Optional[List[str]] = None,
        pytest_markers_with_reasons: Optional[List[List[Dict[str, Any]]]] = None,
    ):
        """
        Generates a pytest function that tests modules with input shapes and data types.

        This function writes a pytest function that:
        1. Creates a list of forge module names, their associated input shapes, and data types and metadata (i.e model and variant name which the shape and dtype belongs to) into a pytest parameter list.
        2. Creates a record_property fixtures for the metadata that are passed from the pytest parameter and for module_metadata argument directly intialize with property name and value
        2. Creates inputs(i.e TensorFromPyTorch) for the forge module by calling the create_from_shape Tensor class method with shapes and dtypes from the pytest parameter.
        3. Initializes the framework model using the forge module from the pytest parameter and call the `process_framework_parameters` function for module.
        4. Runs the framework model with the created inputs.
        5. Compiles the framework model.
        6. Runs the compiled model with the same inputs.
        7. Asserts that the outputs of the framework model and the compiled model are similar within a specified tolerance.

        Args:
            forge_module_names (List[str]): List of names of the modules to be tested, each corresponding to a forge module.
            pytest_input_shapes_and_dtypes_list (List[List[Union[Tuple[Any, ...], str]]]): A list of input shapes and corresponding data types for each module. Each tuple contains the shape and dtype to be tested.
            markers (Optional[List[str]]): A list of pytest markers that will be added above the test function.
            module_metadata (Optional[Dict[str, Any]]): A dictionary containing metadata about the test function. Each key-value pair represents a metadata property name and its corresponding value, which will be recorded using the `record_property` pytest fixtures.
            pytest_metadata_list (Optional[List[Dict[str, Any]]]): A list of dictionaries containing metadata for each pytest parameter.
            use_ids_function(bool): If set, the forge module name and shapes and dtyes will used as id for the pytest parameter.
            exclude_record_property(Optional[List[str]]): A list of pytest metadata property which will be excluded in forge_property_recorder fixtures(i.e pcc)
            pytest_markers_with_reasons(Optional[List[List[Dict[str, Any]]]]): A list of pytest markers with reason to add in the tests parameter in the forge_modules_and_shapes_dtypes_list.
        """
        self.wl("")
        self.wl("")
        if use_ids_function:
            self.wl("def ids_func(param):")
            self.indent += 1
            self.wl("forge_module = param[0]")
            self.wl("shapes_dtypes = param[1]")
            self.wl('return str(forge_module.__name__) + "-" + str(shapes_dtypes)')
            self.indent -= 1
            self.wl("")
        self.wl("forge_modules_and_shapes_dtypes_list = [")
        self.indent += 1
        is_pytest_metadata_list_empty = False
        if pytest_metadata_list is None or len(pytest_metadata_list) == 0:
            pytest_metadata_list = [None] * len(pytest_input_shapes_and_dtypes_list)
            is_pytest_metadata_list_empty = True
        if pytest_markers_with_reasons is None:
            pytest_markers_with_reasons = [None] * len(pytest_input_shapes_and_dtypes_list)
        for forge_module_name, pytest_input_shapes_and_dtypes, pytest_metadata, markers_with_reasons in zip(
            forge_module_names, pytest_input_shapes_and_dtypes_list, pytest_metadata_list, pytest_markers_with_reasons
        ):
            pytest_input_shapes_and_dtypes = [
                (shape, pytorch_df_from_str(dtype, "", return_as_str=False))
                for shape, dtype in pytest_input_shapes_and_dtypes
            ]
            if pytest_metadata is None:
                test_param = f"({forge_module_name}, {pytest_input_shapes_and_dtypes})"
            else:
                test_param = f"({forge_module_name}, {pytest_input_shapes_and_dtypes}, {pytest_metadata})"

            if markers_with_reasons is not None:
                marker_str_list = []
                for marker_with_reason in markers_with_reasons:
                    marker_str = f'pytest.mark.{marker_with_reason["marker_name"]}'
                    marker_reason = marker_with_reason["reason"]
                    if marker_reason is not None:
                        marker_str += f'(reason="{marker_reason}")'
                    marker_str_list.append(marker_str)
                marker_str = ", ".join(marker_str_list)
                self.wl(f"pytest.param({test_param}, marks=[{marker_str}]), ")
            else:
                self.wl(f"{test_param}, ")

        self.indent -= 1
        self.wl("]")
        if markers is not None:
            for marker in markers:
                self.wl(f"@pytest.mark.{marker}")
        if use_ids_function:
            self.wl(
                '@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)'
            )
        else:
            self.wl('@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list)')
        self.wl('@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])')
        self.wl("def test_module(forge_module_and_shapes_dtypes, training_test):")
        self.indent += 1
        if module_metadata is not None:
            for metadata_name, metadata_value in module_metadata.items():
                if metadata_name == "forge_op_name":
                    self.wl("")
                    self.wl(f'record_forge_op_name("{metadata_value}")')
        self.wl("")
        if is_pytest_metadata_list_empty:
            self.wl("forge_module, operand_shapes_dtypes = forge_module_and_shapes_dtypes")
        else:
            self.wl("forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes")
            if exclude_record_property is not None and len(exclude_record_property) != 0:
                self.wl("")
                for exclude_metadata in exclude_record_property:
                    self.wl(f'{exclude_metadata} = metadata.get("{exclude_metadata}")')
            self.wl("")
            self.wl("for metadata_name, metadata_value in metadata.items():")
            self.indent += 1
            if exclude_record_property is not None and len(exclude_record_property) != 0:
                self.wl(f"if metadata_name in {exclude_record_property}:")
                self.indent += 1
                self.wl("continue")
                self.indent -= 1
                self.wl('elif metadata_name == "model_names":')
            else:
                self.wl('if metadata_name == "model_names":')
            self.indent += 1
            self.wl("record_op_model_names(metadata_value)")
            self.indent -= 1
            self.wl('elif metadata_name == "args":')
            self.indent += 1
            self.wl("record_forge_op_args(metadata_value)")
            self.indent -= 1
            self.wl("else:")
            self.indent += 1
            self.wl(
                'logger.warning("No utility function available in forge property handler to record %s property", metadata_name)'
            )
            self.indent -= 2
        self.wl("")
        if is_pytest_metadata_list_empty or (
            exclude_record_property is not None
            and len(exclude_record_property) != 0
            and "pcc" not in exclude_record_property
        ):
            self.wl("pcc = 0.99")
        if is_pytest_metadata_list_empty or (
            exclude_record_property is not None
            and len(exclude_record_property) != 0
            and "max_int" not in exclude_record_property
        ):
            self.wl("max_int = 1000")
        self.wl(
            "inputs = [Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad = training_test) for operand_shape, operand_dtype in operand_shapes_dtypes]"
        )
        self.wl("")
        self.wl(f"framework_model = forge_module(forge_module.__name__)")
        self.wl("")
        self.wl("for name, parameter in framework_model._parameters.items():")
        self.indent += 1
        self.wl(
            "parameter_tensor = Tensor.create_torch_tensor(shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int, requires_grad = training_test)"
        )
        self.wl("framework_model.set_parameter(name, parameter_tensor)")
        self.indent -= 1
        self.wl("")
        self.wl("for name, constant in framework_model._constants.items():")
        self.indent += 1
        self.wl(
            "constant_tensor = Tensor.create_torch_tensor(shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int, requires_grad = training_test)"
        )
        self.wl("framework_model.set_constant(name, constant_tensor)")
        self.indent -= 1
        self.wl("")
        if module_metadata is not None and len(module_metadata) != 0:
            self.wl("record_single_op_operands_info(framework_model, inputs)")
            self.wl("")
        self.wl("compiler_cfg = forge.config.CompilerConfig()")
        self.wl('if "default_df_override" in metadata.keys():')
        self.indent += 1
        self.wl('compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])')
        self.indent -= 1
        self.wl("")
        self.wl(
            "compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)"
        )
        self.wl("")
        self.wl(
            "verify(inputs, framework_model, compiled_model, with_backward=training_test, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))"
        )
        self.wl("")
        self.wl("")
        self.indent -= 1


class PyTorchWriter(PythonWriter):
    incompatible_np_float_types = [
        tf.bfloat16,
    ]

    def __init__(self, module_name, source_framework):
        super().__init__(module_name)

        self.framework = source_framework
        self.param_names = []
        self.const_names = []
        self.num_submodels = 0
        self.dev = "CPUDevice"
        self.submodules = []

    def write_header(self):
        self.wl("import torch")
        self.wl("from torch import nn")
        self.wl("\n")

    def clean_name(self, name):
        new_name = []
        for mod in name.split("."):
            if mod.isdigit():
                new_name.append("l" + mod)
            else:
                new_name.append(mod)
        new_name = ".".join(new_name)

        return new_name.replace("/", ".").replace(":", "_")

    def write_class_definition(self, params, constants, class_name=None, num_submodels=0, is_submodel=False):
        if class_name is None:
            class_name = self.class_name
        self.num_submodels = num_submodels
        self.wl(f"class {class_name}(nn.Module):")
        self.indent += 1
        self.wl("def __init__(self):")
        self.indent += 1
        self.wl(f"super().__init__()")

        for param in params.values():
            full_name, shape, requires_grad, dtype = param
            if dtype == "uint1":
                dtype = "bool"

            name = self.clean_name(full_name)
            self.param_names.append(name)
            prefix = ""
            while "." in name:
                mod, _, name = name.partition(".")
                submodule = prefix + f"{mod}"
                if submodule not in self.submodules:
                    self.wl(f"self.{submodule} = nn.Module()")
                    self.submodules.append(submodule)
                prefix = submodule + "."

            mod = self.clean_name(full_name).rpartition(name)[0]
            self.wl(
                f'self.{mod}register_parameter("{name}", nn.Parameter(torch.zeros(*{shape}, dtype={pytorch_df_from_str(dtype, name)}), requires_grad={requires_grad}))'
            )

        for const in constants.values():
            full_name, shape, dtype = const
            if dtype == "uint1":
                dtype = "bool"

            name = self.clean_name(full_name)
            self.const_names.append(name)
            prefix = ""
            while "." in name:
                mod, _, name = name.partition(".")
                submodule = prefix + f"{mod}"
                if submodule not in self.submodules:
                    self.wl(f"self.{submodule} = nn.Module()")
                    self.submodules.append(submodule)
                prefix = submodule + "."

            mod = self.clean_name(full_name).rpartition(name)[0]
            self.wl(
                f'self.{mod}register_buffer("{name}", torch.zeros(*{shape}, dtype={pytorch_df_from_str(dtype, name)}))'
            )

        self.indent = 0
        self.wl("")

    def get_op_output_structure(self, op):
        """
        Defines operand output structure.

        Args:
            op (Operation): Reference operation

        Returns:
            str: Operation output structure
        """
        op_out_structure = op.output_name

        # Handle special case PyTorch outputs
        if op.function_name == "torch.max":
            op_out_structure += ", _"

        return op_out_structure

    def get_op_input_arguments(self, op):
        """
        Constructs operand inputs in expected order.

        As TVM recognize differences between operand inputs
        and attributes (unlike PyTorch), we reconstruct those
        within PyTorchWriter and generate ordered arguments
        which can be used for relevant PyTorch function/operand.

        Args:
            op (Operation): Reference operation

        Returns:
            str: Ordered function/operation attributes
        """
        op_arguments = []

        assert type(op.args) is dict, "Invalid op input type"

        # Structure TVM inputs
        for name in op.args["inp"]:
            if self.clean_name(name) in self.param_names:
                op_arguments.append(f'self.get_parameter("{self.clean_name(name)}")')
            elif self.clean_name(name) in self.const_names:
                op_arguments.append(f'self.get_buffer("{self.clean_name(name)}")')
            else:
                op_arguments.append(name)

        # Handle special case TVM inputs - concatenate
        if op.function_name == "torch.cat":
            op_arguments = ["(" + ", ".join([arg for arg in op_arguments]) + ")"]

        # Handle special case TVM inputs - stack
        if op.function_name == "torch.stack":
            op_arguments = ["(" + ", ".join([arg for arg in op_arguments]) + ",)"]

        # Handle special case TVM inputs - where
        if op.function_name == "torch.where":
            # Handle when where represents adv_index op
            if "adv_index" in op.output_name:
                # Reverse order of arguments
                op_arguments = op_arguments[::-1]
                # Add empty tensor as last argument
                op_arguments.append(f"torch.zeros_like({op_arguments[0]}, dtype=torch.float32)")

            op_arguments[0] = op_arguments[0] + ".bool()"

        if op.function_name == "torch.index_select" and len(op_arguments) == 2:
            op_arguments[-1] = f"torch.squeeze({op_arguments[-1]}).long()"

        # Handle special case TVM inputs - broadcast_to_like
        if op.function_name == "torch.broadcast_to_like":
            op.function_name = "torch.broadcast_to"
            op_arguments[-1] = op_arguments[-1] + ".shape"

        # Structure TVM attributes
        for attr_name, attr_info in op.args["attr"].items():
            if "as_named" in attr_info:
                op_arguments.insert(attr_info["inp_pos"], f"{attr_name}={attr_info['val']}")
            else:
                op_arguments.insert(attr_info["inp_pos"], f"{attr_info['val']}")

        # Construct operand arguments
        op_arguments = "".join([", " + name for name in op_arguments]).lstrip(", ")

        return op_arguments

    def _write_determine_shape_for_batch_dim(self):
        self.indent = 1
        self.wl(f"def determine_shape_for_batch_dim(self, old_shape, batch_dim):")
        self.indent += 1
        self.wl("old_shape = list(old_shape)")
        self.wl(f"if old_shape[0] == batch_dim:")
        self.indent += 1
        self.wl("return old_shape")
        self.indent -= 1
        self.wl("")

        self.wl(f"if old_shape[0] != 1:")
        self.indent += 1
        self.wl("old_shape.insert(0, 1)")
        self.indent -= 1
        self.wl("")
        self.wl(f"new_shape = [batch_dim]")
        self.wl(f"extracted_batch_dim = False")
        self.wl(f"for i in range(1, len(old_shape)):")
        self.indent += 1
        self.wl(f"if not extracted_batch_dim:")
        self.indent += 1
        self.wl(f"new_shape.append(old_shape[i] // batch_dim)")
        self.wl(f"extracted_batch_dim = True")
        self.indent -= 1
        self.wl(f"else:")
        self.indent += 1
        self.wl(f"new_shape.append(old_shape[i])")
        self.indent -= 1
        self.indent -= 1

        self.wl(f"return new_shape")
        self.indent = 0
        self.wl("")

    def write_forward(self, ops, inputs, outputs, force_batch_dim_outputs=[]):

        if len(force_batch_dim_outputs):
            self._write_determine_shape_for_batch_dim()

        self.indent = 1
        activation_names = "".join([", " + name for name in [inputs[key] for key in sorted(inputs)]])

        self.wl("def forward(self" + activation_names + "):")
        self.indent += 1

        for key in sorted(ops):
            op_attributes = self.get_op_input_arguments(ops[key])
            op_out_structure = self.get_op_output_structure(ops[key])

            if ops[key].args["inplace"]:
                op_attributes = op_attributes.split(", ")
                lhs = op_attributes[0]
                rhs = ", ".join(op_attributes[1:])
                self.wl(f"{op_out_structure} = {lhs}.{ops[key].function_name}({rhs})")
            else:
                self.wl(f"{op_out_structure} = {ops[key].function_name}({op_attributes})")

        outputs = list(outputs.values())
        self.wl("")

        if len(force_batch_dim_outputs):
            self.wl("# Batch dim handling")
            # Force reshape to expose batch dimension
            self.wl(f"batch_dim = {inputs[0]}.shape[0]")
            for name in force_batch_dim_outputs:
                output_idx = outputs.index(name)
                self.wl(
                    f"{outputs[output_idx]} = torch.reshape({name}, self.determine_shape_for_batch_dim({name}.shape, batch_dim))"
                )

            self.wl("")

        if len(outputs) == 1:
            output_names = outputs[0]
        else:
            output_names = ", ".join(outputs)

        self.wl(f"return {output_names}")
        self.indent = 0
        self.wl("")

    def write_param_parser(self, param_names, param_file_name):
        self.indent = 1

        if self.framework == "pytorch":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1

            self.wl("named_parameters = dict(model.named_parameters())")
            self.wl("my_params = dict(self.named_parameters())")
            self.wl("for name, parameter in named_parameters.items():")
            self.indent += 1
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl("if name in my_params:")
            self.indent += 1
            self.wl('module_path, _, param_name = name.rpartition(".")')
            self.wl("setattr(self.get_submodule(module_path), param_name, parameter)")
            self.indent -= 1
            self.indent -= 1
            self.wl("named_buffers = dict(model.named_buffers())")
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl("named_buffers.update(serialized_params)")

            self.wl("self.load_state_dict(named_buffers, strict=False)")
            self.indent = 0
        elif self.framework == "onnx":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1

            self.wl(f"import onnx")
            self.wl(f"import onnx.numpy_helper")
            self.wl(f"import numpy as np")
            self.wl(f"weights = model.graph.initializer")
            self.wl("named_parameters = dict(self.state_dict())")
            self.wl("named_buffers = dict(self.named_buffers())")
            self.wl("for weight in weights:")
            self.indent += 1
            self.wl("name = weight.name")
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl("weight_numpy = onnx.numpy_helper.to_array(weight)")
            self.wl("tensor = torch.tensor(weight_numpy)")
            # self.breakpoint()
            self.wl(
                "tensor.requires_grad = issubclass(weight_numpy.dtype.type, np.floating) or issubclass(weight_numpy.dtype.type, np.complex)"
            )

            self.wl("if name in named_parameters:")
            self.indent += 1
            self.wl('module_path, _, param_name = name.rpartition(".")')
            self.wl("getattr(self.get_submodule(module_path), param_name).data = tensor")
            self.indent -= 1
            self.wl("named_parameters[name] = tensor")

            self.indent -= 1

            if param_file_name is not None:
                self.wl("named_parameters.update(named_buffers)")
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl("named_parameters.update(serialized_params)")

            self.wl("self.load_state_dict(named_parameters, strict=False)")

        elif self.framework == "tensorflow":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            self.wl(f"for weight in model.weights:")
            self.indent += 1
            self.wl(f"name = weight.name.replace('/', '.').replace(':', '_')")
            self.wl("name = '.'.join(['l' + mod if mod.isdigit() else mod for mod in name.split('.')])")
            self.wl(f"value = nn.Parameter(torch.from_numpy(weight.numpy()))")
            self.wl("named_parameters[name] = value\n")
            self.indent -= 1
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace('/', '.').replace(':', '_')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")
        elif self.framework == "jax":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl(f"import flax")
            self.wl(f"import numpy as np")
            self.wl(f"import jax.numpy as jnp")
            self.wl(f"from collections.abc import MutableMapping")
            self.wl(f"from transformers.modeling_flax_utils import FlaxPreTrainedModel")
            self.wl(f"def flatten_params(params, parent_key='', sep='.'):")
            self.indent += 1
            self.wl(f"items = []")
            self.wl(f"for key, val in params.items():")
            self.indent += 1
            self.wl(f"new_key = parent_key + sep + key if parent_key else key")
            self.wl(f"if isinstance(val, MutableMapping):")
            self.indent += 1
            self.wl(f"items.extend(flatten_params(val, new_key, sep=sep).items())")
            self.indent -= 1
            self.wl(f"else:")
            self.indent += 1
            self.wl(f"items.append((new_key, val))")
            self.indent -= 1
            self.indent -= 1
            self.wl(f"return dict(items)")
            self.indent -= 1
            self.wl(f"if isinstance(model, FlaxPreTrainedModel):")
            self.indent += 1
            self.wl(f"model_params = model.params")
            self.indent -= 1
            self.wl(f"elif isinstance(model, flax.linen.Module):")
            self.indent += 1
            self.wl("model_params = {}")
            self.wl("if hasattr(model, 'params'):")
            self.indent += 1
            self.wl(f"model_params = model.variables['params']")
            self.indent -= 1
            self.indent -= 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl(f"for key, val in serialized_params.items():")
                self.indent += 1
                self.wl(f"named_parameters[key] = val")
                self.indent -= 1
            self.wl(f"module_params = flatten_params(model_params)")
            self.wl(f"for key, value in module_params.items():")
            self.indent += 1
            self.wl(f"name = key")
            self.wl(f"value = nn.Parameter(torch.from_numpy(np.array(value)))")
            self.wl(f"value.requires_grad = True")
            self.wl(f"named_parameters[name] = value")
            self.indent -= 1
            self.wl(f"self.load_state_dict(named_parameters, strict=False)")
            self.indent -= 1
            self.indent -= 1

        elif self.framework == "tf_graphdef":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace('/', '.').replace(':', '_')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")

        elif self.framework == "tflite":
            self.wl(f"def process_framework_parameters(self, model):")
            self.indent += 1
            self.wl("named_parameters = {}")
            if param_file_name is not None:
                self.wl(f'serialized_params = torch.load("{param_file_name}")')
                self.wl("serialized_params_cleaned = {}")
                self.wl(f"for key, value in serialized_params.items():")
                self.indent += 1
                self.wl(f"name = key.replace('/', '.').replace(':', '_')")
                self.wl(f"serialized_params_cleaned[name] = value")
                self.indent -= 1
                self.wl("named_parameters.update(serialized_params_cleaned)\n")
            self.wl("self.load_state_dict(named_parameters, strict=False)")
