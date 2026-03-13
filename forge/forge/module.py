# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, TypeAlias, Union
from collections import OrderedDict
import itertools

import flax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import numpy_helper
import paddle
import torch
import tensorflow as tf
from loguru import logger
from transformers import FlaxPreTrainedModel

import forge
from .forgeglobal import lazy_trace_data
from .tensor import (
    SomeTensor,
    Tensor,
    to_pt_tensor,
    to_pt_tensors,
    to_tf_tensors,
    to_tf_variables,
    to_pd_tensors,
    pytorch_dtype_to_forge_dataformat,
    forge_dataformat_to_pytorch_dtype,
)
from .parameter import Parameter

from forge.tvm_utils import map_pt_dtype_to_tf, flatten_structured_output


class Module(ABC):
    """
    Module class is used to abstract different types of modules (PyTorch, TF, etc.) and provide a common interface for them.

    """

    def __init__(self, name: str):
        self.name: str = name
        self.input_names = []
        self.is_loss = False

    def __repr__(self):
        ret = "Module " + self.name
        return ret

    def get_name(self) -> str:
        """
        Returns the name of the module.

        Returns
        -------
        Optional[Device]
            Device, or None if op has not been placed yet
        """
        return self.name

    def __getstate__(self):
        return self.__dict__.copy()

    @abstractmethod
    def get_parameters(self) -> List[Parameter]:
        """
        Return the list of parameters defined in this module

        Returns
        -------
        List[Parameter]
            List of all parameters in this module
        """


class PyTorchModule(Module):
    """
    A wrapper around a PyTorch module. If placed on a CPU device, PyTorchModules will be executed as is, and if placed
    on a TT device, modules will be lowered to Forge.
    """

    def __init__(self, name: str, module: torch.nn.Module, redirect_forward: bool = True):
        """
        Create PyTorch module wrapper.

        Parameters
        ----------
        module: torch.nn.Module
            PyTorch module

        redirect_forward: bool
            Whether the pytorch forward function should be redirected to be able to handle out-of-order inputs
        """

        super().__init__(name)

        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("Pytorch module expected, got " + str(type(module)))

        self.redirect_forward = redirect_forward
        self.original_forward = module.forward
        if self.redirect_forward:
            module.original_forward = module.forward
            module.forward = self.forward

        self.module = module

    def forward(self, *args, **kwargs) -> Tuple[torch.tensor]:
        """
        Run PyTorch module forward, with pre-loaded inputs in input queues

        Parameters
        ----------
        *args
            Inputs into the module

        **kwargs
            Keyword inputs into the moduls

        Returns
        -------
        Tuple[torch.tensor]
            Output tensors, one for each of the module outputs
        """
        if self.redirect_forward:
            if len(self.input_names):
                assert len(self.input_names) == len(args)
                input_dict = OrderedDict(zip(self.input_names, args))
                args = []
                kwargs.update(input_dict)
            outputs = self.module.original_forward(*args, **kwargs)
        else:
            outputs = self.module(*args, **kwargs)

        return outputs

    def cpu_eval_forward(self, *args, **kwargs):
        self.module.cpu()
        self.module.eval()

        outputs = self.forward(*args, **kwargs)
        outputs = flatten_structured_output([outputs])
        return outputs

    def add_parameter(self, name: str, parameter: Parameter):
        """
        Adds a new parameter.

        Parameters
        ----------
        name: str
            Parameter name

        parameter: Parameter
            Parameter to add

        prepend_name: Bool
            Whether to prepend module name to parameter name
        """

        if isinstance(parameter, forge.parameter.Parameter):
            parameter = torch.nn.Parameter(parameter.value(), requires_grad=False)
        if name in self.module._parameters:
            raise RuntimeError(f"Module {self.name} already has parameter '{name}'")
        self.module._parameters[name] = parameter

    def set_parameters(self, **kwargs):
        """
        Set parameters (weights) in this module, by name.

        Parameters
        ----------
        kwargs
            Name-value pairs of parameter/weight names and tensor values
        """
        d = self.module.state_dict()

        for name, value in kwargs.items():
            if name not in d:
                raise RuntimeError("Pytorch module doesn't have parameter called '" + name + "'")

            d[name] = value

        self.module.load_state_dict(d)

    def get_parameters(self) -> List[Parameter]:
        """
        Return the list of parameters defined in this module

        Returns
        -------
        List[Parameter]
            List of all parameters in this module
        """
        params = []
        recorded_names = []
        all_params = [
            self.module.named_parameters(),
            self.module.named_buffers(),
            self.module.state_dict().items(),
            self.module._parameters.items(),
        ]
        for name, param in itertools.chain(*all_params):
            if name in recorded_names:
                continue
            if param == None:
                continue
            forge_param = Parameter(param.cpu(), requires_grad=param.requires_grad, name=name)
            params.append(forge_param)
            recorded_names.append(name)

        return params


class PaddleModule(Module):
    """
    A wrapper around a Paddle module.
    """

    def __init__(self, name: str, module: paddle.nn.Layer):
        super().__init__(name)
        self.module = module

    def forward(self, *args, **kwargs):
        paddle_args = to_pd_tensors(args)
        outputs = self.module(*paddle_args, **kwargs)
        return to_pt_tensors(outputs)

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def set_parameters(self, **kwargs):
        raise NotImplementedError

    def cpu_eval_forward(self, *args, **kwargs):
        self.module.eval()

        outputs = self.forward(*args, **kwargs)
        outputs = flatten_structured_output([outputs])
        return outputs

    def get_parameters(self) -> List[Parameter]:
        params = []
        for param in self.module.parameters():
            forge_param = Parameter(torch.tensor(param.numpy()), requires_grad=param.stop_gradient, name=param.name)
            params.append(forge_param)
        return params


class TFModule(Module):
    """
    A wrapper around a TF module. Currently, TF modules can only run on a CPU device.
    """

    def __init__(self, name: str, module: tf.keras.Model):
        """
        Create TF module wrapper.

        Parameters
        ----------
        module: tf.keras.Model
            TF module
        """

        super().__init__(name)

        if not isinstance(module, (tf.keras.Model, tf.keras.layers.Layer)):
            raise RuntimeError("tf.keras module expected, got " + str(type(module)))

        self.module = module

    def forward(self, *args, **kwargs) -> Tuple[tf.Tensor]:
        """
        Run TF module forward, converting pytorch tensors as necessary

        Parameters
        ----------
        *args
            Inputs into the module

        **kwargs
            Keyword inputs into the moduls

        Returns
        -------
        Tuple[tf.Tensor]
            Output tensors, one for each of the module outputs
        """
        args = to_tf_variables(args)
        kwargs = {
            k: tf.Variable(
                tf.convert_to_tensor(v.detach().numpy(), dtype=map_pt_dtype_to_tf(v.dtype)), trainable=v.requires_grad
            )
            for k, v in kwargs.items()
        }
        outputs = self.call(*args, **kwargs)
        outputs = to_pt_tensors(outputs)
        return outputs

    def cpu_eval_forward(self, *args, **kwargs) -> Tuple[tf.Tensor]:

        args = to_tf_tensors(args, force_float32=True)
        outputs = self.call(*args, **kwargs)
        outputs = flatten_structured_output([outputs])
        outputs = to_pt_tensors(outputs)
        return outputs

    def call(self, *args, **kwargs) -> Tuple[tf.Tensor]:
        """
        Run TF module forward, with pre-loaded inputs in input queues

        Parameters
        ----------
        *args
            Inputs into the module

        **kwargs
            Keyword inputs into the moduls

        Returns
        -------
        Tuple[tf.Tensor]
            Output tensors, one for each of the module outputs
        """
        outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, *args) -> Tuple[tf.Tensor]:
        """
        Run TF module backward, with pre-loaded inputs in input queues

        Parameters
        ----------
        *args: List[Tuple[tf.Tensor, tf.Tensor]]
            List of tuples of output tensors and incoming loss tensors
        """
        raise NotImplementedError

    def set_parameters(self, **kwargs):

        raise NotImplementedError

    def get_parameters(self) -> List[Parameter]:
        params = []
        vars_as_pt = to_pt_tensors(self.module.trainable_variables)
        names = [var.path for var in self.module.trainable_variables]
        for param, name in zip(vars_as_pt, names):
            forge_param = Parameter(param, requires_grad=True, name=name)
            params.append(forge_param)

        return params


class OnnxModule(Module):
    """
    A wrapper around a Onnx module.
    """

    def __init__(self, name: str, module: onnx.onnx_ml_pb2.ModelProto, onnx_path: Optional[str] = None):
        """
        Create Onnx module wrapper.

        Parameters
        ----------
        module: onnx.onnx_ml_pb2.ModelProto
            onnx module
        onnx_path: Optional[str]
            Path to the ONNX model file. Used to directly load the model when its size exceeds 2GB (with external data).
        data_format_override: Optional[Union[forge.DataFormat, torch.dtype]]
            Data format or dtype to override the default data format of the module.
            This is set by the compiler pipeline (generate_forge_module) when default_df_override
            requests a lower precision that ONNXRuntime CPUExecutionProvider cannot natively run.
            Accepts either a forge.DataFormat (e.g. DataFormat.Float16_b) or a torch.dtype
            (e.g. torch.bfloat16); get_parameters() resolves forge.DataFormat automatically.
        """
        super().__init__(name)

        if not isinstance(module, onnx.onnx_ml_pb2.ModelProto):
            raise RuntimeError("onnx.onnx_ml_pb2.ModelProto module expected, got " + str(type(module)))
        self.module: onnx.onnx_ml_pb2.ModelProto = module
        self.onnx_path: Optional[str] = onnx_path
        self.data_format_override: Optional[Union[forge._C.DataFormat, torch.dtype]] = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 2
        so.intra_op_num_threads = 2

        if self.onnx_path is None:
            try:
                model = self.module.SerializeToString()
            except Exception as e:
                raise RuntimeError(
                    "The model exceeds the 2GB limit and cannot be serialized directly. \n"
                    "To proceed, please provide an 'onnx_path' to load the model\n"
                    f"Original error: {e}"
                )
        else:
            model = self.onnx_path

        ort_sess = ort.InferenceSession(
            model,
            sess_options=so,
            use_deterministic_compute=True,
            providers=["CPUExecutionProvider"],
        )
        # Load input names
        input_names = []
        for inp in ort_sess.get_inputs():
            input_names.append(inp.name)

        input_dict = {}
        for name, tensor in zip(input_names, args):
            input_dict[name] = tensor.detach().numpy()

        assert len(input_names) == len(args), "Number of input names must match number of inputs"

        # Load output names
        output_names = []
        for out in ort_sess.get_outputs():
            output_names.append(out.name)

        # Handle batched verification
        slice_single_batch = False
        for inp in ort_sess.get_inputs():
            if all([isinstance(x, int) for x in inp.shape]) and list(inp.shape) != list(  # Don't modify dynamic shapes
                input_dict[inp.name].shape
            ):
                assert inp.shape[0] != input_dict[inp.name].shape[0], "Only batch dim is allowed to be different."
                repeat_times = [inp.shape[0]] + [1] * (len(inp.shape) - 1)
                input_dict[inp.name] = np.tile(input_dict[inp.name], repeat_times)
                slice_single_batch = True

        framework_outputs = ort_sess.run(output_names, input_dict)

        if slice_single_batch:
            framework_outputs = [x[0:1, ...] for x in framework_outputs]

        outputs = flatten_structured_output(framework_outputs)

        outputs = to_pt_tensors(outputs)

        # ONNXRuntime CPUExecutionProvider does not support bfloat16, so the session always
        # runs in float32.  When a lower-precision override has been requested (e.g. bfloat16),
        # cast the float32 outputs to the target dtype so that the golden reference matches the
        # dtype produced by the compiled Forge model.  Integer tensors are left unchanged.
        if self.data_format_override is not None:
            if isinstance(self.data_format_override, forge.DataFormat):
                data_format_override_dtype = forge_dataformat_to_pytorch_dtype(self.data_format_override)
            else:
                data_format_override_dtype = self.data_format_override
            outputs = [
                o.to(data_format_override_dtype) if isinstance(o, torch.Tensor) and torch.is_floating_point(o) else o
                for o in outputs
            ]

        return outputs

    def cpu_eval_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args):
        raise NotImplementedError

    def set_parameters(self, **kwargs):
        raise NotImplementedError

    def set_data_format_override(self, data_format_override: Union[forge._C.DataFormat, torch.dtype]):
        self.data_format_override = data_format_override

    def get_parameters(self) -> List[Parameter]:
        params = []
        for param in self.module.graph.initializer:
            # numpy_helper.to_array() always returns a float32 array even when the ONNX
            # model stores bfloat16 weights, because NumPy has no bfloat16 dtype.  When a
            # lower-precision override is active we therefore cast the resulting tensor to
            # the requested dtype so the Forge graph receives correctly-typed parameters.
            param_data = numpy_helper.to_array(param)
            tensor = torch.tensor(param_data)
            if self.data_format_override is not None and torch.is_floating_point(tensor):
                if isinstance(self.data_format_override, forge._C.DataFormat):
                    data_format_override_dtype = forge_dataformat_to_pytorch_dtype(self.data_format_override)
                else:
                    data_format_override_dtype = self.data_format_override
                tensor = tensor.to(data_format_override_dtype)
            forge_param = Parameter(tensor, requires_grad=False, name=param.name)
            params.append(forge_param)
        return params


class TFLiteModule(Module):
    """
    A wrapper around a TFLite module.
    """

    def __init__(self, name: str, tflite_path: str):
        """
        Create TFLite module wrapper.

        Parameters
        ----------
        tflite_path: str
            path of the tflite object
        """
        super().__init__(name)
        self.tflite_path = tflite_path
        self.module = tf.lite.Interpreter(model_path=tflite_path)

    def forward(self, *args, **kwargs):
        assert self.tflite_path != None
        input_details = self.module.get_input_details()
        output_details = self.module.get_output_details()
        self.module.allocate_tensors()
        args = to_tf_variables(args)
        self.module.set_tensor(input_details[0]["index"], *args)
        self.module.invoke()
        framework_outputs = []
        for i in range(len(output_details)):
            framework_outputs.append(self.module.get_tensor(output_details[i]["index"]))

        outputs = flatten_structured_output(framework_outputs)

        return to_pt_tensors(outputs)

    def cpu_eval_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args):

        raise NotImplementedError

    def set_parameters(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self) -> List[Parameter]:
        return []  # TODO


class TFGraphDefModule(Module):
    """
    A wrapper around a TFGraphDef module.
    """

    def __init__(self, name: str, module, path, output_names):
        super().__init__(name)
        self.module = module
        self.path = path
        self.output_names = output_names

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args):

        raise NotImplementedError

    def set_parameters(self, **kwargs):
        raise NotImplementedError

    def get_parameters(self) -> List[Parameter]:
        return []  # TODO

    def cpu_eval_forward(self, *args, **kwargs):
        input_names = []
        for node in self.module.node:
            if "input" in node.name and node.op == "Placeholder":
                input_names.append(node.name + ":0")

        assert len(input_names) == len(args), "Number of inputs doesn't match number of input nodes"

        detached_args = to_pt_tensors(args)
        detached_args = [x.detach() for x in detached_args]
        inp_dict = dict(zip(input_names, detached_args))
        import tensorflow.compat.v1 as tf

        tf.reset_default_graph()

        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                with tf.io.gfile.GFile(self.path, "rb") as f:
                    tf_graph = tf.GraphDef()
                    tf_graph.ParseFromString(f.read())
                    sess.graph.as_default()

                    tf.graph_util.import_graph_def(tf_graph, name="")

                    _outputs = []
                    for name in self.output_names:
                        _outputs.append(graph.get_tensor_by_name(name))

                    tf.global_variables_initializer()
                    out = sess.run(_outputs, feed_dict=inp_dict)

        outputs = flatten_structured_output(out)
        outputs = to_pt_tensors(outputs)
        return outputs


class JaxModule(Module):
    """
    A wrapper around a Jax module.
    """

    def __init__(self, name: str, module):
        super().__init__(name)
        self.module = module

    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        args = [
            jnp.asarray(
                x.detach().numpy(),
            )
            for x in args
        ]
        outputs = self.module(*args, **kwargs)
        outputs = flatten_structured_output([outputs])
        return to_pt_tensors(outputs)

    def backward(self, *args):
        raise NotImplementedError

    def set_parameters(self, **kwargs):
        raise NotImplementedError

    def cpu_eval_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_parameters(self) -> List[Parameter]:
        parameters = []

        def _set_parameters(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, (dict, flax.core.frozen_dict.FrozenDict)):
                    _set_parameters(value, f"{prefix}{key}." if prefix else key + ".")
                else:
                    param_tensor = to_pt_tensor(value)
                    parameters.append(Parameter(param_tensor, requires_grad=False, name=f"{prefix}{key}"))

        if hasattr(self.module, "params"):  # Handle transformers models
            _set_parameters(self.module.params)
        elif hasattr(self.module, "variables"):  # Handle regular Flax modules
            if "params" in self.module.variables:
                _set_parameters(self.module.variables["params"])
            if "batch_stats" in self.module.variables:
                _set_parameters(self.module.variables["batch_stats"])

        return parameters


class ForgeModule(Module):
    """
    A base class for all Forge modules. User should extend this class and implement `forward` function with workload implementation.
    """

    def __init__(self, name: str):
        super().__init__(name)

        # Parameters in this module. This is auto-managed by __setattr__
        self._parameters: Dict[str, Parameter] = {}
        # Constants that do not require gradients
        self._constants: Dict[str, forge.Tensor] = {}

        # Sub-modules
        self._submodulelists: List[List[Dict[str, "ForgeModule"]]] = []
        self._submodules: Dict[str, "ForgeModule"] = {}

        self._user_inserted_tapout_queues: List[Tuple[str, int]] = []

        self.subgraph_idx = 0

    def get_submodules(self) -> Dict[str, "ForgeModule"]:
        submodules = self._submodules
        for submodulelist in self._submodulelists:
            if not all([isinstance(sm, ForgeModule) for sm in submodulelist]):
                continue
            for submodule in submodulelist:
                submodules[submodule.name] = submodule

        return submodules

    def __getattribute__(self, name: str):
        if name == "forward":
            orig_forward = super(ForgeModule, self).__getattribute__("forward")
            if callable(orig_forward):

                def wrap_forward(*args, **kwargs):
                    if len(self.input_names):
                        assert len(self.input_names) == len(args)
                        input_dict = OrderedDict(zip(self.input_names, args))
                        args = []
                        kwargs.update(input_dict)
                    return orig_forward(*args, **kwargs)

                return wrap_forward
        return super(ForgeModule, self).__getattribute__(name)

    def pre_forward(self, *args, **kwargs):
        """
        Called before forward. Override this function to add custom logic.
        """

    def add_parameter(self, name: str, parameter: Parameter, prepend_name: bool = False):
        """
        Adds a new parameter.

        Parameters
        ----------
        name: str
            Parameter name

        parameter: Parameter
            Parameter to add

        prepend_name: Bool
            Whether to prepend module name to parameter name
        """

        if name in self._parameters:
            raise RuntimeError(f"Module {self.name} already has parameter '{name}'")

        self._parameters[name] = parameter
        if prepend_name:
            parameter._set_auto_name(self.name + "." + name)
        else:
            parameter._set_auto_name(name)

    def add_constant(
        self,
        name: str,
        prepend_name: bool = False,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Adds a new constant.

        Parameters
        ----------
        name: str
            Constant name

        prepend_name: Bool
            Whether to prepend module name to constant name

        shape: Optional[Tuple[int, ...]]
            Shape of the constant tensor

        dtype: torch.dtype
            Datatype of the constant tensor
        """

        if name in self._constants:
            raise RuntimeError(f"Module {self.name} already have constant '{name}'")

        _name = name

        if shape:
            self._constants[_name] = Tensor.create_from_torch(torch.empty(shape, dtype=dtype), constant=True)
        else:
            self._constants[_name] = None

    def get_constant(self, name) -> Tensor:
        """
        Gets a constant by name

        Parameters
        ----------
        name: str
            constant name

        Returns
        -------
        forge.Tensor
            constant in module
        """

        if name not in self._constants:
            raise RuntimeError(f"Module {self.name} doesn't have constant '{name}'.")

        if self._constants[name] == None:
            raise RuntimeError(f"Constant '{name}' in Module {self.name} has not been initialized.")

        return self._constants[name]

    def set_constant(self, name: str, data: SomeTensor):
        """
        Set value for a module constant.

        Parameters
        ----------
        name: str
            constant name

        data: SomeTensor
            Tensor value to be set
        """

        if name not in self._constants:
            raise RuntimeError(f"Module {self.name} doesn't have constant '{name}'")

        logger.trace("Setting constant ({}) value to ".format(name))
        lazy_trace_data(data)

        if isinstance(data, torch.Tensor):
            data = forge.Tensor.create_from_torch(
                data, constant=True, dev_data_format=pytorch_dtype_to_forge_dataformat(data.dtype)
            )

        import numpy as np

        if isinstance(data, np.ndarray):
            data = forge.Tensor.create_from_torch(
                torch.Tensor(data), constant=True, dev_data_format=pytorch_dtype_to_forge_dataformat(data.dtype)
            )

        assert isinstance(data, forge.Tensor)
        self._constants[name] = data

    def get_parameter(self, name) -> Parameter:
        """
        Gets a parameter by name

        Parameters
        ----------
        name: str
            Parameter name

        Returns
        -------
        Parameter
            Module parameter
        """

        if name not in self._parameters:
            raise RuntimeError(f"Module {self.name} doesn't have parameter '{name}'")

        return self._parameters[name]

    def get_parameters(self, submodules: bool = True) -> List[Parameter]:
        """
        Return the list of parameters defined in this module and (optionally) all submodules.

        Parameters
        ----------
        submodules: bool, optional
            If set, parameters of submodules will be returned, as well. True by default.

        Returns
        -------
        List[Parameter]
            List of all parameters in this (and submodules, optionally) module
        """
        if not hasattr(self, "_parameters"):
            raise RuntimeError("Make sure to call super().__init__(name) in module init")

        ret = list(self._parameters.values())
        if submodules:
            for m in self.get_submodules().values():
                for new_param in m.get_parameters(submodules=True):
                    if new_param not in ret:
                        ret.append(new_param)

        return ret

    def set_parameter(self, name: str, data: SomeTensor):
        """
        Set value for a module parameter.

        Parameters
        ----------
        name: str
            Parameter name

        data: SomeTensor
            Tensor value to be set
        """

        if name not in self._parameters:
            raise RuntimeError(f"Module {self.name} doesn't have parameter '{name}'")

        self._parameters[name].set_value(data)

    def load_parameter_dict(self, data: Dict[str, SomeTensor]):
        """
        Load all parameter values specified in the dictionary.

        Parameters
        ----------
        data: Dict[str, SomeTensor]
            Dictionary of name->tensor pairs to be loaded into parameters
        """
        for d in data.items():
            self.set_parameter(*d)

    def initialize_parameters(self):
        pass

    def __setattr__(self, name: str, value):
        """
        Record parameter and sub-modules that are created in this module
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value
            if len(self.name):
                value._set_auto_name(self.name + "." + name)
            else:
                value._set_auto_name(name)
        elif isinstance(value, ForgeModule):
            self._submodules[name] = value
            value.name = self.name + "." + name
        elif isinstance(value, dict):
            # Check if this is a dict of parameters
            for dict_name, dict_value in value.items():
                if isinstance(dict_value, Parameter):
                    self._parameters[dict_name] = dict_value
                    dict_value._set_auto_name(self.name + "." + dict_name)
        elif isinstance(value, list):
            for i, list_item in enumerate(value):
                if isinstance(list_item, Parameter):
                    if not list_item.get_name():
                        list_item._set_auto_name(f"{name}_{i}")
                    self._parameters[list_item.get_name()] = list_item
            if name != "_submodulelists" and name != "input_names":
                self._submodulelists.append(value)

        object.__setattr__(self, name, value)  # default set
        if isinstance(value, ForgeModule):
            value.initialize_parameters()

    def __delattr__(self, name: str):
        """
        Delete parameters and submodules as their attributes are deleted
        """
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._submodules:
            del self._submodules[name]

        object.__delattr__(self, name)

    def insert_tapout_queue_for_op(self, op_name: str, output_index: int):
        """
        Insert an intermediate queue for op (used for checking/debugging)

        Parameters
        ----------
        op_name: str
            Op name

        output_index: int
            Index of the output tensor on the op you want to associate with the queue

        Returns
        -------
        IntQueueHandle
            Unique handle for the tapout queue, used to retrieve values later
        """
        op_name = self.name + "." + op_name
        self._user_inserted_tapout_queues.append((op_name, output_index))
        return IntQueueHandle(self, op_name, output_index)

    def __call__(self, *args) -> Tuple:
        return self.forward(*args)

    def _set_device(self, device: "Device"):
        """
        Sets the device that this module will run on. This is called by the device when module is placed on it, and should be not called by the user.


        Parameters
        ----------
        device: Device
            Parent device
        """
        Module._set_device(self, device)

        for submodule in self.get_submodules().values():
            submodule._set_device(device)

    def process_framework_parameters(self):
        """
        Loads parameters and buffers from the model pt file and sets them to the ForgeModule parameters and constants data members.
        """


class IntQueueHandle:
    """
    Handle for an intermediate queue, a debug device for reading out intermediate operation results from the device
    """

    def __init__(self, module: ForgeModule, op_name: str, output_index: int):
        self.module = module
        self.op_name = op_name
        self.output_index = output_index


def wrap_module(module, name: str) -> Module:
    """
    Wrap a module in a Forge module

    Parameters
    ----------
    module: Any
        Module to wrap

    name: str
        Name of the module

    Returns
    -------
    Module
        Wrapped module
    """
    if isinstance(module, torch.nn.Module):
        return PyTorchModule(name, module)
    elif isinstance(module, tf.keras.Model):
        return TFModule(name, module)
    elif isinstance(module, ForgeModule):
        return module
    elif isinstance(module, paddle.nn.Layer):
        return PaddleModule(name, module)
    elif isinstance(module, onnx.onnx_ml_pb2.ModelProto):
        return OnnxModule(name, module)
    elif isinstance(module, forge.module.OnnxModule):
        return module
    elif isinstance(module, (flax.linen.Module, FlaxPreTrainedModel)):
        return JaxModule(name, module)
    else:
        raise RuntimeError("Unsupported module type: " + str(type(module)))


FrameworkModule: TypeAlias = (
    torch.nn.Module
    | tf.keras.Model
    | paddle.nn.Layer
    | onnx.onnx_ml_pb2.ModelProto
    | OnnxModule
    | flax.linen.Module
    | FlaxPreTrainedModel
)
AnyModule: TypeAlias = FrameworkModule | ForgeModule
