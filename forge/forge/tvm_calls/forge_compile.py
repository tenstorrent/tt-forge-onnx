# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from forge.tensor import to_tf_tensors, to_pt_tensor, to_pt_tensors, to_pd_tensors
from forge.tvm_utils import flatten_inputs, flatten_structured_output
from forge.forge_property_utils import ExecutionStage, record_execution
import torch
import paddle
import flax
import jax.numpy as jnp
import numpy as np
import tvm
from tvm.ir.transform import Pass
import tvm.relay as relay
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
    _construct_concrete_function,
)
from loguru import logger

import copy
import json

from collections.abc import MutableMapping

from json import JSONEncoder
import os
import subprocess
from os.path import exists as file_exists
import onnxruntime as ort
import onnx
import onnx.numpy_helper
from tvm.relay.expr import Tuple
from forge.tvm_calls.relay.op.forge import flatten_IO, compile_for_forge, partition_for_forge
from forge.tvm_calls.relay.op.utils import verify_tvm_compile
from jax.experimental import jax2tf
from jax.tools.jax_to_ir import tf_wrap_with_input_names
from transformers import FlaxPreTrainedModel
from transformers.utils.generic import ModelOutput
from forge.tvm_calls.forge_utils import (
    extract_framework_model_outputs,
    extract_flatten_inputs,
    construct_tvm_ir,
    extract_function_callnodes,
    paddle_trace,
    trace_to_origin,
    has_op,
)
import hashlib

dev_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "tt"}
cpu_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "cpu"}


def retrieve_graph(json_graph, t):
    function_name = t[0]
    if function_name in json_graph["functions"]:
        return

    json_graph["functions"][function_name] = t[1]
    json_graph["param_names"][function_name] = t[2]


@tvm.register_func
def retrieve_forge_json_graph(*args):
    t = tuple(args)
    global dev_json_graph
    retrieve_graph(dev_json_graph, t)


@tvm.register_func
def retrieve_forge_cpudevice_json_graph(*args):
    t = tuple(args)
    global cpu_json_graph
    retrieve_graph(cpu_json_graph, t)


def load_tvm_graph(
    inputs,
    module,
    compiler_cfg,
    graph_name,
    framework,
    path=None,
    verify_cfg=None,
    input_names=[],
):
    """
    Loads TVM graph ported to the Forge from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process from specific framework to the Forge graph representation, or
    skip this compilation process and load serialize TVM graph which is already ported to Forge
    by initial invocation.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations

    path: str
        Path to TFLite file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary, OrderedDict, Tuple, Boolean
        TVM ported graph, Weights, Input tensors, Constant evaluation
    """
    if compiler_cfg.tvm_graph_store_path != "" and compiler_cfg.tvm_graph_load_path != "":
        logger.warning(f"TVM serialization logic will be skipped as both store and load paths are provided")

    json_graphs, flattened_inputs = compile_tvm_graph(
        inputs,
        module,
        compiler_cfg,
        graph_name=graph_name,
        input_names=input_names,
        path=path,
        verify_cfg=verify_cfg,
        framework=framework,
    )

    flattened_pytorch_inputs, weights = format_tvm_graph_weights(
        flattened_inputs, module, compiler_cfg, framework=framework
    )

    serialize_and_store_tvm_graph(json_graphs, compiler_cfg, framework=framework)

    return json_graphs, flattened_pytorch_inputs, weights


def compile_tvm_graph(
    inputs,
    module,
    compiler_cfg,
    graph_name,
    input_names=[],
    path=None,
    verify_cfg=None,
    framework=None,
):
    """
    Compiles TVM graph ported to the Forge from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process or only load serilized TVM graph and thus increase test performance.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Wrapper Module (PyTorchModule, TFModule, ONNXModule, etc)
        Module that contains workload which can be assigned to a single device

    compiler_cfg: CompilerConfig
        Compiler configurations

    path: str
        Path to TFLite/ONNX file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary
        TVM ported graph
    """
    # clear out graphs before starting in case python session is ongoing
    global dev_json_graph
    global cpu_json_graph
    dev_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "tt"}
    cpu_json_graph = {"functions": {}, "graph": "", "param_names": {}, "device": "cpu"}

    if framework == "pytorch":
        json_graphs, inputs = compile_pytorch_for_forge(
            module,
            *inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
            input_names=input_names,
        )
    elif framework == "paddle":
        json_graphs, inputs = compile_paddle_for_forge(
            module,
            *inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
            input_names=input_names,
        )

    elif framework == "tensorflow":
        # convert pytorch tensors to tf tensors
        tf_inputs = to_tf_tensors(inputs)
        json_graphs, inputs = compile_tf_for_forge(
            module,
            *tf_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
        )
    elif framework == "tf_graphdef":
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(None if t is None else tf.convert_to_tensor(t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs
        json_graphs = compile_tf_graphdef_for_forge(
            module,
            *tf_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
        )
    elif framework == "onnx":
        assert all([isinstance(x, torch.Tensor) for x in inputs])
        onnx_inputs = [x.detach().numpy() for x in inputs]
        json_graphs, _ = compile_onnx_for_forge(
            module,
            path,
            *onnx_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
        )
    elif framework == "jax":
        tf_inputs = to_tf_tensors(inputs, force_float32=True)
        json_graphs, inputs = compile_jax_for_forge(
            module,
            *tf_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
        )
    elif framework == "tflite":
        tf_inputs = to_tf_tensors(inputs, force_float32=True)
        json_graphs, inputs = compile_tflite_for_forge(
            module,
            path,
            *tf_inputs,
            graph_name=graph_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
        )
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return json_graphs, inputs


def save_nid_to_input_idx(traced_model_inputs, json_graph):
    existing_graph = json.loads(json_graph["graph"])

    nid_to_input_idx = {}

    # reorder arg nodes to be inputs first, then parameters
    for i, arg_idx in enumerate(existing_graph["arg_nodes"]):
        name = existing_graph["nodes"][arg_idx]["name"]
        if name not in traced_model_inputs:
            continue

        nid_to_input_idx[arg_idx] = traced_model_inputs.index(name)

    json_graph["nid_to_input_idx"] = nid_to_input_idx


def extract_graphs(partitioned_mod, forge_params, input_names, weight_names, param_name_lookup={}, graph_hash=""):
    mod = partitioned_mod["main"]
    main_graph = str(mod.astext())
    cpu_json_graph["hash"] = graph_hash
    dev_json_graph["hash"] = graph_hash

    assert (
        len(cpu_json_graph["functions"].keys()) <= 2
    ), "At most two cpu functions should exist (one pre and one post). If there are more they should have been merged into one during tvm partitioning."
    cpu_functions = list(cpu_json_graph["functions"].keys())
    assert (
        len(dev_json_graph["functions"].keys()) <= 1
    ), "At most one device function should exist. If there are more they should have been merged into one during tvm partitioning."
    device_function = list(dev_json_graph["functions"].keys())[0]

    cpu_pre_function = None
    cpu_post_function = None

    func_callnodes = extract_function_callnodes(partitioned_mod["main"], partitioned_mod.get_global_vars())
    for node in func_callnodes:
        if node.op.name_hint == device_function:
            for arg in node.args:
                op = None
                if isinstance(arg, tvm.relay.expr.TupleGetItem):
                    op = arg.tuple_value.op
                elif isinstance(arg, tvm.relay.expr.Call):
                    op = arg.op
                if op is not None:
                    if op.name_hint != cpu_pre_function:
                        if not cpu_pre_function:
                            cpu_pre_function = op.name_hint
                        else:
                            assert (
                                op.name_hint == cpu_pre_function
                            ), "There is more than one cpu pre function. This should not be possible. They should have been merged into one during tvm partitioning."
    if cpu_pre_function is not None:
        cpu_functions.remove(cpu_pre_function)
    assert (
        len(cpu_functions) <= 1
    ), "There is more than one cpu post function. This should not be possible. They should have been merged into one during tvm partitioning."
    cpu_post_function = cpu_functions[0] if len(cpu_functions) else None

    if cpu_pre_function is not None:
        cpu_pre_json_graph = copy.deepcopy(cpu_json_graph)
        cpu_pre_json_graph["graph"] = cpu_json_graph["functions"][cpu_pre_function]

        # Only keep the pre function in the pre json
        functions_to_remove = []
        for function in cpu_pre_json_graph["functions"]:
            if function != cpu_pre_function:
                functions_to_remove.append(function)

        for func in functions_to_remove:
            del cpu_pre_json_graph["functions"][func]

        cpu_pre_json_graph["params"] = {}
        for function_name in forge_params.keys():
            if function_name == cpu_pre_function:
                cpu_pre_json_graph["params"].update(
                    {
                        name: v.numpy()
                        for (k, v), name in zip(
                            forge_params[function_name].items(), cpu_pre_json_graph["param_names"][function_name]
                        )
                    }
                )
    else:
        cpu_pre_json_graph = {"graph": ""}

    dev_json_graph["graph"] = dev_json_graph["functions"][device_function]

    dev_json_graph["params"] = {}
    for function_name in forge_params.keys():
        if function_name in dev_json_graph["param_names"]:
            dev_json_graph["params"].update(
                {
                    name: v.numpy()
                    for (k, v), name in zip(
                        forge_params[function_name].items(), dev_json_graph["param_names"][function_name]
                    )
                }
            )

    if cpu_post_function is not None:
        cpu_post_json_graph = copy.deepcopy(cpu_json_graph)
        cpu_post_json_graph["graph"] = cpu_json_graph["functions"][cpu_post_function]

        # Only keep the post function in the post json
        functions_to_remove = []
        for function in cpu_post_json_graph["functions"]:
            if function != cpu_post_function:
                functions_to_remove.append(function)

        for func in functions_to_remove:
            del cpu_post_json_graph["functions"][func]

        cpu_post_json_graph["params"] = {}
        for function_name in forge_params.keys():
            if function_name == cpu_post_function:
                cpu_post_json_graph["params"].update(
                    {
                        name: v.numpy()
                        for (k, v), name in zip(
                            forge_params[function_name].items(), cpu_post_json_graph["param_names"][function_name]
                        )
                    }
                )
    else:
        cpu_post_json_graph = {"graph": ""}

    json_graphs = []
    if cpu_pre_function is not None:
        save_nid_to_input_idx(input_names, cpu_pre_json_graph)  # Input order might not be preserved by TVM
        cpu_pre_json_graph["num_forge_inputs"] = len(input_names)
        json_graphs.append(
            copy.deepcopy(
                clean_names(
                    json_graph=cpu_pre_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup
                )
            )
        )
    else:
        save_nid_to_input_idx(input_names, dev_json_graph)  # Input order might not be preserved by TVM
        dev_json_graph["num_forge_inputs"] = len(input_names)

    json_graphs.append(
        copy.deepcopy(
            clean_names(json_graph=dev_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup)
        )
    )

    if cpu_post_json_graph["graph"] != "":
        json_graphs.append(
            copy.deepcopy(
                clean_names(
                    json_graph=cpu_post_json_graph, forge_params=forge_params, param_name_lookup=param_name_lookup
                )
            )
        )

    return json_graphs


class ConvertEmulatedDtypes:
    """
    This class converts data formats which must be emulated by CPU into float32 within its context.
    It is used when a model's parameters and inputs would be slow to compute on CPU and precision is not important.
    """

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.emulated_dfs = [torch.bfloat16]
        self.fallback = torch.float32

    def flatten_object(self, obj):
        if isinstance(obj, (list, tuple)):
            return [item for sublist in obj for item in self.flatten_object(sublist)]
        elif isinstance(obj, dict):
            return [item for key, value in obj.items() for item in self.flatten_object(value)]
        else:
            return [obj]

    def __enter__(self):
        # Convert emulated model parameters to fallback
        self.param_dfs = {}
        for name, param in self.model.named_parameters():
            if param.dtype in self.emulated_dfs:
                self.param_dfs[name] = param.dtype
                param.data = param.data.to(self.fallback)

        # Convert buffers
        self.buffer_dfs = {}
        for name, buf in self.model.named_buffers():
            if buf.dtype in self.emulated_dfs:
                self.buffer_dfs[name] = buf.dtype
                buf.data = buf.data.to(self.fallback)

        # Convert emulated inputs to fallback
        self.input_dfs = []
        for inp in self.flatten_object(self.inputs):
            self.input_dfs.append(inp.dtype)
            if inp.dtype in self.emulated_dfs:
                inp.data = inp.data.to(self.fallback)

        # Conver tensor attributes that are not buffers nor parameters
        self.attr_dfs = {}  # To track tensor attributes for restoration later
        if isinstance(self.model, torch.nn.Module):
            for mod_name, mod in self.model.named_modules():  # Recursively walk submodules
                param_keys = set(dict(mod.named_parameters(recurse=False)).keys())  # current module's params
                buffer_keys = set(dict(mod.named_buffers(recurse=False)).keys())  # current module's buffers

                for attr in dir(mod):  # inspect each attribute
                    if attr.startswith("_") or attr in param_keys or attr in buffer_keys:
                        continue  # skip private/internal or known param/buffer

                    try:
                        val = getattr(mod, attr)
                    except Exception:
                        continue  # skip broken properties

                    if isinstance(val, torch.Tensor) and val.dtype in self.emulated_dfs:
                        key = f"{mod_name}.{attr}" if mod_name else attr
                        self.attr_dfs[key] = (mod, attr, val.dtype)  # store module, attr name, and dtype
                        setattr(mod, attr, val.to(self.fallback))  # overwrite with fallback dtype
                    elif isinstance(val, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in val):
                        original_dtypes = [v.dtype for v in val]
                        if any(dt in self.emulated_dfs for dt in original_dtypes):
                            key = f"{mod_name}.{attr}" if mod_name else attr
                            self.attr_dfs[key] = (mod, attr, original_dtypes)
                            converted_val = [v.to(self.fallback) if v.dtype in self.emulated_dfs else v for v in val]
                            if isinstance(val, tuple):
                                converted_val = tuple(converted_val)
                            setattr(mod, attr, converted_val)

    def __exit__(self, *args):
        # Convert model parameters back to original dtype
        for name, param in self.model.named_parameters():
            if name in self.param_dfs:

                param.data = param.data.to(self.param_dfs[name])

        for name, buf in self.model.named_buffers():
            if name in self.buffer_dfs:
                buf.data = buf.data.to(self.buffer_dfs[name])

        # Convert inputs back to original dtype
        for inp, df in zip(self.flatten_object(self.inputs), self.input_dfs):
            inp.data = inp.data.to(df)

        # Restore all extra tensor attributes
        for key, (mod, attr, orig_dtype) in self.attr_dfs.items():
            try:
                val = getattr(mod, attr)
                if isinstance(val, torch.Tensor):
                    restored_val = val.to(orig_dtype)
                else:
                    restored_val = [v.to(dt) for v, dt in zip(val, orig_dtype)]
                    restored_val = tuple(restored_val) if isinstance(val, tuple) else restored_val

                setattr(mod, attr, restored_val)

            except Exception as e:
                logger.warning(f"Failed to restore attribute '{attr}' on module '{mod}': {e}")


def compile_pytorch_for_forge(torchmod, *inputs, graph_name, compiler_cfg, verify_cfg=None, input_names=[]):
    training_mode = torchmod.training

    with ConvertEmulatedDtypes(torchmod, inputs):
        # Extract framework model outputs
        # ConvertEmulatedDtypes converts tochmod parameters to float32 if they are bfloat16
        # It also converts inputs to float32 if they are bfloat16
        framework_outputs = extract_framework_model_outputs(
            framework="pytorch",
            model=torchmod,
            inputs=inputs,
            verify_tvm_compile=verify_cfg.verify_tvm_compile,
        )

        # (Temporary): Remove when forge supports dropout
        if training_mode and compiler_cfg.enable_tvm_dropout == False:
            torchmod.eval()

        if isinstance(torchmod, torch.jit.ScriptModule):
            torchmod.eval()
            torchmod = torch.jit.freeze(torchmod)

        # Trace framework model
        traced_model = torch.jit.trace(torchmod, inputs, check_trace=False, strict=False)
        # Extract flatten inputs
        flattened_inputs, flattened_input_names, flattened_name_map, input_structure = extract_flatten_inputs(
            framework="pytorch",
            model=traced_model,
            inputs=inputs,
            input_names=input_names,
        )
        # Generate TVM module
        convert_params = compiler_cfg.convert_framework_params_to_tvm
        mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_structure, do_convert_params=convert_params)

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_string = traced_model.graph.str().encode("utf-8")
        graph_hash.update(graph_string)
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="pytorch")
        if cached_graphs is not None:
            return cached_graphs, flattened_inputs

    record_execution(ExecutionStage.FAILED_TVM_RELAY_IO_FLATTENING)
    logger.trace("From PyTorch")
    logger.trace(mod.functions)

    mod = flatten_IO(mod, flattened_name_map)
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    # Construct TVM IR
    mod, _ = construct_tvm_ir(
        framework="pytorch",
        model=torchmod,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    flattened_inputs_as_float = (act.float() if torch.is_floating_point(act) else act for act in flattened_inputs)
    np_inputs = {name: inp.detach().numpy() for name, inp in zip(flattened_input_names, flattened_inputs_as_float)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        input_names=flattened_input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    if training_mode:
        torchmod.train()

    # Extract Graphs (TT, CPU, ...)
    json_graphs = extract_graphs(
        partitioned_mod,
        forge_params,
        flattened_input_names,
        torchmod.state_dict().keys(),
        graph_hash=graph_hash.hexdigest(),
    )

    return json_graphs, flattened_inputs


def compile_paddle_for_forge(paddlemod, *inputs, graph_name, compiler_cfg, verify_cfg=None, input_names=[]):

    paddle_inputs = to_pd_tensors(inputs)

    with ConvertEmulatedDtypes(paddlemod, inputs):
        framework_outputs = extract_framework_model_outputs(
            framework="paddle",
            model=paddlemod,
            inputs=paddle_inputs,
            verify_tvm_compile=verify_cfg.verify_tvm_compile,
        )

    flattened_inputs, input_names, flattened_name_map, input_spec = extract_flatten_inputs(
        framework="paddle",
        model=paddlemod,
        inputs=inputs,
        input_names=input_names,
    )

    if isinstance(paddlemod, paddle.jit.TranslatedLayer):
        traced_model, param_name_lookup = paddlemod, None
    else:
        traced_model, param_name_lookup = paddle_trace(paddlemod, input_spec)

    # Input names must match with the ones in the forward method
    named_inputs = {name: inp.shape for name, inp in zip(input_names, paddle_inputs)}

    # Generate TVM module
    logger.info("Running TVM frontend conversion: PaddlePaddle to Relay IR")
    mod, params = tvm.relay.frontend.from_paddle(traced_model, named_inputs)
    logger.info("TVM frontend conversion completed")

    record_execution(ExecutionStage.FAILED_TVM_RELAY_IO_FLATTENING)

    logger.trace("From Paddle")
    logger.trace(mod.functions)

    mod = flatten_IO(mod, flattened_name_map)

    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    # Construct TVM IR
    mod, _ = construct_tvm_ir(
        framework="paddle",
        model=paddlemod,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    flattened_inputs_as_float = (act.float() if torch.is_floating_point(act) else act for act in flattened_inputs)
    np_inputs = {name: inp.detach().numpy() for name, inp in zip(input_names, flattened_inputs_as_float)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        input_names=input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Extract Graphs (TT, CPU, ...)
    json_graphs = extract_graphs(
        partitioned_mod,
        forge_params,
        input_names,
        paddlemod.state_dict().keys(),
        param_name_lookup=param_name_lookup,
        graph_hash="",
    )

    return json_graphs, flattened_inputs


def compile_tvm_for_forge(
    mod,
    params,
    inputs,
    golden_outputs,
    graph_name,
    input_names=[],
    return_params=False,
    compiler_cfg=None,
    verify_cfg=None,
):
    target = "llvm"
    verify_args = {"inputs": inputs, "framework_outputs": golden_outputs, "verify_cfg": verify_cfg}
    mod, params = compile_for_forge(
        mod,
        target=target,
        params=params,
        graph_name=graph_name,
        **verify_args,
    )

    if verify_cfg is not None and verify_cfg.verify_tvm_compile:
        compiler_cfg.convert_framework_params_to_tvm = True
        # If we have conv2d_transpose ops that are channel-last, tvm cannot execute the module, skip in this case
        skip_verify = has_op(mod["main"], "nn.conv2d_transpose", {"data_layout": "NHWC"})
        if skip_verify:
            logger.warning(
                "Module contains a channel-last nn.conv2d_transpose op, this is not supported in TVM (but may be supported in Forge). Skipping verification..."
            )
        else:
            verify_tvm_compile(mod, params, inputs, target, golden_outputs, "compile_for_forge", verify_cfg=verify_cfg)

    # Reconstruct Ops + export forge graph
    logger.info("Running TVM partition")
    mod, forge_params = partition_for_forge(
        mod, graph_name=graph_name, compiler_cfg=compiler_cfg, input_names=input_names
    )
    logger.info("TVM partition completed")
    tvm.relay.build_module.build(mod, target=target, params=params)
    record_execution(ExecutionStage.FAILED_FORGE_MODULE_GENERATION)

    if return_params:
        return mod, forge_params
    else:
        return mod


def clean_names(json_graph, forge_params, param_name_lookup={}):
    precursor = (
        "tvmgen_default_forge_main_" if json_graph["device"] != "cpu" else "tvmgen_default_forge_cpudevice_main_"
    )

    def trim_count(name):
        for i in range(-1, -len(name), -1):
            if name[i] == "_":
                return name[:i], name[i + 1 :]

    if len(json_graph["params"]) > 0:

        old_params = json_graph["params"]
        json_graph["params"] = {}
        for k, v in old_params.items():
            if precursor in k:
                num_digits = k.replace(precursor, "").find("_")
                key = k.replace(precursor, "")[num_digits + 1 :] + k.replace(precursor, "")[:num_digits]
            else:
                name, num_digits = trim_count(k)
                old_name = param_name_lookup[name] if name in param_name_lookup else name
                param_name_lookup[k] = old_name + f"_{num_digits}"
                key = param_name_lookup[k]  # This is done to sync the node names with the param names
            json_graph["params"][key] = v

    graph = json.loads(json_graph["graph"])

    for node in graph["nodes"]:
        if precursor in node["name"]:
            num_digits = node["name"].replace(precursor, "").find("_")
            node["name"] = (
                node["name"].replace(precursor, "")[num_digits + 1 :] + node["name"].replace(precursor, "")[:num_digits]
            )
        elif param_name_lookup is not None and node["name"] in param_name_lookup:
            node["name"] = param_name_lookup[node["name"]]

    json_graph["graph"] = json.dumps(graph)

    return json_graph


def compile_onnx_for_forge(onnx_mod, onnx_path, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    import onnxruntime as ort

    # Set default num threads to 2, hangs on some hosts otherwise https://github.com/microsoft/onnxruntime/issues/10166
    so = ort.SessionOptions()
    so.inter_op_num_threads = 2
    so.intra_op_num_threads = 2

    if onnx_path is not None:
        onnx_model = onnx_path
    else:
        onnx_model = onnx_mod.SerializeToString()

    onnx_session = ort.InferenceSession(onnx_model, sess_options=so, providers=["CPUExecutionProvider"])

    input_names = []
    for inp in onnx_session.get_inputs():
        input_names.append(inp.name)

    input_dict = {}
    input_shape_dict = {}
    for name, tensor in zip(input_names, inputs):
        input_dict[name] = tensor
        input_shape_dict[name] = tensor.shape

    assert len(input_names) == len(inputs), "Number of input names must match number of inputs"

    framework_outputs = extract_framework_model_outputs(
        framework="onnx",
        model=onnx_mod,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
        input_dict=input_dict,
        onnx_session=onnx_session,
    )

    # These are large objects - release them since we don't need them anymore.
    del onnx_session
    del onnx_model

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_string = str(onnx_mod).encode("utf-8")
        graph_hash.update(graph_string)
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="onnx")
        if cached_graphs is not None:
            return cached_graphs, inputs

    logger.info("Running TVM frontend conversion: ONNX to Relay IR")
    mod, params = relay.frontend.from_onnx(onnx_mod, input_shape_dict, freeze_params=False)
    logger.info("TVM frontend conversion completed")
    mod = relay.transform.DynamicToStatic()(mod)
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        propped_params = {k: (v, True) for k, v in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        input_dict,
        framework_outputs,
        input_names=input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    weight_names = [weight.name for weight in onnx_mod.graph.initializer]
    json_graphs = extract_graphs(
        partitioned_mod, forge_params, input_names, weight_names, graph_hash=graph_hash.hexdigest()
    )

    return json_graphs, inputs


def compile_tflite_for_forge(module, path, *inputs, graph_name, compiler_cfg, verify_cfg=None):

    assert path != None, "TFLite compile needs path to .tflite file on disk."
    tflite_model_buf = open(path, "rb").read()

    input_details = module.get_input_details()

    framework_outputs = extract_framework_model_outputs(
        framework="tflite",
        model=module,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
    )

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input_shape_dict = {}
    input_dict = {}
    input_names = []

    for i, details in enumerate(input_details):
        input_names.append(details["name"])
        input_shape_dict[details["name"]] = list(details["shape"])
        input_dict[details["name"]] = inputs[i]

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_string = str(module).encode("utf-8")
        graph_hash.update(graph_string)
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="tflite")
        if cached_graphs is not None:
            return cached_graphs, inputs

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict=input_shape_dict,
    )
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    assert len(input_names) == len(inputs), "Number of input names must match number of inputs"

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        propped_params = {k: (v, True) for k, v in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        input_dict,
        framework_outputs,
        input_names=input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    json_graphs = extract_graphs(partitioned_mod, forge_params, input_names, [], graph_hash=graph_hash.hexdigest())

    return json_graphs, inputs


from tf2onnx.tf_loader import convert_variables_to_constants_large_model


def get_frozen_graph_for_large_jax_model(
    jaxmodel,
    compiler_cfg,
    *inputs,
):
    # This conversion path will perform the following steps:
    #     - Feed jax parameters as input to jax model
    #     - convert jax model to TF function without parameters
    #     - convert jax parameters to TF Variables
    #     - trace TF function with Variables as input
    #     - use helper function to handle variable freezing (This will avoid using Protobuf)
    #     - attach output shape for each node in frozen graph.

    params = {}
    batch_stats = {}
    if hasattr(jaxmodel, "variables"):
        # Handle regular Flax nn.Module.
        params = jaxmodel.variables.get("params", {})
        batch_stats = jaxmodel.variables.get("batch_stats", {})
    elif hasattr(jaxmodel, "params"):
        # Handle FlaxPreTrainedModel of transformers.
        params = jaxmodel.params.get("params", {})
        batch_stats = jaxmodel.params.get("batch_stats", {})

    if len(params) == 0:
        tf_fn = jax2tf.convert(jaxmodel, enable_xla=compiler_cfg.enable_xla_jax_convert)
        tf_func = tf.function(tf_fn, autograph=False, jit_compile=True)
        tf_func = tf_func.get_concrete_function(*inputs)
    else:
        if isinstance(jaxmodel, FlaxPreTrainedModel):
            # Wrap it to use transformers model's __call__ method.
            def wrapped_flax_model(params, batch_stats, input):
                variables = {"params": params}
                if batch_stats is not None:
                    variables["batch_stats"] = batch_stats
                return jaxmodel(*input, variables, train=False)

            predict_fn = wrapped_flax_model
        else:
            predict_fn = lambda params, batch_stats, input: jaxmodel.apply(
                {"params": params, "batch_stats": batch_stats}, *input
            )

        # Convert model from Jax to TensorFlow
        tf_fn = jax2tf.convert(predict_fn, enable_xla=compiler_cfg.enable_xla_jax_convert)

        # Convert parameters TF variables.
        tf_params = tf.nest.map_structure(lambda param: tf.Variable(param, trainable=True), params)
        tf_batch_stats = tf.nest.map_structure(lambda stat: tf.Variable(stat, trainable=False), batch_stats)
        tf_func = tf.function(
            lambda inputs: tf_fn(tf_params, tf_batch_stats, inputs), autograph=False, jit_compile=True
        )
        # Get graph definition
        tf_func = tf_func.get_concrete_function(inputs)

    graph_def = convert_variables_to_constants_large_model(tf_func)

    # Add Shapes to frozen graph
    for node in graph_def.node:
        op = tf_func.graph._nodes_by_name[node.name]
        if op.outputs:
            node.attr["_output_shapes"].list.shape.extend([output.get_shape().as_proto() for output in op.outputs])

    return graph_def, tf_func


def compile_jax_for_forge(jaxmodel, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    # Extract framework model outputs
    framework_outputs = extract_framework_model_outputs(
        framework="jax",
        model=jaxmodel,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
    )

    if compiler_cfg.enable_tvm_jax_freeze_large_model:
        graph_def, tf_func = get_frozen_graph_for_large_jax_model(
            jaxmodel,
            compiler_cfg,
            *inputs,
        )
    else:
        # Convert model from Jax to TensorFlow
        tf_model = jax2tf.convert(jaxmodel, enable_xla=compiler_cfg.enable_xla_jax_convert)
        tf_func = tf.function(tf_model, autograph=False, jit_compile=True)
        # Get graph definition
        tf_func = tf_func.get_concrete_function(*inputs)
        graph_def = tf_func.graph.as_graph_def(add_shapes=True)

    # Extract flatten inputs
    flattened_inputs, flattened_input_names, _, _ = extract_flatten_inputs(
        framework="jax",
        model=tf_func,
        inputs=inputs,
    )

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_hash.update(str(graph_def).encode("utf-8"))
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="jax")
        if cached_graphs is not None:
            return cached_graphs, flattened_inputs

    outputs = [output.name for output in tf_func.outputs]
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NCHW", outputs=outputs)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    # Write Graph to the TensorBoard
    # writer = tf.summary.create_file_writer("generated_modules/tensorboard/jax")
    # with writer.as_default():
    #     tf.summary.graph(tf_fun.graph)

    # Construct TVM IR
    mod, param_name_lookup = construct_tvm_ir(
        framework="jax",
        model=jaxmodel,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    np_inputs = {i: None if x is None else x.numpy() for i, x in zip(flattened_input_names, flattened_inputs)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Extract Graphs (TT, CPU, ...)
    def flatten_params(params, parent_key="", sep="."):
        items = []
        for key, val in params.items():
            new_key = parent_key + sep + key if parent_key else key

            if isinstance(val, MutableMapping):
                items.extend(flatten_params(val, new_key, sep=sep).items())
            else:
                items.append((new_key, val))

        return dict(items)

    model_params = {}
    if hasattr(jaxmodel, "params"):
        model_params = jaxmodel.params
    elif hasattr(jaxmodel, "variables") and "params" in jaxmodel.variables:
        model_params = jaxmodel.variables["params"]

    weight_names = list(flatten_params(model_params).keys())
    json_graphs = extract_graphs(
        partitioned_mod,
        forge_params,
        flattened_input_names,
        weight_names,
        param_name_lookup,
        graph_hash=graph_hash.hexdigest(),
    )

    return json_graphs, flattened_inputs


def compile_tf_for_forge(tfmod, *inputs, graph_name, compiler_cfg, verify_cfg=None):
    # Extract framework model outputs
    framework_outputs = extract_framework_model_outputs(
        framework="tensorflow",
        model=tfmod,
        inputs=inputs,
        verify_tvm_compile=verify_cfg.verify_tvm_compile,
    )

    # Trace module & get graph definition
    @tf.function
    def trace(*inputs):
        kwargs = {}
        import inspect

        arg_names = inspect.getfullargspec(tfmod.call).args
        if "return_dict" in arg_names:
            kwargs["return_dict"] = False

        if "training" in arg_names:
            kwargs["training"] = False
        return tfmod(*inputs, **kwargs)

    full_model = trace.get_concrete_function(*inputs)
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

    # Extract flatten inputs
    flattened_inputs, flattened_input_names, _, _ = extract_flatten_inputs(
        framework="tensorflow",
        model=frozen_func,
        inputs=inputs,
    )

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_hash.update(str(graph_def).encode("utf-8"))
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="tensorflow")
        if cached_graphs is not None:
            return cached_graphs, flattened_inputs

    flattened_outputs = flatten_structured_output([full_model.structured_outputs])
    # Generate TVM module
    outputs = [x.name for x in flattened_outputs]
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, outputs=outputs)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    # Construct TVM IR
    mod, param_name_lookup = construct_tvm_ir(
        framework="tensorflow",
        model=tfmod,
        tvm_mod=mod,
        params=params,
        compiler_cfg=compiler_cfg,
    )

    # Construct NumPy inputs
    np_inputs = {i: None if x is None else x.numpy() for i, x in zip(flattened_input_names, flattened_inputs)}

    # Compile TVM for Forge
    partitioned_mod, forge_params = compile_tvm_for_forge(
        mod,
        params,
        np_inputs,
        framework_outputs,
        input_names=flattened_input_names,
        graph_name=graph_name,
        return_params=True,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Extract Graphs (TT, CPU, ...)
    weight_names = [weight.name for weight in tfmod.weights]
    json_graphs = extract_graphs(
        partitioned_mod,
        forge_params,
        flattened_input_names,
        weight_names,
        param_name_lookup,
        graph_hash=graph_hash.hexdigest(),
    )

    return json_graphs, flattened_inputs


# TODO: Verify graphdef output vs. TVM output
def compile_tf_graphdef_for_forge(
    graph_def,
    *inputs,
    graph_name,
    compiler_cfg,
):
    output_list_ = compiler_cfg.framework_model_output_names
    # framework_outputs = tfmod(*inputs)
    # if not isinstance(framework_outputs, (list, tuple)):
    #     framework_outputs = [framework_outputs]

    # supported_outputs = (tf.Tensor, torch.Tensor)
    # framework_outputs = [x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)]

    # @tf.function
    # def trace(*inputs):
    #     return tfmod(*inputs, training=False)

    # full_model = trace.get_concrete_function(*inputs)

    # frozen_func = convert_variables_to_constants_v2(full_model)
    # graph_def = frozen_func.graph.as_graph_def()
    input_names = []
    for node in graph_def.node:
        if "input" in node.name and node.op == "Placeholder":
            input_names.append(node.name)

    graph_hash = hashlib.sha256()
    if is_tvm_cache_enabled():
        graph_hash.update(str(graph_def).encode("utf-8"))
        cached_graphs = load_serialized_tvm_graph(compiler_cfg, graph_hash.hexdigest(), framework="tf_graphdef")
        if cached_graphs is not None:
            return cached_graphs

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NCHW", outputs=output_list_)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)
    record_execution(ExecutionStage.FAILED_TVM_RELAY_IR_TRANSFORMATION)

    assert (
        compiler_cfg.enable_tvm_constant_prop == True
    ), "Forge Compile only support tf graphdef model with TVM parameter binding."
    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        assert (
            compiler_cfg.convert_framework_params_to_tvm
        ), "Cannot use constant prop without converting framework params to relay"
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {
                k: (v, True)
                for k, v, in params.items()
                if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])
            }
        else:
            propped_params = {k: (v, True) for k, v, in params.items()}
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    target = "llvm"
    mod, params = compile_for_forge(mod, target=target, params=params, graph_name=graph_name)

    # Reconstruct Ops + export forge graph
    partitioned_mod, forge_params = tvm.relay.op.contrib.forge.partition_for_forge(
        mod, graph_name=graph_name, compiler_cfg=compiler_cfg, input_names=input_names
    )

    tvm.relay.build_module.build(partitioned_mod, target=target, params=params)
    record_execution(ExecutionStage.FAILED_FORGE_MODULE_GENERATION)

    json_graphs = extract_graphs(partitioned_mod, forge_params, input_names, [], graph_hash=graph_hash.hexdigest())

    return json_graphs


def format_tvm_graph_weights(inputs, module, compiler_cfg, framework=None):
    """
    Formats model weights based on specific framework.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    OrderedDict, Boolean, Tuple
        Weights, Constant evaluation, Input tensors
    """
    if framework == "pytorch":
        if compiler_cfg.enable_training:
            for param in module.parameters():
                param.requires_grad = True

        torch_weights = module.state_dict()
        named_buffers = dict(module.named_buffers())
        torch_weights.update(named_buffers)
        named_params = dict(module.named_parameters())
        weights = {
            key: (value, named_params[key].requires_grad if key in named_params else False)
            for key, value in torch_weights.items()
        }

    elif framework == "paddle":
        paddle_weights = module.state_dict()
        named_buffers = dict(module.named_buffers())
        paddle_weights.update(named_buffers)
        named_params = dict(module.named_parameters())
        weights = {
            value.name: (to_pt_tensor(value), value.trainable if key in named_params else False)
            for key, value in paddle_weights.items()
        }

    elif framework == "tensorflow":
        trainable_weight_names = [t_weight.path for t_weight in module.trainable_weights]
        weights = {
            weight.path: (
                torch.Tensor(
                    tf.cast(weight, tf.float32).numpy() if tf.as_dtype(weight.dtype).is_floating else weight.numpy()
                ),
                weight.path in trainable_weight_names,
            )
            for weight in module.weights
        }

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = to_pt_tensors(inputs)  # Maybe we can switch all tensors to numpy?
    elif framework == "tf_graphdef":
        weights = {}
    elif framework == "onnx":
        numpy_weights = [onnx.numpy_helper.to_array(weight) for weight in module.graph.initializer]
        names = [weight.name for weight in module.graph.initializer]
        weights = {
            name: (torch.tensor(weight), issubclass(weight.dtype.type, np.floating))
            for name, weight in zip(names, numpy_weights)
        }

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(onnx.numpy_helper.to_array(x)) for x in inputs if x is not None]
    elif framework == "jax":

        def flatten_params(params, parent_key="", sep="."):
            items = []
            for key, val in params.items():
                new_key = parent_key + sep + key if parent_key else key

                if isinstance(val, MutableMapping):
                    items.extend(flatten_params(val, new_key, sep=sep).items())
                else:
                    items.append((new_key, val))

            return dict(items)

        module_params = {}
        if hasattr(module, "params"):
            module_params = module.params
        elif hasattr(module, "variables") and "params" in module.variables:
            module_params = module.variables["params"]

        module_params = flatten_params(module_params)
        weights = {}
        for key, value in module_params.items():
            torch_tensor = torch.Tensor(np.array(value))
            weights[key] = (torch_tensor, True)

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]

    elif framework == "tflite":
        weights = {}
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return inputs, weights


def is_tvm_cache_enabled():
    return bool(os.environ.get("FORGE_ENABLE_TVM_CACHE", 0))


def get_auto_path(graph_hash, compiler_cfg, is_load):
    """
    Returns auto cache path based on graph hash and tvm submodule git commit

    Parameters
    ----------
    graph_hash: String
        graph hash string

    compiler_cfg: CompilerConfig
        Compiler configurations

    is_load: Boolean
        Are we looking for load path (otherwise store)

    Returns
    -------
    String
        path to store/load graph
    """

    auto_cache = int(os.environ.get("FORGE_ENABLE_TVM_CACHE", 0))
    if bool(auto_cache) and compiler_cfg.tvm_graph_store_path == "" and compiler_cfg.tvm_graph_load_path == "":
        assert (
            auto_cache == -1 or auto_cache == 1
        ), f"FORGE_ENABLE_TVM_CACHE value of {auto_cache} not understood. Set to 1 to enable cache, 0 to disable and -1 to recache module"
        if auto_cache == -1 and is_load:
            auto_path = ""
        else:
            tvm_path = os.path.dirname(tvm.__file__)
            if "site-packages" in tvm_path:
                # uv-installed case.
                tvm_info = subprocess.check_output(["uv", "pip", "show", "tvm"]).decode("utf-8")
                for line in tvm_info.split("\n"):
                    if "Version:" in line:
                        tvm_ver = line.split("Version: ")[-1]
                        tvm_hash = tvm_ver.split("+")[-1]
                        break
            else:
                # Source build case.
                tvm_hash = (
                    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=tvm_path)
                    .decode("utf-8")
                    .strip()
                )
            auto_path = "generated_modules/tvm_cache/" + tvm_hash + "_" + graph_hash
    else:
        auto_path = compiler_cfg.tvm_graph_load_path if is_load else compiler_cfg.tvm_graph_store_path

    return auto_path


def load_serialized_tvm_graph(compiler_cfg, graph_hash, framework):
    """
    Loads serialized TVM graph representation ported to Forge in form of python dictionary.

    Parameters
    ----------
    compiler_cfg: CompilerConfig
        Compiler configurations

    graph_hash: String
        graph hash string

    Returns
    -------
    Dictionary
        Deserialized TVM graph
    """

    load_path = get_auto_path(graph_hash, compiler_cfg, True)

    if (
        load_path == ""
        or not file_exists(load_path)
        or (compiler_cfg.enable_tvm_constant_prop and framework == "pytorch")
    ):
        return None

    with open(load_path, "r") as file:
        serialized_graph_str = json.load(file)

    json_graphs = []
    for id, json_graph in serialized_graph_str.items():
        serialized_dict = {}
        serialized_dict["graph"] = json.dumps(json_graph["graph"])
        serialized_dict["hash"] = json.dumps(json_graph["hash"])
        serialized_dict["params"] = json_graph["params"]
        serialized_dict["device"] = json_graph["device"]
        if "nid_to_input_idx" in json_graph.keys():
            serialized_dict["nid_to_input_idx"] = {int(k): v for k, v in json_graph["nid_to_input_idx"].items()}
        json_graphs.append(serialized_dict)

    logger.info(f"Successfully loaded serialized TVM graph from {load_path} path")

    return json_graphs


def serialize_and_store_tvm_graph(json_graphs, compiler_cfg, framework):
    """
    Serializes TVM graph representation ported to Forge in form of JSON and stores it
    on the desired destination.

    Parameters
    ----------
    json_graph: Dictionary
        Previously compiled TVM graph pored to Forge representation

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    """

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    graph_hash = json_graphs[0]["hash"]
    store_path = get_auto_path(graph_hash, compiler_cfg, False)
    if (
        store_path == ""
        or (compiler_cfg.enable_tvm_constant_prop and framework == "pytorch")
        or not len(dev_json_graph["graph"])
    ):
        return

    serilized_dict = {}

    for id, json_graph in enumerate(json_graphs):
        graph_dict = json.loads(json_graph["graph"])
        params_dict = json_graph["params"]
        serilized_dict[str(id)] = {}
        serilized_dict[str(id)]["graph"] = graph_dict
        serilized_dict[str(id)]["params"] = params_dict
        serilized_dict[str(id)]["device"] = json_graph["device"]
        serilized_dict[str(id)]["hash"] = json_graph["hash"]
        if "nid_to_input_idx" in json_graph.keys():
            serilized_dict[str(id)]["nid_to_input_idx"] = json_graph["nid_to_input_idx"]

    serilized_str = json.dumps(serilized_dict, cls=NumpyArrayEncoder, indent=2)

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as file:
        file.write(serilized_str)

    logger.info(f"Successfully stored serilized TVM graph to {store_path} path")
