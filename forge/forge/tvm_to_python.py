# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import re
import json
import ast

from loguru import logger

import torch
import numpy as np
import pytest

# import forge._C.pattern_matcher as pypattern_matcher
from forge.module import OnnxModule, ForgeModule, TFLiteModule
from forge.verify.config import _get_global_verify_config
import forge
from forge.tensor import to_pt_tensors, forge_dataformat_to_pytorch_dtype
from forge.tvm_utils import flatten_inputs

import os
import sys
import importlib

from forge.python_codegen import PyTorchWriter, ForgeWriter, PythonWriter, pytorch_df_from_str
from forge.tvm_unique_op_generation import Operation, NodeType, extract_and_export_unique_ops_config


class ExitTest(Exception):
    """Signal to abort the current test and mark it PASSED."""

    def __init__(self, reason=None):
        super().__init__(reason)
        self.reason = reason


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def populate_torch_all_to_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_broadcast_to_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "shape": {
            "val": [-1] + list(curr_node["attrs"]["shape"][0][0])[1:],
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_index_select_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    strides = [int(strides) for strides in curr_node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in curr_node["attrs"]["begin"][0]]
    end = [int(e) for e in curr_node["attrs"]["end"][0]]

    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, "Strided slice should be on 1 axis"
    assert int(curr_node["attrs"]["num_inputs"]) == 1
    assert len(list(curr_node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axes"][0][0]),
            "inp_pos": 1,
        },
        "index": {
            "val": f"torch.arange({begin[0]}, {end[0]}, {strides[0]})",
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_adv_index_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    curr_node_shape = curr_node["attrs"]["shape"][0][0]
    assert (len(curr_node_shape) == 1) or (
        curr_node_shape[0] == 1 and len(curr_node_shape)
    ), "Only 1D indexing is supported"

    node_to_index = graph["nodes"][int(curr_node["inputs"][0][0])]
    ndim_node_to_index = len(node_to_index["attrs"]["shape"][0][0])

    args["attr"] = {
        "dim": {
            "val": -ndim_node_to_index,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_reshape_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    output_shape = list(curr_node["forge_shape"])
    output_shape[0] = -1

    args["attr"] = {
        "shape": {
            "val": output_shape,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_cumsum_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_log_softmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_softmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_squeeze_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    assert len(curr_node["attrs"]["axis"][0][0]) == 1, "Currently support only single axis for squeeze"

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_scatter_add_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    assert curr_node["attrs"]["reduction"][0][0] == "add", "TODO Add other scatter elements reductions"
    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_scatter_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_cast_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    dtype = curr_node["attrs"]["dtype"][0][0]
    if dtype == "uint1":
        dtype = "bool"

    args["attr"] = {
        "dtype": {
            "val": f"torch.{dtype}",
            "inp_pos": 1,
        }
    }
    args["inplace"] = True

    return args


def populate_torch_transpose_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # Handle TVM axes stile for PyTorch case. More precisely,
    # TVM defines new operand positions as axes, while PyTorch
    # requires 2 indexes which defines which dimensions should
    # be transposed.
    #
    # E.g.
    # [0, 2, 1, 3] => [1, 2]
    changed_axes = []
    axes = [int(axis) for axis in curr_node["attrs"]["axes"][0]]
    for i, axis in enumerate(axes):
        if i != axis:
            changed_axes.append(str(i))
    curr_node["attrs"]["axes"][0] = changed_axes

    assert len(curr_node["attrs"]["axes"][0]) == 2, "Pytorch Transpose only supports 2 dimensions"

    args["attr"] = {
        "dim0": {
            "val": int(curr_node["attrs"]["axes"][0][0]),
            "inp_pos": 1,
        },
        "dim1": {
            "val": int(curr_node["attrs"]["axes"][0][1]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_sum_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    dim = int(curr_node["attrs"]["axis"][0][0])
    keepdims = bool(curr_node["attrs"]["keepdims"][0][0])
    args["attr"] = {"dim": {"val": dim, "inp_pos": 1}, "keepdims": {"val": keepdims, "inp_pos": 2}}

    return args


def populate_torch_argmax_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
        "keepdims": {
            "val": bool(curr_node["attrs"]["keepdims"][0][0]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_argwhere_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # args['attr'] = {
    #     "as_tuple": {
    #         "val": True,
    #         "inp_pos": 2,
    #         "as_named": True,
    #     },
    # }

    return args


def populate_torch_reduce_max_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
        "keepdims": {
            "val": bool(curr_node["attrs"]["keepdims"][0][0]),
            "inp_pos": 2,
        },
    }

    return args


def populate_torch_concat_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    # Determine dim
    dim = None
    act_shape = graph["nodes"][curr_node["inputs"][0][0]]["forge_shape"]
    curr_node_shape = curr_node["forge_shape"]
    for dim, (i, o) in enumerate(zip(act_shape, curr_node_shape)):
        if i != o:
            dim = int(dim - len(act_shape))
            break
    assert dim, "Could not find concatenate axis"

    args["attr"] = {
        "dim": {
            "val": dim,
            "inp_pos": 1,
        }
    }

    return args


def populate_torch_stack_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "dim": {
            "val": int(curr_node["attrs"]["axis"][0][0]),
            "inp_pos": 1,
        },
    }

    return args


def populate_torch_tile_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)
    args["attr"] = {
        "reps": {
            "val": [int(reps) for reps in curr_node["attrs"]["reps"][0]],
            "inp_pos": 1,
        },
    }

    return args


def populate_torch_layernorm_args(graph, nid, compiler_cfg):
    curr_node, args = _populate_torch_init_args(graph, nid)

    epsilon = float(curr_node["attrs"]["epsilon"][0][0])
    epsilon = round(epsilon, 10)
    input_shape = curr_node["attrs"]["shape"][0][0]
    dim = int(curr_node["attrs"]["axis"][0][0])
    normalized_shape = input_shape[dim]
    args["attr"] = {
        "normalized_shape": {
            "val": (normalized_shape,),
            "inp_pos": 1,
        },
        "epsilon": {
            "val": epsilon,
            "inp_pos": 4,
        },
    }

    return args


def populate_torch_dropout_args(graph, nid, training):
    curr_node, args = _populate_torch_init_args(graph, nid)

    args["attr"] = {
        "training": {
            "val": training,
            "inp_pos": 2,
        },
    }
    return args


def _populate_torch_init_args(graph, nid):
    curr_node = graph["nodes"][nid]

    args = {"inp": [], "attr": {}, "inplace": False}
    for i in range(0, int(curr_node["attrs"]["num_inputs"])):
        args["inp"].append(graph["nodes"][curr_node["inputs"][i][0]]["forge_name"])

    return curr_node, args


tvm_to_pytorch_op_map = {
    "abs": "abs",
    "add": "add",
    "all": "all",
    "argmax": "argmax",
    "argwhere": "argwhere",
    "broadcast_to_like": "broadcast_to_like",
    "broadcast_to": "broadcast_to",
    "cast": "cast",
    "concatenate": "concatenate",
    "const": "const",
    "cumsum": "cumsum",
    "embedding": "embedding",
    "equal": "equal",
    "exp": "exp",
    "floor": "floor",
    "greater_equal": "greater_equal",
    "identity": "identity",
    "less_equal": "less_equal",
    "less": "less",
    "logical_and": "logical_and",
    "logical_not": "logical_not",
    "max": "reduce_max",
    "maximum": "maximum",
    "minimum": "minimum",
    "multiply": "multiply",
    "nn.batch_matmul": "matmul",
    "nn.dense": "linear",
    "nn.log_softmax": "log_softmax",
    "nn.matmul": "matmul",
    "nn.softmax": "softmax",
    "not_equal": "not_equal",
    "power": "power",
    "forge_cpudevice.adv_index": "adv_index",
    "forge_cpudevice.concatenate": "concatenate",
    "reciprocal": "reciprocal",
    "reshape": "reshape",
    "scatter_elements": "scatter_add",
    "scatter": "scatter",
    "sigmoid": "sigmoid",
    "sign": "sign",
    "sqrt": "sqrt",
    "squeeze": "squeeze",
    "stack": "stack",
    "strided_slice": "index_select",
    "subtract": "subtract",
    "sum": "sum",
    "take": "embedding",
    "tile": "tile",
    "transpose": "transpose",
    # "take"                        : "take",
    "where": "where",
    "layernorm": "layernorm",
    "forge_cpudevice.dropout": "dropout",
}

pytorch_op_to_function_name = {
    "abs": "torch.abs",
    "add": "torch.add",
    "adv_index": "torch.index_select",
    "all": "torch.all",
    "argmax": "torch.argmax",
    "argwhere": "torch.nonzero",
    "broadcast_to_like": "torch.broadcast_to_like",
    "broadcast_to": "torch.broadcast_to",
    "cast": "to",
    "concatenate": "torch.cat",
    "const": "torch.Tensor",
    "cumsum": "torch.cumsum",
    "embedding": "torch.embedding",
    "equal": "torch.eq",
    "exp": "torch.exp",
    "floor": "torch.floor",
    "greater_equal": "torch.ge",
    "identity": "torch.nn.Identity()",
    "index_select": "torch.index_select",
    "less_equal": "torch.le",
    "less": "torch.less",
    "linear": "torch.nn.functional.linear",
    "log_softmax": "torch.nn.functional.log_softmax",
    "logical_and": "torch.logical_and",
    "logical_not": "torch.logical_not",
    "matmul": "torch.matmul",
    "maximum": "torch.maximum",
    "minimum": "torch.minimum",
    "multiply": "torch.multiply",
    "not_equal": "torch.not_equal",
    "power": "torch.pow",
    "reciprocal": "torch.reciprocal",
    "reduce_max": "torch.max",
    "repeat": "torch.Tensor.repeat",
    "reshape": "torch.reshape",
    "scatter_add": "torch.scatter_add",
    "scatter": "torch.scatter",
    "sigmoid": "torch.sigmoid",
    "sign": "torch.sign",
    "softmax": "torch.nn.functional.softmax",
    "sqrt": "torch.sqrt",
    "squeeze": "torch.squeeze",
    "stack": "torch.stack",
    "subtract": "torch.subtract",
    "sum": "torch.sum",
    "tile": "torch.Tensor.repeat",
    "transpose": "torch.transpose",
    # "take"                          : "torch.take",
    "where": "torch.where",
    "layernorm": "torch.nn.functional.layer_norm",
    "dropout": "torch.nn.functional.dropout",
}

pytorch_ops_needing_arguments = {
    "adv_index": populate_torch_adv_index_args,
    "all": populate_torch_all_to_args,
    "argmax": populate_torch_argmax_args,
    "argwhere": populate_torch_argwhere_args,
    "broadcast_to": populate_torch_broadcast_to_args,
    "cast": populate_torch_cast_args,
    "concatenate": populate_torch_concat_args,
    "cumsum": populate_torch_cumsum_args,
    "index_select": populate_torch_index_select_args,
    "log_softmax": populate_torch_log_softmax_args,
    "reduce_max": populate_torch_reduce_max_args,
    "reshape": populate_torch_reshape_args,
    "scatter_add": populate_torch_scatter_add_args,
    "scatter": populate_torch_scatter_args,
    "softmax": populate_torch_softmax_args,
    "squeeze": populate_torch_squeeze_args,
    "stack": populate_torch_stack_args,
    "sum": populate_torch_sum_args,
    "tile": populate_torch_tile_args,
    "transpose": populate_torch_transpose_args,
    "layernorm": populate_torch_layernorm_args,
    "dropout": populate_torch_dropout_args,
}


def populate_conv2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # Retaining the Padding format for Forge Conv2d Pad Format (Top,Left,Bottom,Right)
    args.append(
        (
            "padding",
            f"{padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    groups = int(node["attrs"]["groups"][0][0])
    args.append(
        (
            "groups",
            f"{groups}",
        )
    )

    channel_last = node["attrs"]["data_layout"][0][0] == "NHWC"
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_cumsum_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    axis = node["attrs"]["axis"][0][0]
    args.append(
        (
            "dim",
            f"{axis}",
        )
    )
    return args


def populate_conv2d_transpose_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    if len(set(padding)) == 1:
        # Uniform padding
        args.append(
            (
                "padding",
                f"{padding[0]}",
            )
        )
    else:
        # Tuple padding
        args.append(
            (
                "padding",
                f"{tuple(padding)}",
            )
        )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    assert all([x == 1 for x in dilation]), "Only supports dilation of 1"
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    in_channel = None
    for input_ in node["inputs"]:
        input_nid = input_[0]
        input_node = graph["nodes"][input_nid]
        if input_node["op"] == "parameter" and input_node["name"].endswith("weight"):
            in_channel = input_node["attrs"]["shape"][0][0][0]
            break

    groups = int(node["attrs"]["groups"][0][0])
    assert groups == 1 or (in_channel is not None and groups == in_channel), "Only supports group of 1 or in_channel"
    args.append(
        (
            "groups",
            f"{groups}",
        )
    )

    kernel_size = [int(kernel) for kernel in node["attrs"]["kernel_size"][0]]
    channel_last = node["attrs"]["data_layout"][0][0] == "NHWC"
    args.append(("channel_last", f"{channel_last}"))

    output_padding = [int(opad) for opad in node["attrs"]["output_padding"][0]]
    args.append(("output_padding", f"{output_padding}"))

    return args


def populate_argmax_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    # Handle the case where axis is not None (None is represented as empty string in TVM)
    if node["attrs"]["axis"][0][0] != "":
        dim = int(node["attrs"]["axis"][0][0])
        args.append(("dim", f"{dim}"))

    keep_dim = bool(int(node["attrs"]["keepdims"][0][0]))
    args.append(("keep_dim", f"{keep_dim}"))

    return args


def populate_avgpool2d_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    args.append(
        (
            "kernel_size",
            f"{kernel_size}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    reordered_padding = [
        padding[1],
        padding[3],
        padding[0],
        padding[2],
    ]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    # dilation = [int(dilate) for dilate in node["attrs"]["dilation"][0]]
    # attrs.append(dilation[0])

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])  # 1 for True
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    count_include_pad = int(node["attrs"]["count_include_pad"][0][0])
    count_include_pad = "True" if count_include_pad == 1 else "False"
    args.append(("count_include_pad", count_include_pad))

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_avgpool1d_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    args.append(
        (
            "kernel_size",
            f"{kernel_size}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    args.append(
        (
            "stride",
            f"{strides}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    args.append(
        (
            "padding",
            f"{padding}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])  # 1 for True
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    count_include_pad = int(node["attrs"]["count_include_pad"][0][0])
    count_include_pad = "True" if count_include_pad == 1 else "False"
    args.append(("count_include_pad", count_include_pad))
    return args


def populate_maxpool1d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert len(kernel_size) == 1
    args.append(
        (
            "kernel_size",
            f"{kernel_size[0]}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert len(strides) == 1
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    assert all([p == padding[0] for p in padding])
    args.append(
        (
            "padding",
            f"{padding[0]}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert len(dilation) == 1
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    return args


def populate_maxpool2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert all([dim == kernel_size[0] for dim in kernel_size])
    args.append(
        (
            "kernel_size",
            f"{kernel_size[0]}",
        )
    )

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    reordered_padding = [
        padding[1],
        padding[3],
        padding[0],
        padding[2],
    ]
    args.append(
        (
            "padding",
            f"{reordered_padding}",
        )
    )

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    args.append(
        (
            "dilation",
            f"{dilation[0]}",
        )
    )

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0])
    ceil_mode = "True" if ceil_mode == 1 else "False"
    args.append(
        (
            "ceil_mode",
            f"{ceil_mode}",
        )
    )

    channel_last = int(node["attrs"]["layout"][0][0] == "NHWC")
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_unsqueeze_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = [("dim", f"{dim}")]
    return args


def populate_squeeze_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = [("dim", f"{dim}")]
    return args


def populate_index_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    strides = [int(strides) for strides in node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in node["attrs"]["begin"][0]]
    end = [int(e) for e in node["attrs"]["end"][0]]

    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, "Strided slice should be on a single axis"
    assert int(node["attrs"]["num_inputs"]) == 1

    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    # Use negative indexing
    if dim >= 0:
        dim -= len(input_shape)

    if end[0] == (2**31 - 1):  # Max int32 - To the end from TVM way of processing data
        end[0] = node["attrs"]["shape"][0][0][dim]

    args = []
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "start",
            f"{begin[0]}",
        )
    )
    args.append(
        (
            "stop",
            f"{end[0]}",
        )
    )
    args.append(
        (
            "stride",
            f"{strides[0]}",
        )
    )
    return args


def populate_broadcast_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    output_shape = node["attrs"]["shape"][0][0]

    dim = 0
    shape = output_shape[dim]
    for i, (inp_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
        if inp_dim == out_dim:
            continue

        dim = i
        shape = out_dim
        input_shape[i] = out_dim
        assert input_shape == output_shape, "Forge broadcast only supports 1 dim"

    dim = dim - len(input_shape)
    args = []
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "shape",
            f"{shape}",
        )
    )
    return args


def populate_reduce_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    dim = int(node["attrs"]["axis"][0][0])
    assert len(node["attrs"]["axis"][0]) == 1, "Forge only supports reduce with a single axis"

    keep_dim = int(node["attrs"]["keepdims"][0][0])

    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    if dim >= 0:
        dim -= len(input_shape)

    args = [("dim", f"{dim}"), ("keep_dim", f"{bool(keep_dim)}")]
    return args


def populate_select_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    begin = [int(x) for x in node["attrs"]["begin"][0]]
    end = [int(x) for x in node["attrs"]["end"][0]]
    strides = [int(x) for x in node["attrs"]["strides"][0]]

    assert end[0] > begin[0], "Select end must be bigger than begin"
    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])

    assert dim is not None
    if dim >= 0:
        dim -= len(input_shape)

    assert int(input_shape[dim]) % int(strides[0]) == 0, "Shape must be divisible by stride"
    select_stride = int(input_shape[dim]) // int(strides[0])
    args.append(
        (
            "dim",
            f"{dim}",
        )
    )
    args.append(
        (
            "index",
            f"({begin[0]}, {end[0]})",
        )
    )
    args.append(
        (
            "stride",
            f"{select_stride}",
        )
    )
    return args


def populate_softmax_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    dim = int(node["attrs"]["axis"][0][0])
    args = [("dim", f"{dim}")]
    return args


def populate_layernorm_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    epsilon = float(node["attrs"]["epsilon"][0][0])
    epsilon = round(epsilon, 10)

    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    dim = int(node["attrs"]["axis"][0][0])
    if dim > 0:
        dim = dim - len(input_shape)
    args = []
    args.append(("dim", f"{dim}"))
    args.append(("epsilon", f"{epsilon}"))

    return args


def populate_cast_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    args = []
    dtype = node["attrs"]["dtype"][0][0]
    args.append(("dtype", pytorch_df_from_str(dtype, node["forge_name"])))
    return args


def populate_transpose_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]
    axes = [int(axis) for axis in node["attrs"]["axes"][0]]
    transpose_shape = list(graph["nodes"][nid]["forge_shape"])

    assert int(node["attrs"]["num_inputs"]) == 1

    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] += len(transpose_shape)

    node["attrs"]["axes"] = axes

    transpose_axes = []
    for idx, axis in enumerate(axes):
        if axis != idx:
            transpose_axes.insert(0, axis)

    # Tmp. Needs to be removed after full Jax Bert support
    if len(transpose_axes) == 0:
        transpose_axes = axes

    assert len(transpose_axes) == 2, "only single axis transpose supported at this time, decompose in tvm"

    transpose_axes = [axis - len(transpose_shape) for axis in transpose_axes]

    args = []
    args.append(("dim0", f"{transpose_axes[0]}"))
    args.append(("dim1", f"{transpose_axes[1]}"))

    return args


def populate_reshape_args(graph, nid, compiler_cfg):
    output_shape = graph["nodes"][nid]["forge_shape"]
    args = []
    args.append(("shape", f"{output_shape}"))
    return args


def populate_concatenate_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    concat_axis = int(node["attrs"]["axis"][0][0])
    if concat_axis >= 0:
        concat_axis -= len(forge_shape)

    args = [
        ("axis", f"{concat_axis}"),
    ]
    return args


def populate_repeat_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    reps = list(map(int, node["attrs"]["reps"][0]))
    args = [("repeats", f"{reps}")]
    return args


def populate_repeat_interleave_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    repeats = int(node["attrs"]["repeats"][0][0])
    dim = int(node["attrs"]["axis"][0][0])

    args = [("repeats", f"{repeats}"), ("dim", f"{dim}")]
    return args


def populate_stack_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    forge_shape = node["forge_shape"]
    stack_axis = int(node["attrs"]["axis"][0][0])
    if stack_axis >= 0:
        stack_axis -= len(forge_shape)

    args = [
        ("axis", f"{stack_axis}"),
    ]
    return args


def populate_clip_transpose_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    min = float(node["attrs"]["a_min"][0][0])
    max = float(node["attrs"]["a_max"][0][0])

    if min == float("inf"):
        min = "float('inf')"

    if max == float("inf"):
        max = "float('inf')"

    if compiler_cfg.default_df_override is not None:
        torch_dtype = forge_dataformat_to_pytorch_dtype(compiler_cfg.default_df_override)
        min_val = torch.finfo(torch_dtype).min
        max_val = torch.finfo(torch_dtype).max
        if isinstance(min, str):
            min = ast.literal_eval(min)
        if isinstance(max, str):
            max = ast.literal_eval(max)
        if min < min_val:
            min = str(min_val)
        if max > max_val:
            max = str(max_val)

    args.append(("min", f"{min}"))
    args.append(("max", f"{max}"))
    return args


def populate_pad_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    pad_width = [int(x) for x in node["attrs"]["pad_width"][0]]
    shape = node["attrs"]["shape"][0][0]
    channel_last = False

    mode = node["attrs"]["pad_mode"][0][0]
    assert mode in ["constant", "edge", "reflect"], "Forge pad only support constant/replicate/reflect padding for now"
    if len(shape) > 2:
        # Forge Pad only supports padding on last 2 dims
        assert len(pad_width) == len(shape) * 2
        assert all([x == 0 for x in pad_width[0:-6]]), "Forge Pad does not support padding on W dim"
        assert all([x == 0 for x in pad_width[-6:-4]]) or all(
            [x == 0 for x in pad_width[-2:]]
        ), "Forge only support Z dim padding for channel-last inputs"
        if any([x != 0 for x in pad_width[-6:-4]]):
            pad_width = pad_width[-6:-2]
            channel_last = True
        else:
            pad_width = pad_width[-4:]

    # TVM nn.pad axis start from the last axis, need to swap
    pad_width_by_axis = [pad_width[x : x + 2] for x in range(0, len(pad_width), 2)]
    pad_width_by_axis.reverse()
    pad_width_final = [item for axis in pad_width_by_axis for item in axis]

    if len(pad_width_final) == 2:
        args.append(
            (
                "pad",
                f"({pad_width_final[0]}, {pad_width_final[1]})",
            )
        )
    elif len(pad_width_final) == 4:
        args.append(
            (
                "pad",
                f"({pad_width_final[0]}, {pad_width_final[1]}, {pad_width_final[2]}, {pad_width_final[3]})",
            )
        )
    else:
        raise ValueError("Pad only support 2D or 4D padding")

    tvm_pad_mode_to_forge_mode = {
        "constant": "constant",
        "edge": "replicate",
        "reflect": "reflect",
    }

    args.append(
        (
            "mode",
            f'"{tvm_pad_mode_to_forge_mode[mode]}"',
        )
    )

    args.append(
        (
            "channel_last",
            f"{channel_last}",
        )
    )

    return args


def populate_resize1d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    sizes = [int(x) for x in node["attrs"]["size"][0]]
    assert len(sizes) == 1, "Resize1D should only have one size dimension"

    method = node["attrs"]["method"][0][0]
    mode = "nearest" if method == "nearest_neighbor" else method

    assert mode in ["nearest", "linear"], "Resize1d op only support nearest and linear mode for now"
    assert int(node["attrs"]["num_inputs"]) == 1

    args.append(
        (
            "size",
            f"{sizes[0]}",
        )
    )
    args.append(
        (
            "mode",
            f'"{mode}"',
        )
    )

    coordinate_transform_mode = node["attrs"]["coordinate_transformation_mode"][0][0]
    align_corners = "True" if coordinate_transform_mode == "align_corners" else "False"
    args.append(
        (
            "align_corners",
            f"{align_corners}",
        )
    )

    channel_last = bool(int(node["attrs"]["layout"][0][0] == "NWC"))
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_resize2d_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    sizes = [int(x) for x in node["attrs"]["size"][0]]
    assert len(sizes) == 2

    method = node["attrs"]["method"][0][0]
    if method == "nearest_neighbor":
        mode = "nearest"
    elif method == "linear":
        mode = "bilinear"
    else:
        mode = method

    assert mode in ["nearest", "bilinear"], "Resize2d op only support nearest and bilinear mode for now"

    assert int(node["attrs"]["num_inputs"]) == 1

    args.append(
        (
            "sizes",
            f"[{sizes[0]}, {sizes[1]}]",
        )
    )
    args.append(
        (
            "mode",
            f'"{mode}"',
        )
    )

    coordinate_transform_mode = node["attrs"]["coordinate_transformation_mode"][0][0]
    align_corners = "True" if coordinate_transform_mode == "align_corners" else "False"
    args.append(
        (
            "align_corners",
            f"{align_corners}",
        )
    )

    channel_last = bool(int(node["attrs"]["layout"][0][0] == "NHWC"))
    args.append(("channel_last", f"{channel_last}"))

    return args


def populate_index_copy_args(graph, nid, compiler_cfg):
    dim = graph["nodes"][nid]["attrs"]["axis"][0][0]
    args = []
    args.append(("dim", f"{dim}"))
    return args


def populate_unsupported_args(graph, nid, compiler_cfg):
    node = graph["nodes"][nid]

    args = []
    op_type = node["name"]
    args.append(("op_type", f'"{op_type}"'))
    output_shape = node["forge_shape"]
    args.append(("output_shape", f"{output_shape}"))
    for k, v in node["attrs"].items():
        if k == "num_outputs" or k == "num_inputs" or k == "shape":
            continue
        while isinstance(v, list) and len(v) == 1:
            v = v[0]
        args.append((k, f'"{v}"'))
    return args


def populate_leaky_relu_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    alpha = node["attrs"]["alpha"][0][0]

    args.append(("alpha", alpha))

    return args


def populate_gelu_args(graph, nid, compiler_cfg):

    args = []
    node = graph["nodes"][nid]
    approx = node["attrs"]["approximate"][0][0]
    args.append(
        (
            "approximate",
            f'"{approx}"',
        )
    )

    return args


def populate_pixel_shuffle_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    upscale_factor = node["attrs"]["upscale_factor"][0][0]
    args.append(("upscale_factor", f"{upscale_factor}"))

    return args


def populate_prelu_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    axis = int(node["attrs"]["axis"][0][0])

    forge_shape = node["forge_shape"]
    if axis >= 0:
        axis -= len(forge_shape)
    args.append(
        (
            "axis",
            f'"{axis}"',
        )
    )


# def populate_dropout_args(graph, nid, training):
#     args = []
#     node = graph["nodes"][nid]

#     p = node["attrs"]["rate"][0][0]
#     args = (("p", p), ("training", str(training)))

#     return args


def populate_quantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]

    args.append(("out_dtype", "torch." + node["attrs"]["out_dtype"][0][0]))
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    return args


def populate_dequantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    args.append(("out_dtype", "torch." + node["attrs"]["dtype"][0][0]))

    return args


def populate_requantize_args(graph, nid, compiler_cfg):
    args = []
    node = graph["nodes"][nid]
    args.append(("axis", f"{int(node['attrs']['axis'][0][0])}"))
    args.append(("out_dtype", "torch." + node["attrs"]["out_dtype"][0][0]))

    return args


# keep sorted
tvm_to_forge_op_map = {
    "abs": "abs",
    "add": "add",
    "floor_mod": "remainder",
    "argmax": "argmax",
    "broadcast_to": "broadcast",
    "cast": "cast",
    "clip": "clip",
    "concatenate": "concatenate",
    "cos": "cos",
    "cumsum": "cumsum",
    "divide": "divide",
    "embedding": "embedding",
    "equal": "equal",
    "erf": "erf",
    "exp": "exp",
    "gelu": "gelu",
    "greater_equal": "greater_equal",
    "greater": "greater",
    "identity": "identity",
    "image.resize1d": "resize1d",
    "image.resize2d": "resize2d",
    "layernorm": "layernorm",
    "less_equal": "less_equal",
    "less": "less",
    "log": "log",
    "logical_and": "logical_and",
    "logical_not": "logical_not",
    "bitwise_and": "bitwise_and",
    "max": "reduce_max",
    "maximum": "maximum",
    "mean": "reduce_avg",
    "minimum": "minimum",
    "multiply": "multiply",
    "nn.avg_pool1d": "avg_pool1d",
    "nn.avg_pool2d": "avg_pool2d",
    "nn.batch_matmul": "matmul",
    "nn.conv2d_transpose": "conv2d_transpose",
    "nn.conv2d": "conv2d",
    "nn.leaky_relu": "leaky_relu",
    "nn.log_softmax": "log_softmax",
    "nn.matmul": "matmul",
    "nn.max_pool1d": "max_pool1d",
    "nn.max_pool2d": "max_pool2d",
    "nn.pad": "pad",
    "nn.relu": "relu",
    "nn.softmax": "softmax",
    "not_equal": "not_equal",
    "pixel_shuffle": "pixel_shuffle",
    "power": "power",
    "nn.prelu": "prelu",
    "forge.adv_index": "adv_index",
    "forge.forge_conv2d_transpose_with_bias": "conv2d_transpose",
    "forge.forge_conv2d_with_bias": "conv2d",
    "forge.concatenate": "concatenate",
    "forge.dropout": "dropout",
    "forge.matmul": "matmul",
    "reciprocal": "reciprocal",
    "reshape": "reshape",
    "scatter_elements": "index_copy",
    "sigmoid": "sigmoid",
    "sigmoid": "sigmoid",
    "sin": "sin",
    "sqrt": "sqrt",
    "stack": "stack",
    "strided_slice": "index",
    "subtract": "subtract",
    "sum": "reduce_sum",
    "take": "take",
    "tanh": "tanh",
    "tile": "repeat",
    "repeat": "repeat_interleave",
    "transpose": "transpose",
    "where": "where",
    "expand_dims": "unsqueeze",
    "squeeze": "squeeze",
    "qnn.dense": "matmul",
    "atan": "atan",
}

forge_op_to_function_name = {
    "abs": "forge.op.Abs",
    "add": "forge.op.Add",
    "remainder": "forge.op.Remainder",
    "adv_index": "forge.op.AdvIndex",
    "argmax": "forge.op.Argmax",
    "avg_pool1d": "forge.op.AvgPool1d",
    "avg_pool2d": "forge.op.AvgPool2d",
    "broadcast": "forge.op.Broadcast",
    "cast": "forge.op.Cast",  # Datatype cast
    "clip": "forge.op.Clip",
    "concatenate": "forge.op.Concatenate",
    "conv2d_transpose": "forge.op.Conv2dTranspose",
    "conv2d": "forge.op.Conv2d",
    "cos": "forge.op.Cosine",
    "cumsum": "forge.op.CumSum",
    "divide": "forge.op.Divide",
    "dropout": "forge.op.Identity",  # (Temporary): change when forge supports dropout
    "embedding": "forge.op.Embedding",
    "equal": "forge.op.Equal",
    "erf": "forge.op.Erf",
    "exp": "forge.op.Exp",
    "gelu": "forge.op.Gelu",
    "greater_equal": "forge.op.GreaterEqual",
    "greater": "forge.op.Greater",
    "identity": "forge.op.Identity",
    "index_copy": "forge.op.IndexCopy",
    "index": "forge.op.Index",
    "layernorm": "forge.op.Layernorm",
    "leaky_relu": "forge.op.LeakyRelu",
    "less_equal": "forge.op.LessEqual",
    "less": "forge.op.Less",
    "log_softmax": "forge.op.LogSoftmax",
    "log": "forge.op.Log",
    "logical_and": "forge.op.LogicalAnd",
    "logical_not": "forge.op.LogicalNot",
    "bitwise_and": "forge.op.BitwiseAnd",
    "matmul": "forge.op.Matmul",
    "max_pool1d": "forge.op.MaxPool1d",
    "max_pool2d": "forge.op.MaxPool2d",
    "maximum": "forge.op.Max",
    "mean": "forge.op.ReduceAvg",
    "minimum": "forge.op.Min",
    "multiply": "forge.op.Multiply",
    "not_equal": "forge.op.NotEqual",
    "pad": "forge.op.Pad",
    "pixel_shuffle": "forge.op.PixelShuffle",
    "power": "forge.op.Power",
    "prelu": "forge.op.Prelu",
    "reciprocal": "forge.op.Reciprocal",
    "reduce_avg": "forge.op.ReduceAvg",
    "reduce_max": "forge.op.ReduceMax",
    "reduce_sum": "forge.op.ReduceSum",
    "relu": "forge.op.Relu",
    "repeat": "forge.op.Repeat",
    "repeat_interleave": "forge.op.RepeatInterleave",
    "reshape": "forge.op.Reshape",
    "resize1d": "forge.op.Resize1d",
    "resize2d": "forge.op.Resize2d",
    "select": "forge.op.Select",
    "sigmoid": "forge.op.Sigmoid",
    "sin": "forge.op.Sine",
    "softmax": "forge.op.Softmax",
    "sqrt": "forge.op.Sqrt",
    "stack": "forge.op.Stack",
    "subtract": "forge.op.Subtract",
    "take": "forge.op.AdvIndex",
    "tanh": "forge.op.Tanh",
    "transpose": "forge.op.Transpose",
    "unsupported": "Unsupported",
    "where": "forge.op.Where",
    "unsqueeze": "forge.op.Unsqueeze",
    "squeeze": "forge.op.Squeeze",
    "atan": "forge.op.Atan",
}
forge_ops_needing_arguments = {
    "argmax": populate_argmax_args,
    "avg_pool1d": populate_avgpool1d_args,
    "avg_pool2d": populate_avgpool2d_args,
    "broadcast": populate_broadcast_args,
    "cast": populate_cast_args,
    "clip": populate_clip_transpose_args,
    "concatenate": populate_concatenate_args,
    "conv2d_transpose": populate_conv2d_transpose_args,
    "conv2d": populate_conv2d_args,
    "cumsum": populate_cumsum_args,
    "gelu": populate_gelu_args,
    "index_copy": populate_index_copy_args,
    "index": populate_index_args,
    "layernorm": populate_layernorm_args,
    "leaky_relu": populate_leaky_relu_args,
    "log_softmax": populate_softmax_args,
    "max_pool1d": populate_maxpool1d_args,
    "max_pool2d": populate_maxpool2d_args,
    "pad": populate_pad_args,
    "pixel_shuffle": populate_pixel_shuffle_args,
    "prelu": populate_prelu_args,
    "reduce_avg": populate_reduce_args,
    "reduce_max": populate_reduce_args,
    "reduce_sum": populate_reduce_args,
    "repeat": populate_repeat_args,
    "repeat_interleave": populate_repeat_interleave_args,
    "reshape": populate_reshape_args,
    "resize1d": populate_resize1d_args,
    "resize2d": populate_resize2d_args,
    "select": populate_select_args,
    "softmax": populate_softmax_args,
    "stack": populate_stack_args,
    "transpose": populate_transpose_args,
    "unsupported": populate_unsupported_args,
    "unsqueeze": populate_unsqueeze_args,
    "squeeze": populate_squeeze_args,
    # "dropout"                      : populate_dropout_args,
}


class ModuleWrapper(ForgeModule):
    def __init__(self, torchmod, name):
        super().__init__(name=name)
        self.torchmod = torchmod

    def forward(self, *acts):
        return self.torchmod(*acts)


def get_framework(module):
    if isinstance(module, forge.module.PyTorchModule):
        framework = "pytorch"
    elif isinstance(module, forge.module.TFModule):  # or isinstance(module, tf.keras.layers.Layer):
        framework = "tensorflow"
    elif isinstance(module, forge.module.TFGraphDefModule):
        framework = "tf_graphdef"
    elif isinstance(module, forge.module.OnnxModule):
        framework = "onnx"
    elif isinstance(module, forge.module.JaxModule):
        framework = "jax"
    elif isinstance(module, forge.module.TFLiteModule):
        framework = "tflite"
    elif isinstance(module, forge.module.PaddleModule):
        framework = "paddle"
    else:
        assert False, f"Unsupported framework: {type(module)}"

    return framework


counter = 0
generated_files = []


def cleanup_temporary_files():
    global generated_files
    for filename in generated_files:
        if os.path.exists(filename):
            os.remove(filename)

    generated_files = []


def get_forge_outputs(forge_mods, devices, forge_inputs):
    from forge.tensor import to_forge_tensors, to_pt_tensors

    for i, (mod, dev) in enumerate(zip(forge_mods, devices)):
        if dev == "CPUDevice":
            forge_inputs = to_pt_tensors(forge_inputs)
        else:
            forge_inputs = to_forge_tensors(to_pt_tensors(forge_inputs))
        forge_inputs = mod.forward(*forge_inputs)

    if not isinstance(forge_inputs, (list, tuple)):
        forge_inputs = [forge_inputs]
    return to_forge_tensors(forge_inputs)


def verify_framework_vs_forge_codegen(frame_outputs, forge_outputs, verify_cfg):
    from forge.verify.compare import compare_tensor_to_golden

    test_pass = True
    for i, (golden, output) in enumerate(zip(frame_outputs, forge_outputs)):
        test_pass &= compare_tensor_to_golden(
            f"Framework vs. Forge codegen output {i}", golden, output.value(), verify_cfg=verify_cfg
        )

        assert test_pass, f"Data mismatch on output {i} between framework and Forge codegen"
    logger.info("Verified python codegen agains framework")


def save_writers_metadata(modules, inputs, sorted_inputs, module_name):
    metadata = {}
    metadata["writers"] = []

    for index, module in enumerate(modules):
        metadata["writers"].append({})
        metadata["writers"][index]["module_directory"] = module.module_directory
        metadata["writers"][index]["class_name"] = module.class_name
        metadata["writers"][index]["dev"] = module.dev
        metadata["writers"][index]["module_name"] = module.module_name
        metadata["writers"][index]["filename"] = module.filename

    metadata["input_indices"] = {}
    for index, unsorted_input in enumerate(inputs):
        for sorted_index, sorted_input in enumerate(sorted_inputs):
            if sorted_input.dtype == unsorted_input.dtype and torch.equal(sorted_input, unsorted_input):
                metadata["input_indices"][index] = sorted_index

    metadata["framework"] = modules[0].framework

    filename = module_name + ".json"
    with open(os.path.join(modules[0].module_directory, filename), "w") as metadata_file:
        json.dump(metadata, metadata_file)
    return


def metadata_path(module_name):
    module_directory = "generated_modules"
    filename = module_name + ".json"
    return os.path.join(module_directory, filename)


def load_writers_metadata(module_name, inputs):
    filepath = metadata_path(module_name)
    assert os.path.exists(
        filepath
    ), f"{filepath} not found, has the test been run with FORGE_RELOAD_GENERATED_MODULES disabled and compiler_cfg.retain_tvm_python_files enabled"
    with open(filepath, "r") as metadata_file:
        metadata = json.load(metadata_file)

    input_indices = metadata["input_indices"]
    flattened_inputs, _, _ = flatten_inputs(inputs)
    ordered_inptus = [None] * len(flattened_inputs)

    for k, v in input_indices.items():
        ordered_inptus[int(k)] = flattened_inputs[v]

    module_writers = []
    for module in metadata["writers"]:
        writer = PythonWriter(module["module_name"], open_file=False)
        writer.module_directory = module["module_directory"]
        writer.class_name = module["class_name"]
        writer.dev = module["dev"]
        writer.module_name = module["module_name"]
        writer.filename = module["filename"]
        module_writers.append(writer)

    return module_writers, ordered_inptus


def generate_forge_module(
    framework_mod,
    inputs,
    compiler_cfg=None,
    graph_name=None,
    verify_cfg=None,
    clean_later=False,
    input_names=[],
):
    global counter

    if compiler_cfg is None:
        compiler_cfg = CompilerConfig()

    if verify_cfg is None:
        verify_cfg = _get_global_verify_config()

    pytorch_inputs = to_pt_tensors(inputs)

    if compiler_cfg.default_df_override is not None and isinstance(framework_mod, OnnxModule):
        framework_mod.set_data_format_override(compiler_cfg.default_df_override)

    if graph_name is None:
        graph_name = framework_mod.name

    reload = bool(int(os.environ.get("FORGE_RELOAD_GENERATED_MODULES", "0")))
    if reload and not compiler_cfg.retain_tvm_python_files:
        compiler_cfg.retain_tvm_python_files = True
        if not os.path.exists(metadata_path(graph_name)):
            reload = False

    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        framework_outputs = framework_mod.cpu_eval_forward(*pytorch_inputs)

    if not reload:
        module_name = graph_name if counter == 0 else f"{graph_name}_{counter}"
        logger.info(f"Generating Forge module: {module_name}")
        module_writers, flattened_inputs = compile_tvm_to_python(
            framework_mod,
            graph_name,
            pytorch_inputs,
            module_name=module_name,
            compiler_cfg=compiler_cfg,
            verify_cfg=verify_cfg,
            input_names=input_names,
        )
        logger.info(f"Forge module generation completed: {module_name}")
    else:
        module_writers, flattened_inputs = load_writers_metadata(graph_name, inputs)

    counter += 1
    sys.path.append(".")

    forge_mods = []
    devices = []
    for writer in module_writers:
        # Load the generated module
        module_name = writer.module_name
        file_path = os.path.join(writer.module_directory, writer.filename)
        module = import_from_path(module_name, file_path)

        TestClass = getattr(module, writer.class_name)

        devices.append(writer.dev)
        if writer.dev == "CPUDevice":
            forge_mod = forge.PyTorchModule(writer.module_name, TestClass())
            forge_mod.module.process_framework_parameters(framework_mod.module)
        else:
            forge_mod = TestClass(writer.module_name)

            forge_mod.process_framework_parameters(framework_mod.module)

            assert not any(
                [param.value() is None for param in forge_mod.get_parameters()]
            ), f"Could not retrieve parameters from framework and tvm"

        forge_mods.append(forge_mod)

        if not compiler_cfg.retain_tvm_python_files:
            global generated_files
            generated_files.append(writer.filename)
            param_filename = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
            if os.path.exists(param_filename):
                generated_files.append(os.path.abspath(param_filename))

            if not clean_later:
                cleanup_temporary_files()

    if devices[0] == "CPUDevice":
        forge_inputs = forge.tensor.to_pt_tensors(flattened_inputs)
    else:
        forge_inputs = forge.tensor.to_forge_tensors(flattened_inputs)

    if verify_cfg is not None and verify_cfg.verify_forge_codegen_vs_framework:
        forge_outputs = get_forge_outputs(forge_mods, devices, forge_inputs)
        verify_framework_vs_forge_codegen(framework_outputs, forge_outputs, verify_cfg=verify_cfg)

    return forge_mods, devices, forge_inputs


def compile_tvm_to_python(
    framework_mod,
    graph_name,
    inputs,
    module_name=None,
    compiler_cfg=None,
    verify_cfg=None,
    input_names=[],
):
    if compiler_cfg is None:
        compiler_cfg = CompilerConfig()

    is_training = False if verify_cfg == None else verify_cfg.test_kind.is_training()

    framework = get_framework(framework_mod)
    if framework in ["pytorch", "paddle"]:
        if is_training:
            framework_mod.module.train()
            verify_cfg.verify_tvm_compile = False
            logger.warning("Cannot verify TVM output vs. framework in training mode.")
        else:
            framework_mod.module.eval()

    # Path is needed for TFLite model verification against TVM compile.
    path = None
    if isinstance(framework_mod, OnnxModule):
        path = framework_mod.onnx_path
    elif isinstance(framework_mod, TFLiteModule):
        path = framework_mod.tflite_path

    # Load here to avoid importing tvm unnecessarily when this file is loaded
    from forge.tvm_calls.forge_compile import load_tvm_graph

    logger.info(f"Running TVM graph compilation (framework: {framework})")
    json_graphs, flattened_pytorch_inputs, weights = load_tvm_graph(
        inputs,
        framework_mod.module,
        compiler_cfg,
        graph_name,
        framework,
        path=path,
        verify_cfg=verify_cfg,
        input_names=input_names,
    )
    logger.info("TVM graph compilation completed")

    df_override_dtype = None
    if compiler_cfg.default_df_override is not None and framework == "onnx":
        df_override_dtype = forge_dataformat_to_pytorch_dtype(compiler_cfg.default_df_override)
        flattened_pytorch_inputs = [
            inp.to(df_override_dtype) if isinstance(inp, torch.Tensor) and torch.is_floating_point(inp) else inp
            for inp in flattened_pytorch_inputs
        ]

    def _determine_node_dtype(node):
        if "framework_dtype" in node["attrs"].keys() and node["attrs"]["framework_dtype"] != "N/A":
            return node["attrs"]["framework_dtype"]
        else:
            logger.trace(
                f"Node '{node['forge_name']}' does not have a framework dtype specified. Using TVM generated dtype."
            )
            return node["attrs"]["dtype"][0][0]

    span_lexer = re.compile("\S+$").search

    def span_to_src_layer(node):
        if "span" not in node["attrs"]:
            return None
        match = span_lexer(node["attrs"]["span"])
        return match.group(0) if match is not None else None

    modules = []
    for graph_index, json_graph in enumerate(json_graphs):
        graph = json.loads(json_graph["graph"])

        is_cpu_pre = graph_index == 0 and json_graph["device"] == "cpu"

        output_nodes = [head[0] for head in graph["heads"]]
        # reshape nops are added in tvm to passthrough nodes, prune them
        def is_nop_reshape(nid):
            node = graph["nodes"][nid]
            if node["name"] != "reshape":
                return False

            input_shape = graph["nodes"][node["inputs"][0][0]]["attrs"]["shape"]
            node_shape = node["attrs"]["shape"]
            return input_shape == node_shape

        input_nodes = graph["arg_nodes"]

        graph_input_names = {}
        params = {}
        constants = {}
        ops = {}
        returns = {}
        returns_requiring_batch_dim_fix = []
        forge_inputs = [None] * len(flattened_pytorch_inputs)
        params_from_tvm = {}
        node_name_to_node_type = {}

        def make_parser_friendly_name(node, node_type):
            if framework == "tensorflow" or framework == "tf_graphdef" or framework == "jax":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_")
            elif framework == "pytorch":
                node["forge_name"] = node["forge_name"].replace(".", "_")
            elif framework == "onnx":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_").replace(":", "_")
            elif framework == "tflite":
                node["forge_name"] = node["forge_name"].replace(".", "_").replace("/", "_").replace(":", "_")

            # Preventing variable names starting with an integer in generated python
            if node["forge_name"][0] in [f"{n}" for n in range(10)]:
                node["forge_name"] = f"{node_type}_{node['name']}"

        # Clean up Forge name
        for node in graph["nodes"]:
            node["forge_name"] = node["name"]

        # Check for unsupported ops
        has_unsupported_ops = False
        json_graph_op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
        for nid, node in enumerate(graph["nodes"]):
            if node["op"] != "kernel":
                continue

            if node["name"] not in json_graph_op_map:
                has_unsupported_ops = True
                unsupported_msg = (
                    f"Encountered unsupported op node type: {node['name']}, on device: {json_graph['device']}"
                )
                logger.warning(unsupported_msg) if compiler_cfg.enable_tvm_unsupported_ops else logger.error(
                    unsupported_msg
                )
        if has_unsupported_ops:
            assert (
                compiler_cfg.enable_tvm_unsupported_ops
            ), "Encountered unsupported op types. Check error logs for more details"

        for nid, node in enumerate(graph["nodes"]):

            node["nid"] = nid
            node["users"] = []
            shape = node["attrs"]["shape"][0][0]
            node["forge_shape"] = tuple(shape)
            if node["op"] == "input":
                if node["name"] not in weights:
                    make_parser_friendly_name(node, "input_")
                    # TVM might not preserve input order; check json graph
                    inp_idx = nid
                    if "nid_to_input_idx" in json_graph.keys() and len(json_graph["nid_to_input_idx"]) != 0:
                        inp_idx = json_graph["nid_to_input_idx"][nid]
                        forge_inputs[inp_idx] = flattened_pytorch_inputs[inp_idx]

                    graph_input_names[inp_idx] = node["forge_name"]
                    node_name_to_node_type[node["forge_name"]] = NodeType.Activation
                    node["op"] = "*"
                    logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: input")
                else:
                    tensor, requires_grad = weights[node["name"]]
                    tensor.requires_grad = requires_grad

                    if (requires_grad or json_graph["device"] == "cpu") and len(tensor.shape) > 0:
                        # CPU PyTorch module requires non-trainable weights in a nn.Parameter with
                        # requires_grad=False, a constant buffer does not work.
                        params[node["nid"]] = (
                            node["forge_name"],
                            node["forge_shape"],
                            requires_grad,
                            _determine_node_dtype(node),
                        )
                        node["op"] = "parameter"
                        node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                        logger.trace(
                            f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: parameter, requires_grad: {requires_grad}"
                        )
                    else:
                        if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                            tensor = tensor.reshape((1,))
                        if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                            tensor = tensor.reshape(tensor.shape[-4:])
                        if requires_grad:
                            node["op"] = "parameter"
                            params[node["nid"]] = (
                                node["forge_name"],
                                tensor.shape,
                                requires_grad,
                                _determine_node_dtype(node),
                            )
                            node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                            logger.trace(
                                f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Parameter"
                            )
                        else:
                            node["op"] = "constant"
                            constants[node["nid"]] = (node["forge_name"], tensor.shape, _determine_node_dtype(node))
                            node_name_to_node_type[node["forge_name"]] = NodeType.Constant
                            logger.trace(
                                f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Constant"
                            )

            elif node["op"] == "const":
                if isinstance(json_graph["params"][node["name"]], np.ndarray):
                    tensor = torch.from_numpy(json_graph["params"][node["name"]])
                else:
                    tensor = torch.tensor(json_graph["params"][node["name"]])

                requires_grad = node["attrs"]["is_param"] != "0"

                if requires_grad and len(tensor.shape) > 0:
                    if tensor.dtype == torch.bool:
                        requires_grad = False
                        node["attrs"]["is_param"] = "0"
                    params_from_tvm[node["forge_name"]] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
                    params[node["nid"]] = (
                        node["forge_name"],
                        node["forge_shape"],
                        requires_grad,
                        _determine_node_dtype(node),
                    )
                    node["op"] = "parameter"
                    node_name_to_node_type[node["forge_name"]] = NodeType.Parameter
                    logger.trace(
                        f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: parameter, requires_grad: {requires_grad}"
                    )
                else:
                    if torch.numel(tensor) == 1 and len(tensor.shape) == 0:
                        tensor = tensor.reshape((1,))
                    if len(tensor.shape) > 4 and all([x == 1 for x in tensor.shape[0:-4]]):
                        tensor = tensor.reshape(tensor.shape[-4:])
                    params_from_tvm[node["forge_name"]] = tensor
                    node["op"] = "constant"
                    constants[node["nid"]] = (node["forge_name"], tensor.shape, _determine_node_dtype(node))
                    node_name_to_node_type[node["forge_name"]] = NodeType.Constant
                    logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']} type: Constant")

            elif node["op"] == "kernel":
                op_map = tvm_to_forge_op_map if json_graph["device"] == "tt" else tvm_to_pytorch_op_map
                if node["name"] in op_map:
                    op_type = op_map[node["name"]]
                else:
                    op_type = "unsupported"
                node["op"] = op_type

                function_map = (
                    forge_op_to_function_name if json_graph["device"] == "tt" else pytorch_op_to_function_name
                )
                function_name = function_map[op_type]
                node["forge_name"] = op_type + f"_{nid}"

                args = ()
                argument_getter = (
                    forge_ops_needing_arguments if json_graph["device"] == "tt" else pytorch_ops_needing_arguments
                )
                if op_type in argument_getter:
                    if op_type == "dropout" and json_graph["device"] != "tt":
                        if is_training:
                            logger.warning(
                                "Dropout op cannot be cpu fallback in training mode due to the absence of rate/p(probability) argument and it may also result in pcc mismatch"
                            )
                        args = argument_getter[op_type](graph=graph, nid=nid, training=is_training)
                    else:
                        args = argument_getter[op_type](graph=graph, nid=nid, compiler_cfg=compiler_cfg)
                    assert args is not None

                if args == () and json_graph["device"] == "cpu" and op_type not in argument_getter:
                    _, args = _populate_torch_init_args(graph, nid)

                logger.trace(f"Node: {nid} shape: {node['forge_shape']} name: {node['forge_name']}  type: op")

                assert "num_inputs" in node["attrs"]

                # TVM nn.pad has 2 inputs [Data, pad_value]
                # We remove the constant and move it to the arguments
                if node["name"] == "nn.pad" and int(node["attrs"]["num_inputs"]) == 2:
                    pad_value_node = graph["nodes"][node["inputs"][1][0]]
                    pad_value_node_name = pad_value_node["name"]
                    pad_value = json_graph["params"][pad_value_node_name]
                    assert pad_value_node["nid"] in constants
                    args.append(("value", f"{float(pad_value.item())}"))
                    del constants[pad_value_node["nid"]]
                    node["attrs"]["num_inputs"] = "1"

                if node["name"] == "qnn.quantize":
                    assert int(node["attrs"]["num_inputs"]) == 3
                    zp_node = graph["nodes"][node["inputs"][2][0]]
                    zp_node_name = zp_node["name"]
                    assert zp_node["nid"] in constants
                    zp_value = json_graph["params"][zp_node_name]
                    del constants[zp_node["nid"]]
                    args.append(("zero_point", f"{float(zp_value.item())}"))
                    node["attrs"]["num_inputs"] = "2"

                if node["name"] == "qnn.dequantize":
                    assert int(node["attrs"]["num_inputs"]) == 3

                    zp_node = graph["nodes"][node["inputs"][2][0]]
                    zp_node_name = zp_node["name"]
                    assert zp_node["nid"] in constants
                    zp_value = json_graph["params"][zp_node_name]
                    del constants[zp_node["nid"]]
                    args.append(("zero_point", f"{zp_value.item()}"))
                    node["attrs"]["num_inputs"] = "2"

                if node["name"] == "qnn.requantize":
                    assert int(node["attrs"]["num_inputs"]) == 5
                    inp_zp_node = graph["nodes"][node["inputs"][2][0]]
                    inp_zp_node_name = inp_zp_node["name"]
                    assert inp_zp_node["nid"] in constants
                    inp_zp_value = json_graph["params"][inp_zp_node_name]
                    args.append(("input_zero_point", f"{inp_zp_value.item()}"))

                    out_zp_node = graph["nodes"][node["inputs"][4][0]]
                    out_zp_node_name = out_zp_node["name"]
                    assert out_zp_node["nid"] in constants
                    out_zp_value = json_graph["params"][out_zp_node_name]
                    args.append(("output_zero_point", f"{out_zp_value.item()}"))

                    node["inputs"] = [node["inputs"][0], node["inputs"][1], node["inputs"][3]]
                    del constants[inp_zp_node["nid"]]
                    del constants[out_zp_node["nid"]]
                    node["attrs"]["num_inputs"] = "3"

                input_names = []
                input_shapes = []
                input_dtypes = []
                input_node_types = []
                for input_port in range(int(node["attrs"]["num_inputs"])):
                    input_nid = node["inputs"][input_port][0]
                    input_node = graph["nodes"][input_nid]
                    if "users" not in input_node:
                        input_node["users"] = []
                    input_node["users"].append(nid)
                    input_names.append(input_node["forge_name"])
                    input_shapes.append(input_node["forge_shape"])
                    input_dtypes.append(_determine_node_dtype(input_node))
                    if input_nid in params.keys():
                        input_node_types.append(NodeType.Parameter)
                    elif input_nid in constants.keys():
                        input_node_types.append(NodeType.Constant)
                    else:
                        input_node_types.append(NodeType.Activation)

                # Handle concatenate case when a single node name in referenced twice in the input list
                if node["name"] == "forge.concatenate" and len(input_names) == 1:
                    inp_shape = graph["nodes"][node["inputs"][input_port][0]]["attrs"]["shape"][0][0]
                    out_shape = node["attrs"]["shape"][0][0]

                    if inp_shape[:2] == out_shape[:2] and inp_shape[2] * 2 == out_shape[2]:
                        input_names = [input_names[0], input_names[0]]
                        input_shapes = [input_shapes[0], input_shapes[0]]
                        input_dtypes = [input_dtypes[0], input_dtypes[0]]
                        input_node_types = [input_node_types[0], input_node_types[0]]

                if json_graph["device"] == "tt" and node["name"] == "embedding":
                    input_names = [input_names[1], input_names[0]]
                    input_shapes = [input_shapes[1], input_shapes[0]]
                    input_dtypes = [input_dtypes[1], input_dtypes[0]]
                    input_node_types = [input_node_types[1], input_node_types[0]]

                node_name_to_node_type[node["forge_name"]] = NodeType.Activation
                ops[node["nid"]] = Operation(
                    function_name=function_name,
                    # node_name=node["forge_name"],
                    output_name=node["forge_name"],
                    input_names=input_names,
                    args=args,
                    src_layer=span_to_src_layer(node),
                    input_shapes=input_shapes,
                    input_dtypes=input_dtypes,
                    input_node_types=input_node_types,
                )

        if any([input is None for input in forge_inputs]):
            forge_inputs = flattened_pytorch_inputs

        for output_nid in output_nodes:
            output_node = graph["nodes"][output_nid]
            returns[output_nid] = output_node["forge_name"]
            if len(output_node["forge_shape"]) == 0:
                returns_requiring_batch_dim_fix.append(output_node["forge_name"])
            elif output_node["forge_shape"][0] != 1:
                returns_requiring_batch_dim_fix.append(output_node["forge_name"])

            new_output_nid = len(graph["nodes"])
            graph["nodes"].append(
                {
                    "forge_name": output_node["forge_name"] + "_output",
                    "op": "output",
                    "nid": new_output_nid,
                    "inputs": [[output_nid, 0, 0]],
                    "attrs": {"num_inputs": "1"},
                }
            )

        def replace_node_name(orig, new):
            for op in ops.values():
                while orig in op.input_names:
                    op.input_names[op.input_names.index(orig)] = new

            if orig in returns.values():
                returns[list(returns.keys())[list(returns.values()).index(orig)]] = new
                if orig in returns_requiring_batch_dim_fix:
                    returns_requiring_batch_dim_fix[returns_requiring_batch_dim_fix.index(orig)] = new

        submodule = False
        param_names = {}
        const_names = {}
        # if compiler_cfg.tvm_module_to_num_patterns.get(framework_mod.get_name(), None):
        #     match_subgraph_patterns = compiler_cfg.tvm_module_to_num_patterns[framework_mod.get_name()]

        #     ret = pypattern_matcher.lower_json_to_pattern_matcher(graph, match_subgraph_patterns)
        #     subgraph_matches = ret.subgraph_matches

        #     if len(subgraph_matches) > 1:
        #         submodule = True

        #         matched_params = {}
        #         matched_consts = {}
        #         matched_ops = {}
        #         submodule_input_ports = {}
        #         submodule_outputs = {}
        #         submodule_outputs_requiring_batch_dim_fix = []
        #         submodule_output_shapes = {}

        #         # Collect submodule IOs
        #         for orig_nid in subgraph_matches[0].keys():
        #             node = graph["nodes"][orig_nid]
        #             if "num_inputs" in node["attrs"]:
        #                 for input_port in range(int(node["attrs"]["num_inputs"])):
        #                     if node["inputs"][input_port][0] not in subgraph_matches[0]:
        #                         submodule_input_ports[orig_nid] = input_port

        #             node = graph["nodes"][orig_nid]
        #             if "users" in node:
        #                 for user in node["users"]:
        #                     if user not in subgraph_matches[0] and node["op"] != "*":
        #                         submodule_outputs[node["nid"]] = node["forge_name"]
        #                         if node["forge_shape"][0] != 1:
        #                             submodule_outputs_requiring_batch_dim_fix.append(node["forge_name"])

        #         # add ops for each submodule call
        #         idx = max(sorted(submodule_input_ports)) + 0.5
        #         input_nids = list(sorted(submodule_input_ports.keys()))

        #         input_nodes = [graph["nodes"][input_nid] if submodule_input_ports[input_nid] == -1 else graph["nodes"][graph["nodes"][input_nid]["inputs"][submodule_input_ports[input_nid]][0]] for input_nid in input_nids]
        #         submodule_inputs = {input_node["nid"]:input_node["forge_name"] for input_node in input_nodes}
        #         activations = [input_node_name for _, input_node_name in sorted(graph_input_names.items())]

        #         ops[idx] = Operation(
        #             function_name="self.layers[0]",
        #             output_name="layer_0",
        #             input_names=activations,
        #         )
        #         ops[idx].is_submodule_call = True

        #         output_nids = list(submodule_outputs.keys())
        #         assert len(output_nids) == 1, "TODO"

        #         for i in range(1, len(subgraph_matches)):
        #             #if the input node is in the submodule
        #             activations = []
        #             for input_nid in input_nids:
        #                 if submodule_input_ports[input_nid] == -1:
        #                     matched_nid = subgraph_matches[i][input_nid]
        #                 else:
        #                     matched_user = subgraph_matches[i][input_nid]
        #                     matched_nid = graph["nodes"][matched_user]["inputs"][submodule_input_ports[input_nid]][0]

        #                 idx = matched_nid + 0.5
        #                 activations.append(graph["nodes"][matched_nid]["forge_name"])

        #             # unlike ops, submodules should not have repeated inputs
        #             activations = list(dict.fromkeys(activations))
        #             ops[idx] = Operation(
        #                 function_name=f"self.layers[{i}]",
        #                 output_name=f"layer_{i}",
        #                 input_names=activations,
        #             )
        #             ops[idx].is_submodule_call = True

        #         # build submodule param / op dicts, remove from main
        #         for orig_nid in subgraph_matches[0].keys():
        #             if orig_nid in params:
        #                 matched_params[orig_nid] = params[orig_nid]
        #                 param_name = params[orig_nid][0]
        #                 param_names[param_name] = (f"layer_{0}", param_name)
        #                 del params[orig_nid]
        #                 for index, subgraph in enumerate(subgraph_matches[1:]):
        #                     param_names[params[subgraph[orig_nid]][0]] = (f"layer_{index + 1}", param_name)
        #                     del params[subgraph[orig_nid]]

        #             if orig_nid in constants:
        #                 matched_consts[orig_nid] = constants[orig_nid]
        #                 const_name = constants[orig_nid][0]
        #                 const_names[const_name] = (f"layer_{0}", const_name)
        #                 del constants[orig_nid]
        #                 for index, subgraph in enumerate(subgraph_matches[1:]):
        #                     const_names[constants[subgraph[orig_nid]][0]] = (f"layer_{index + 1}", const_name)
        #                     del constants[subgraph[orig_nid]]

        #             if orig_nid in ops:
        #                 matched_ops[orig_nid] = ops[orig_nid]
        #                 del ops[orig_nid]
        #                 for subgraph in subgraph_matches[1:]:
        #                     del ops[subgraph[orig_nid]]

        #         #replace references to outputs of each submodule with submodule
        #         for idx, subgraph in enumerate(subgraph_matches):
        #             name_to_replace = graph["nodes"][subgraph[output_nids[0]]]["forge_name"]

        #             replace_node_name(name_to_replace, f"layer_{idx}")

        # Some float types (e.g. tf.bfloat16) are not compatible with numpy
        # We must signal to the ForgeWriter if the model contains these types so it can implement the workaround
        contains_incompatible_np_floats = False
        if framework == "tensorflow":
            for weight in framework_mod.module.weights:
                if weight.dtype in ForgeWriter.incompatible_np_float_types:
                    contains_incompatible_np_floats = True

        current_module_name = module_name
        if current_module_name is None:
            current_module_name = graph_name

        if len(json_graphs) > 1:
            current_module_name += f"_{json_graph['device']}_{graph_index}"

        if json_graph["device"] == "tt":
            delete_inputs = not verify_cfg.enable_op_level_comparision
            if not delete_inputs:
                logger.warning(
                    "Preserving Intermediate tensor values in ForgeModule forward may cause out-of-memory issues"
                )
            writer = ForgeWriter(
                current_module_name,
                framework,
                contains_incompatible_np_floats=contains_incompatible_np_floats,
                delete_inputs=delete_inputs,
                df_override_dtype=df_override_dtype,
            )
        else:
            writer = PyTorchWriter(current_module_name, source_framework=framework)

        writer.write_header()

        if submodule:
            writer.write_class_definition(matched_params, matched_consts, class_name="Submodel", is_submodel=True)
            if is_cpu_pre:
                writer.write_forward(
                    matched_ops, submodule_inputs, submodule_outputs, submodule_outputs_requiring_batch_dim_fix
                )
            else:
                writer.write_forward(matched_ops, submodule_inputs, submodule_outputs)
            writer.write_class_definition(params, constants, num_submodels=len(subgraph_matches))
        else:
            writer.write_class_definition(params, constants)

        # can submodules be called in a loop? IE one outputs into the next
        loop_start = None
        prev_op = None
        for key in sorted(ops):
            op = ops[key]
            if op.is_submodule_call:
                if prev_op is None:
                    loop_start = op
                    prev_op = op
                elif prev_op.output_name in ops[key].input_names:
                    if len(loop_start.loop_with) == 0:
                        input_index = ops[key].input_names.index(prev_op.output_name)
                        loop_start.output_name = loop_start.input_names[input_index]
                        start, end = re.search("\[\d\]", loop_start.function_name).span()
                        loop_start.loop_start_index = int(loop_start.function_name[start + 1 : end - 1])
                        loop_start.function_name = re.sub("\[\d\]", "[i]", loop_start.function_name)

                    loop_start.loop_with.append(ops[key])
                    prev_op = ops[key]
                    del ops[key]
                else:
                    loop_start = None
                    prev_op = None
            else:
                loop_start = None
                prev_op = None

        # if there are any unconsumed inputs in the framework graph, they will not be part of the
        # generated graph, so we should add dummy variables to cunsume them. This is only needed for
        # the first module.
        if graph_index == 0:
            for input_index, _ in enumerate(forge_inputs):
                if input_index not in graph_input_names:
                    graph_input_names[input_index] = f"unused_input_{input_index}"

        for key in sorted(ops):
            if len(ops[key].loop_with):
                replace_node_name(ops[key].loop_with[-1].output_name, ops[key].output_name)

        def delete_unneeded_outputs(ops, returns):
            consumers = {}
            op_outputs = set()
            for ret in returns.values():
                consumers[ret] = 1
            for op in ops.values():
                op_outputs.add(op.output_name)
                for input_name in op.input_names:
                    if input_name not in consumers:
                        consumers[input_name] = 1
                    else:
                        consumers[input_name] = consumers[input_name] + 1
            for key in sorted(ops):
                for input_name in ops[key].input_names:
                    if input_name not in op_outputs:
                        continue
                    assert input_name in consumers
                    consumers[input_name] = consumers[input_name] - 1
                    if consumers[input_name] == 0:
                        ops[key].inputs_to_delete.append(input_name)

        delete_unneeded_outputs(ops, returns)
        if is_cpu_pre:
            writer.write_forward(ops, graph_input_names, returns, returns_requiring_batch_dim_fix)
        else:
            writer.write_forward(ops, graph_input_names, returns)

        param_file_name = None
        if len(params_from_tvm):
            param_file_name = os.path.join(writer.module_directory, writer.module_name + "_params.pt")
            torch.save(params_from_tvm, param_file_name)

        param_names.update(const_names)
        writer.write_param_parser(param_names, param_file_name)

        writer.close_file()

        modules.append(writer)

        if compiler_cfg.extract_tvm_unique_ops_config:

            # Running the inference to verify the generated forge module which helps
            # to avoid extracting unique ops configuration for not properly traced models
            file_path = os.path.join(writer.module_directory, writer.filename)
            module = import_from_path(writer.module_name, file_path)

            TestClass = getattr(module, writer.class_name)
            forge_mod = TestClass(writer.module_name)
            forge_mod.process_framework_parameters(framework_mod.module)

            framework_mod.cpu_eval_forward(*inputs)
            get_forge_outputs([forge_mod], ["TTDevice"], forge_inputs)

            extract_and_export_unique_ops_config(
                framework_mod,
                ops,
                current_module_name,
                framework,
                node_name_to_node_type,
                param_file_name,
                compiler_cfg,
            )

            raise ExitTest("Extract the unique ops configuration and exited the test")

    if compiler_cfg.retain_tvm_python_files:
        save_writers_metadata(modules, flattened_pytorch_inputs, forge_inputs, graph_name)

    return modules, forge_inputs
