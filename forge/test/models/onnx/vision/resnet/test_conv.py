# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
import torch
import forge
from forge.config import CompilerConfig, MLIRConfig
from forge.verify.verify import verify


def _build_conv7x7() -> tuple[onnx.ModelProto, list[np.ndarray]]:
    """Return a Conv7x7 + Relu ONNX model and a deterministic sample input."""
    in_ch, out_ch = 3, 16
    height, width, kernel = 384, 1664, 7
    stride, pads = [2, 2], [2, 2, 3, 3]

    rng = np.random.default_rng(42)
    weight = rng.standard_normal((out_ch, in_ch, kernel, kernel)).astype(np.float32)
    bias = rng.standard_normal((out_ch,)).astype(np.float32)

    out_h = (height + pads[0] + pads[2] - kernel) // stride[0] + 1
    out_w = (width + pads[1] + pads[3] - kernel) // stride[1] + 1

    graph = oh.make_graph(
        nodes=[
            oh.make_node(
                "Conv",
                inputs=["input", "weight", "bias"],
                outputs=["conv_out"],
                name="Conv7x7_0",
                kernel_shape=[kernel, kernel],
                strides=stride,
                pads=pads,
                group=1,
            ),
            oh.make_node("Relu", inputs=["conv_out"], outputs=["output"], name="Relu_0"),
        ],
        name="conv7x7_reproducer",
        inputs=[oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, in_ch, height, width])],
        outputs=[oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, out_ch, out_h, out_w])],
        initializer=[
            onh.from_array(weight, name="weight"),
            onh.from_array(bias, name="bias"),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    onnx.checker.check_model(model)

    sample_input = np.random.default_rng(0).standard_normal((1, in_ch, height, width)).astype(np.float32)
    return model, [sample_input]


def test_conv7x7_relu():

    model, np_inputs = _build_conv7x7()
    inputs = [torch.from_numpy(arr) for arr in np_inputs]

    onnx_module = forge.OnnxModule("conv7x7_relu", model)
    compiler_cfg = CompilerConfig(mlir_config=MLIRConfig().set_enable_optimizer(True))
    compiled_model = forge.compile(
        onnx_module,
        sample_inputs=inputs,
        compiler_cfg=compiler_cfg,
    )

    verify(
        inputs,
        onnx_module,
        compiled_model,
    )