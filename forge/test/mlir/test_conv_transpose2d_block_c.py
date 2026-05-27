"""
Standalone test for ConvTranspose2d ops extracted from block_C_cylinder_backbone.

All 4 ops share the same config:
  - weight: in_ch=192 out_ch=192, kernel=2x2, stride=2x2, pad=0, dilation=1,
            groups=1, no bias, dtype=f32

Input shapes (NCHW):
  op0: 1x192x10x18  -> 1x192x20x36
  op1: 1x192x20x36  -> 1x192x40x72
  op2: 1x192x40x72  -> 1x192x80x144  (failing at opt_level_2)
  op3: 1x192x80x144 -> 1x192x160x288
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig

ONNX_OPSET = 11


# ---------------------------------------------------------------------------
# Helper: build single-node ONNX ConvTranspose model
# ---------------------------------------------------------------------------

def make_conv_transpose_onnx(
    batch, in_channels, out_channels,
    in_h, in_w,
    kernel, stride, pad, dilation, groups, output_padding,
    name="conv_transpose",
):
    """Return an onnx.ModelProto with a single ConvTranspose node and fixed weights."""
    np.random.seed(42)
    weight_np = np.random.randn(
        in_channels, out_channels // groups, kernel[0], kernel[1]
    ).astype(np.float32)

    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [batch, in_channels, in_h, in_w])
    W = onh.from_array(weight_np, name="W")

    out_h = (in_h - 1) * stride[0] - 2 * pad[0] + dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
    out_w = (in_w - 1) * stride[1] - 2 * pad[1] + dilation[1] * (kernel[1] - 1) + output_padding[1] + 1
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [batch, out_channels, out_h, out_w])

    node = oh.make_node(
        "ConvTranspose",
        inputs=["X", "W"],
        outputs=["Y"],
        name=name,
        kernel_shape=kernel,
        strides=stride,
        pads=[pad[0], pad[1], pad[0], pad[1]],
        dilations=dilation,
        group=groups,
        output_padding=output_padding,
    )

    graph = oh.make_graph([node], name, [X], [Y], initializer=[W])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", ONNX_OPSET)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Compiler config helpers
# ---------------------------------------------------------------------------

def _cfg_opt_level_0():
    cfg = CompilerConfig()
    return cfg


def _cfg_opt_level_1():
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(1)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    return cfg


def _cfg_opt_level_2():
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    return cfg


_CFGS = {
    "opt_level_0": _cfg_opt_level_0,
    "opt_level_1": _cfg_opt_level_1,
    "opt_level_2": _cfg_opt_level_2,
}

# ---------------------------------------------------------------------------
# Block-C ConvTranspose2d configurations (all ops are identical in structure)
# ---------------------------------------------------------------------------

BLOCK_C_OPS = [
    dict(in_h=10,  in_w=18,  tag="op0_10x18"),
    dict(in_h=20,  in_w=36,  tag="op1_20x36"),
    dict(in_h=40,  in_w=72,  tag="op2_40x72"),
    dict(in_h=80,  in_w=144, tag="op3_80x144"),
]

COMMON = dict(
    batch=1, in_channels=192, out_channels=192,
    kernel=[2, 2], stride=[2, 2], pad=[0, 0],
    dilation=[1, 1], groups=1, output_padding=[0, 0],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opt_cfg", list(_CFGS.keys()))
@pytest.mark.parametrize("op_cfg", BLOCK_C_OPS, ids=[c["tag"] for c in BLOCK_C_OPS])
def test_block_c_conv_transpose2d(op_cfg, opt_cfg):
    """Compile each block-C ConvTranspose2d at all opt levels."""
    in_h, in_w = op_cfg["in_h"], op_cfg["in_w"]
    model = make_conv_transpose_onnx(in_h=in_h, in_w=in_w, **COMMON)

    sample_input = torch.randn(1, 192, in_h, in_w)
    compiler_cfg = _CFGS[opt_cfg]()

    # compile() raises on failure — no verify needed to catch the opt_level_2 bug
    forge.compile(model, sample_inputs=[sample_input], compiler_cfg=compiler_cfg,
                  module_name=f"conv_transpose2d_{op_cfg['tag']}_{opt_cfg}")
