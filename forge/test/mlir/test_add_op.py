# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for a single ONNX Add through the tt-forge-onnx compile flow.

Deliberately split into two halves. The structural tests name a target architecture,
so they compile without touching hardware and assert on what the compiler actually
produced -- the TTNN op stream inside the flatbuffer. The numerical test needs a
device and only checks that the answer is right.

Keeping them apart matters: a compile can succeed and still lower to the wrong ops,
and a numerical pass on Wormhole says nothing about what a Quasar compile emitted.

`scripts/add_op_walkthrough.py` steps through the same compile interactively.
"""

import json

import pytest
import torch
from onnx import TensorProto, helper

import forge
from forge.config import CompilerConfig, MLIRConfig
from forge.verify.verify import verify

ONNX_OPSET_VERSION = 21
SHAPE = [2, 32, 32]

ARCHS = [forge._C.Arch.WORMHOLE_B0, forge._C.Arch.BLACKHOLE, forge._C.Arch.QUASAR]


def arch_slug(arch):
    """`Arch.WORMHOLE_B0` -> `wormhole_b0`.

    module_name becomes a generated Python class name, so it must not contain the
    dot that str(Arch.X) carries.
    """
    return str(arch).split(".")[-1].lower()


def build_add_model(shape=SHAPE):
    """A single-node ONNX Add graph -- the smallest thing that exercises the whole flow."""
    input_a = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, shape)
    input_b = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    node = helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(nodes=[node], name="AddGraph", inputs=[input_a, input_b], outputs=[output])
    return helper.make_model(
        graph,
        producer_name="AddModel",
        opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)],
    )


def compile_for(arch, module_name):
    """Compile the Add for a named arch. Touches no hardware."""
    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(arch))
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
    return forge.compile(build_add_model(), inputs, module_name=module_name, compiler_cfg=cfg)


def op_types(compiled_model):
    """The TTNN op stream the compiler emitted, read back out of the flatbuffer."""
    binary = json.loads(compiled_model.compiled_binary.as_json())
    return [op["type_type"] for op in binary["programs"][0]["operations"]]


# ---------------------------------------------------------------------------
# Structural — no device
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.parametrize("arch", ARCHS, ids=arch_slug)
def test_add_lowers_to_ttnn_add(arch):
    """The Add must survive lowering as an eltwise binary, on every supported arch.

    Asserting on the op stream rather than just "did it compile" is the point: a
    decomposition regression would still compile cleanly.
    """
    ops = op_types(compile_for(arch, f"add_{arch_slug(arch)}"))
    assert "EltwiseBinaryOp" in ops, f"Add did not lower to an eltwise binary; got {ops}"


@pytest.mark.push
def test_add_flatbuffer_is_self_describing():
    """The binary carries everything the runtime needs to replay it without the compiler."""
    binary = json.loads(compile_for(forge._C.Arch.WORMHOLE_B0, "add_fb").compiled_binary.as_json())

    for key in ("version", "schema_hash", "system_desc", "programs"):
        assert key in binary, f"flatbuffer missing {key}"

    program = binary["programs"][0]
    assert program["inputs"], "no input tensor descriptors"
    assert program["outputs"], "no output tensor descriptors"
    assert program["operations"], "no operations"

    # The TTNN module is embedded, which is what makes the binary debuggable.
    assert binary["mlir"]["name"] == "ttnn"
    assert "ttnn.add" in binary["mlir"]["source"]


@pytest.mark.push
def test_add_emits_layout_conversions_and_deallocates():
    """Around the add sit the layout changes and liveness deallocates.

    Locks in that the compiler is doing memory management rather than leaking every
    intermediate -- a silent regression that only shows up as OOM on a real model.
    """
    ops = op_types(compile_for(forge._C.Arch.WORMHOLE_B0, "add_layout"))
    assert "ToLayoutOp" in ops, f"expected a layout conversion; got {ops}"
    assert "DeallocateOp" in ops, f"expected deallocates; got {ops}"
    assert ops.index("EltwiseBinaryOp") > ops.index("ToLayoutOp"), "inputs must be laid out before the add"


@pytest.mark.push
def test_add_compiles_identically_for_every_arch():
    """Same op stream on every arch -- the architecture is data, not control flow.

    If this ever fails, an arch conditional has appeared in the lowering path, which
    is worth knowing about deliberately rather than discovering later.
    """
    streams = {arch_slug(a): op_types(compile_for(a, f"add_cmp_{arch_slug(a)}")) for a in ARCHS}
    reference = streams[arch_slug(ARCHS[0])]
    for name, stream in streams.items():
        assert stream == reference, f"{name} lowered differently: {stream} vs {reference}"


# ---------------------------------------------------------------------------
# Numerical — needs a device
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_add_numerical():
    """The answer is right, on whatever device is attached."""
    onnx_model = build_add_model()
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]

    onnx_module = forge.OnnxModule("add", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs, module_name="add_numerical")

    verify(inputs, onnx_module, compiled_model)
