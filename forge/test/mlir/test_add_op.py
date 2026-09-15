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


def compile_for_dtype(arch, module_name, df):
    """Compile the Add for a named arch with a data-format override. No hardware."""
    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(arch))
    cfg.default_df_override = df
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
    return forge.compile(build_add_model(), inputs, module_name=module_name, compiler_cfg=cfg)


def op_types(compiled_model):
    """The TTNN op stream the compiler emitted, read back out of the flatbuffer."""
    binary = json.loads(compiled_model.compiled_binary.as_json())
    return [op["type_type"] for op in binary["programs"][0]["operations"]]


def ttnn_source(compiled_model):
    """The final TTNN module, which the flatbuffer carries verbatim."""
    binary = json.loads(compiled_model.compiled_binary.as_json())
    return binary["mlir"]["source"]


def chip_desc(compiled_model):
    """The chip descriptor the compile was performed against."""
    binary = json.loads(compiled_model.compiled_binary.as_json())
    return binary["system_desc"]["chip_descs"][0]


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
# Data format — no device
#
# The dtype is not a detail for Quasar: f32 makes is_binary_sfpu_op true for every
# op including add, so it routes the SFPU kernel, while bf16 takes the FPU binary_ng
# kernel. Only bf16 currently executes (see test_quasar_sim.py), so the
# compiler emitting the dtype that was asked for is worth asserting device-free
# rather than discovering on a simulator run that takes minutes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS, ids=arch_slug)
def test_add_defaults_to_f32(arch):
    """With no override the Add stays float32, matching the ONNX graph."""
    compiled = compile_for(arch, f"add_default_df_{arch_slug(arch)}")
    source = ttnn_source(compiled)

    assert '"ttnn.add"' in source
    add_line = next(line for line in source.splitlines() if '"ttnn.add"' in line)
    assert "xf32," in add_line, f"expected f32 operands, got: {add_line.strip()}"


@pytest.mark.parametrize("arch", ARCHS, ids=arch_slug)
def test_add_honours_bf16_override(arch):
    """default_df_override reaches the emitted TTNN tensors, not just the graph.

    This is the knob that makes add executable on Quasar, so it has to be verified
    where it lands -- in the IR -- rather than where it is set.
    """
    compiled = compile_for_dtype(
        arch, f"add_bf16_{arch_slug(arch)}", forge._C.DataFormat.Float16_b
    )
    source = ttnn_source(compiled)

    add_line = next(line for line in source.splitlines() if '"ttnn.add"' in line)
    assert "xbf16," in add_line, f"expected bf16 operands, got: {add_line.strip()}"
    assert "xf32," not in add_line

    # The override must not perturb the op stream: same ops, different element type.
    assert op_types(compiled) == op_types(compile_for(arch, f"add_cmp_{arch_slug(arch)}"))


def test_quasar_descriptor_excludes_block_float():
    """Quasar's descriptor must not advertise formats the device cannot run.

    tt-metal's is_supported_quasar excludes Bfp2/Bfp4/Bfp8 and their _b variants --
    Quasar's narrow formats are MX -- and it has no unsigned 16/32-bit device format.
    Advertising them makes every legality check downstream believe bf8_b is available
    and defers the failure to a tt-metal host format-validator throw.
    """
    compiled = compile_for(forge._C.Arch.QUASAR, "add_desc_quasar")
    formats = chip_desc(compiled)["supported_data_types"]

    assert formats == ["Float32", "Float16", "BFloat16", "UInt8", "Int32"], formats
    assert not [f for f in formats if f.startswith("BFP")]
    assert "UInt16" not in formats and "UInt32" not in formats


def test_wormhole_descriptor_keeps_block_float():
    """The counterpart: narrowing Quasar must not have narrowed Wormhole."""
    compiled = compile_for(forge._C.Arch.WORMHOLE_B0, "add_desc_wormhole")
    formats = chip_desc(compiled)["supported_data_types"]

    assert len(formats) == 13, formats
    for expected in ("BFP_BFloat8", "BFP_BFloat4", "UInt16", "UInt32"):
        assert expected in formats


def test_quasar_rejects_bf8_weight_override():
    """Asking for a format Quasar lacks must fail at config time, not in the runtime."""
    cfg = CompilerConfig(
        mlir_config=MLIRConfig()
        .set_target_arch(forge._C.Arch.QUASAR)
        .set_experimental_weight_dtype(forge._C.DataFormat.Bfp8_b)
    )
    with pytest.raises(Exception, match="Quasar"):
        forge.compile(
            build_add_model(),
            [torch.rand(SHAPE), torch.rand(SHAPE)],
            module_name="add_bfp8_quasar",
            compiler_cfg=cfg,
        )


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
