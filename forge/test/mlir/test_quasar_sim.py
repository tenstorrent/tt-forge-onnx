# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-op smoke tests executed on the craq-sim Quasar simulator.

There is no Quasar silicon, so this is the only way to run -- rather than merely
compile -- a Forge graph for Quasar. The simulator presents a virtual QSR device to
UMD, so nothing here is Quasar-specific on the forge side: the ops go through the
ordinary compile-and-verify path and land on a device that happens to be simulated.

Running these
-------------
    source ./scripts/quasar_sim_env.sh
    pytest -svv forge/test/mlir/test_quasar_sim.py

Budget hours, not minutes. Compilation for Quasar takes seconds; execution on a
cycle-accurate simulator is what costs, and craq-sim's own Quasar op CI allows 240
minutes per run. A sub-hour timeout will kill a healthy run, and a killed run is
indistinguishable from a hang.

In their OWN pytest process. tt-metal's RunTimeOptions and forge's TTSystem are both
construct-once-per-process singletons, so the first test to touch a device fixes
hardware-vs-simulator for the whole session -- mixing these with ordinary tests would
silently run one of the two groups against the wrong target. This file is deliberately
left out of pytest.ini's testpaths so a bare `pytest` never collects it.

The skip is decided at collection time rather than in a fixture, because the root
conftest's autouse property-recorder fixture already probes the device and there is no
ordering guarantee that would let a fixture here run first.
"""

import os

import pytest
import torch
from onnx import TensorProto, helper

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

ONNX_OPSET_VERSION = 21
opset_imports = [helper.make_operatorsetid("", ONNX_OPSET_VERSION)]

SHAPE = [2, 32, 32]


def _quasar_sim_available() -> bool:
    """True when scripts/quasar_sim_env.sh has been sourced into this process."""
    simulator = os.environ.get("TT_METAL_SIMULATOR", "")
    if not simulator.endswith("libttsim.so") or not os.path.isfile(simulator):
        return False
    return os.environ.get("ARCH_NAME", "").lower() == "quasar"


pytestmark = [
    pytest.mark.quasar_sim,
    pytest.mark.skipif(
        not _quasar_sim_available(),
        reason="Quasar simulator not configured; run `source ./scripts/quasar_sim_env.sh` first",
    ),
]


def _run_binary_op(op_type: str, name: str):
    """Compile and numerically verify a single-node binary ONNX graph."""
    input_a = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE)
    input_b = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    node = helper.make_node(op_type, inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name=f"{op_type}Graph",
        inputs=[input_a, input_b],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name=f"{op_type}Model",
        opset_imports=opset_imports,
    )

    # Div: keep the divisor away from zero so a verify failure means a Quasar
    # problem rather than a division blowing up.
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE) + 1.0]

    onnx_module = forge.OnnxModule(name, onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, onnx_module, compiled_model)


def _run_unary_op(op_type: str, name: str, pcc: float = 0.99):
    """Compile and numerically verify a single-node unary ONNX graph."""
    input_a = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    node = helper.make_node(op_type, inputs=["input_A"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name=f"{op_type}Graph",
        inputs=[input_a],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name=f"{op_type}Model",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand(SHAPE) - 0.5]

    onnx_module = forge.OnnxModule(name, onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)
    verify(
        inputs,
        onnx_module,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )


# ---------------------------------------------------------------------------
# Known green: eltwise binary is dispatched to the Quasar op library
# (ttnn::operations::experimental::quasar) and both compiles and verifies.
# ---------------------------------------------------------------------------


def test_add():
    _run_binary_op("Add", "quasar_add")


def test_mul():
    _run_binary_op("Mul", "quasar_mul")


def test_sub():
    _run_binary_op("Sub", "quasar_sub")


def test_div():
    _run_binary_op("Div", "quasar_div")


def test_relu():
    """Green, but only because it is rewritten rather than dispatched.

    Quasar has no unary op family under experimental/quasar/ at all -- only binary
    and binary_ng -- so relu is emitted as add(x, 0) with relu fused as an LHS
    activation. relu(x) + 0 == relu(x), and adding 0.0f is exact in bf16.

    PCC is relaxed to 0.95: that is what this rewrite measures on craq-sim, and
    holding it to the 0.99 default would fail for a reason that has nothing to do
    with whether the op works.
    """
    _run_unary_op("Relu", "quasar_relu", pcc=0.95)


# ---------------------------------------------------------------------------
# Known red. xfail rather than omit, so the list is forced to stay honest --
# pytest.ini sets xfail_strict, so an op that starts passing fails the run.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="comparison SFPU kernels are guarded #ifndef ARCH_QUASAR, so Greater has " "no kernel on Quasar",
    strict=True,
)
def test_greater():
    input_a = helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE)
    input_b = helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE)
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, SHAPE)

    node = helper.make_node("Greater", inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name="GreaterGraph",
        inputs=[input_a, input_b],
        outputs=[output],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="GreaterModel",
        opset_imports=opset_imports,
    )

    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]

    onnx_module = forge.OnnxModule("quasar_greater", onnx_model)
    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, onnx_module, compiled_model)


# conv2d is deliberately absent: it is blocked in tt-metal (the
# conv_bmm_tilize_metal2 deadlock, tt-metal #48552) and does not reach a result at
# all, so an xfail would misrepresent it as a test that runs and fails.
