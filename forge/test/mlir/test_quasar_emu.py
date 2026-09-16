# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-op Add executed on the Quasar ZeBu hardware emulator.

Companion to ``test_quasar_sim.py``. That file runs on craq-sim, a functional C++
model; this one runs the same graph on **real Quasar RTL** compiled onto a Synopsys
ZeBu-Server5. Develop on the simulator, confirm here.

What this actually exercises
----------------------------
``ttnn.add`` emitted by forge is dispatched, in the tt-mlir runtime, to Quasar's own
op rather than the mainline one::

    return utils::isQuasar()
               ? ::ttnn::operations::experimental::quasar::binary::add(args...)
               : ::ttnn::add(args...);
                    -- runtime/lib/ttnn/operations/eltwise/binary/binary.cpp

That substitution is the point of the test. Mainline ``::ttnn::add`` is not merely
slower on Quasar, it is refused: its program factory constructs a
``DataMovementKernel``, whose constructor ``TT_FATAL``s with "not supported on Quasar.
Use QuasarDataMovementKernel". ``to_layout`` is dispatched the same way, so a passing
Add here covers both call sites plus the separate output-readback path.

Selecting the emulator
----------------------
``TT_METAL_SIMULATOR`` picks the target purely by file type
(``simulation_device_factory.cpp``): a path to a **.so** gives craq-sim, in-process; a
path to a **directory** gives ZeBu, over a socket. Nothing on the forge side is
emulator-specific.

    source /proj_sw/user_dev/$USER/emu-run/env_quasar_emu.sh
    export TT_METAL_HOME=<repo>/third_party/tt-mlir/third_party/tt-metal/src/tt-metal
    cd "$TT_METAL_HOME"
    timeout 3600 pytest -svv <repo>/forge/test/mlir/test_quasar_emu.py

In its OWN pytest process. tt-metal's ``RunTimeOptions`` and forge's ``TTSystem`` are
both construct-once-per-process singletons, so the first test to touch a device fixes
the target for the whole session. Kept out of ``pytest.ini``'s testpaths for that
reason, exactly as ``test_quasar_sim.py`` is.

Cost, so a healthy run is not mistaken for a hang
-------------------------------------------------
ZeBu is physically shared and licensed by token, one job at a time. Budget:

  * queue for hardware   0 -> 30+ min  ("Waiting for ack msg from remote...")
  * model load           38-66 s       (once per process)
  * per program launch   ~5-25 s

A single Add is 7 ops and 4 program launches. Always use a wall-clock ``timeout``: a
completion wait against a simulator target is unbounded by design, so a genuine hang
never returns on its own.

bf16, not f32
-------------
Measured on the emulator 2026-09-15: the stock f32 path hangs. One program launches,
then the host blocks forever in ``wait_until_cores_done`` -> ``Cluster::read_core``,
polling a done-flag that ``ttnn::prim::qsr::TilizeDeviceOperation`` never sets. The
emulator stays healthy throughout; it is the host that waits. See ``test_add_f32``
below, which is skipped rather than xfailed -- an xfail on a run that never returns
stalls the suite instead of reporting it.
"""

import os

import pytest
import torch
from onnx import TensorProto, helper

import forge

ONNX_OPSET_VERSION = 21
opset_imports = [helper.make_operatorsetid("", ONNX_OPSET_VERSION)]

SHAPE = [2, 32, 32]

# Add on ZeBu measures 0.999985, identical to craq-sim. 0.99 leaves room for an
# honest bf16 result while still failing a genuinely wrong one.
MIN_PCC = 0.99


def _quasar_emulator_available() -> bool:
    """True when the ZeBu emulator environment is present in THIS process.

    A directory (not a .so) in TT_METAL_SIMULATOR is what makes UMD choose ZeBu, and
    NNG_SOCKET_ADDR is how the remote testbench dials back. Checking all three keeps
    a misconfigured run from silently landing on craq-sim or on silicon.
    """
    simulator = os.environ.get("TT_METAL_SIMULATOR", "")
    if not simulator or not os.path.isdir(simulator):
        return False
    if not os.environ.get("NNG_SOCKET_ADDR"):
        return False
    return os.environ.get("ARCH_NAME", "").lower() == "quasar"


pytestmark = [
    pytest.mark.quasar_emu,
    pytest.mark.skipif(
        not _quasar_emulator_available(),
        reason=(
            "Quasar emulator not configured; "
            "source emu-run/env_quasar_emu.sh (TT_METAL_SIMULATOR must be a DIRECTORY)"
        ),
    ),
]


@pytest.fixture(autouse=True)
def _tt_metal_cwd():
    """Run inside $TT_METAL_HOME, which the JIT kernel include paths require.

    Quasar's binary_ng factory hands tt-metal include paths that are relative to the
    tt-metal root, and they are resolved against the process cwd rather than
    TT_METAL_HOME. tt-metal's own suite always runs from its repo root so this never
    bites there; forge is a different repo. Restores the cwd afterwards.
    """
    home = os.environ.get("TT_METAL_HOME")
    if not home or not os.path.isdir(home):
        pytest.skip("TT_METAL_HOME is not set; source emu-run/env_quasar_emu.sh")
    prev = os.getcwd()
    os.chdir(home)
    try:
        yield
    finally:
        os.chdir(prev)


def _add_onnx():
    """A one-node Add ONNX graph, float32 in and out.

    The graph stays f32; bf16 is imposed by the compiler config, which is what routes
    Quasar's binary_ng FPU kernel.
    """
    node = helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name="AddGraph",
        inputs=[
            helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE),
            helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
    )
    return helper.make_model(
        graph, producer_name="AddModel", opset_imports=opset_imports
    )


def test_device_is_quasar():
    """Fail fast, and cheaply, if this process is not actually on a Quasar device.

    Without this a misconfiguration shows up as a puzzling numerics or dispatch
    failure much later, after the expensive model load. No device work beyond the
    arch query, so it costs nothing on top of opening the device.
    """
    from forge._C.runtime.experimental import TTSystem

    devices = TTSystem.get_system().devices
    assert devices, "no device found; is the emulator env sourced?"
    arch = str(devices[0].arch)
    assert "QUASAR" in arch.upper(), f"expected a Quasar device, got {arch}"


def test_add_bf16():
    """A single eltwise Add executes on the ZeBu emulator and matches torch.

    This is the whole point of the file: it only passes if ``ttnn.add`` reached
    ``ttnn::operations::experimental::quasar::binary::add``. On the mainline path the
    run does not produce a wrong number, it aborts in the DataMovementKernel
    constructor -- so a PCC this close to 1 is itself evidence the substitution
    happened.

    An explicit PCC assert rather than verify(), so the measured correlation is in the
    test output; on hardware that takes minutes per run, you want the number even when
    the test passes.
    """
    from forge.config import CompilerConfig

    torch.manual_seed(0)
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
    reference = inputs[0] + inputs[1]

    cfg = CompilerConfig()
    cfg.default_df_override = forge._C.DataFormat.Float16_b

    compiled = forge.compile(
        _add_onnx(), inputs, module_name="add_bf16_quasar_emu", compiler_cfg=cfg
    )

    out = compiled(*inputs)
    got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()

    expected, actual = reference.flatten().float(), got.flatten()
    pcc = torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()
    max_abs_err = (expected - actual).abs().max().item()
    print(f"[quasar-emu] Add bf16: pcc={pcc:.6f} max_abs_err={max_abs_err:.5f}")

    assert pcc > MIN_PCC, f"Add bf16 on the Quasar emulator: pcc={pcc}"


@pytest.mark.skip(
    reason=(
        "f32 hangs on Quasar: qsr::TilizeDeviceOperation never sets RUN_MSG_DONE, so "
        "the host blocks forever in wait_until_cores_done -> Cluster::read_core. "
        "Skipped, not xfailed -- the run never returns, so an xfail would stall the "
        "suite and hold the shared ZeBu tokens rather than report a failure."
    )
)
def test_add_f32():
    """Same graph without the bf16 override. Enable once the f32 tilize is fixed."""
    torch.manual_seed(0)
    inputs = [torch.rand(SHAPE), torch.rand(SHAPE)]
    reference = inputs[0] + inputs[1]

    compiled = forge.compile(_add_onnx(), inputs, module_name="add_f32_quasar_emu")

    out = compiled(*inputs)
    got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu()
    expected, actual = reference.flatten().float(), got.flatten()
    pcc = torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()

    assert pcc > MIN_PCC, f"Add f32 on the Quasar emulator: pcc={pcc}"
