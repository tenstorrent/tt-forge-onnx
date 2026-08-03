# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Slice×2 → Conv2d×2 → Clip×2 → Concat pattern compiled through Forge.

Extracted ONNX subgraphs from the BEV model using onnx.utils.extract_model:

  Block A cam0  (block_A_cam0_slice_conv_pattern.onnx)
    AvgPool → Slice×2 → Conv2d×2(groups=1) → Clip×2 → Concat
    input  : [1, 64, 384, 384]
    output : [1, 64, 192, 192]

  Block C  (block_C_slice_conv_pattern.onnx)
    Slice×2 → Conv2d×2(groups=1) → Clip×2 → Concat
    input  : [1, 64, 320, 576]
    output : [1, 64, 320, 576]

Each test:
  1. Compiles through Forge with HiFi3 / opt_level=2 / Float16_b
  2. Dumps TTIR and TTNN MLIR into SLICE_CONV_GROUPED_IR/<block>/
  3. Runs one inference pass
  4. Verifies output against ONNX Runtime reference (pcc >= 0.99)

IR dump directory:
    SLICE_CONV_GROUPED_IR/block_A/   — ttir_<name>.mlir  ttnn_<name>.mlir
    SLICE_CONV_GROUPED_IR/block_C/   — ttir_<name>.mlir  ttnn_<name>.mlir

Usage:
    pytest forge/test/models/onnx/vision/bev/test_bev_slice_conv_grouped_conv.py -s -v \\
        2>&1 | tee SLICE_CONV_GROUPED_IR/test.log
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import List

import onnx
import pytest
import torch

import forge
import forge._C
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SPLIT_DIR = _REPO_ROOT / "BEV_model" / "split_models"

_MODEL_BLOCK_A = _SPLIT_DIR / "block_A_cam0_slice_conv_pattern.onnx"
_MODEL_BLOCK_C = _SPLIT_DIR / "block_C_slice_conv_pattern.onnx"

_IR_ROOT = _REPO_ROOT / "SLICE_CONV_GROUPED_IR"

_PCC = 0.99

# ---------------------------------------------------------------------------
# Compiler config
# ---------------------------------------------------------------------------

def _make_cfg() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(False)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg


def _configure_device() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def _close_devices() -> None:
    """Close all open TT devices so the next test starts with a clean MetalContext."""
    try:
        from forge._C import runtime as forge_runtime
        system = forge_runtime.experimental.TTSystem.get_system()
        if system.is_initialized():
            system.close_devices()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def teardown_devices():
    """Close devices after every test to prevent MetalContext leaks across tests."""
    yield
    _close_devices()


# ---------------------------------------------------------------------------
# IR dump context manager
# ---------------------------------------------------------------------------

@contextmanager
def _dump_ir(output_dir: Path):
    """Set TTMLIR_DUMP_PIPELINE_IR + TTMLIR_DUMP_DIR for the enclosed block.

    TTIR is dumped before the MLIR pipeline; TTNN is dumped after.
    Files written: ttir_<module_name>.mlir  and  ttnn_<module_name>.mlir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prev_dump = os.environ.get("TTMLIR_DUMP_PIPELINE_IR")
    prev_dir  = os.environ.get("TTMLIR_DUMP_DIR")
    os.environ["TTMLIR_DUMP_PIPELINE_IR"] = "1"
    os.environ["TTMLIR_DUMP_DIR"] = str(output_dir)
    try:
        yield output_dir
    finally:
        if prev_dump is None:
            os.environ.pop("TTMLIR_DUMP_PIPELINE_IR", None)
        else:
            os.environ["TTMLIR_DUMP_PIPELINE_IR"] = prev_dump
        if prev_dir is None:
            os.environ.pop("TTMLIR_DUMP_DIR", None)
        else:
            os.environ["TTMLIR_DUMP_DIR"] = prev_dir


# ---------------------------------------------------------------------------
# Compile + run helper
# ---------------------------------------------------------------------------

def _compile_onnx(
    model_path: Path,
    sample_inputs: List[torch.Tensor],
    module_name: str,
    ir_output_dir: Path,
):
    """Compile ONNX through Forge, dumping TTIR/TTNN into ir_output_dir."""
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print(f"\n[{module_name}] Compiling {model_path.name}  "
          f"input={list(sample_inputs[0].shape)}")
    print(f"[{module_name}] IR output → {ir_output_dir}")

    # Force device re-init between tests so MetalContext is clean each time
    os.environ["TT_METAL_FORCE_REINIT"] = "1"

    # Pop profiler env vars that interfere with compile measurement
    _dispatch = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync     = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        with _dump_ir(ir_output_dir):
            compiled = forge.compile(
                onnx_model,
                sample_inputs=sample_inputs,
                compiler_cfg=_make_cfg(),
                module_name=module_name,
            )
    finally:
        if _dispatch is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
        if _sync is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync

    # Print IR files that were written
    mlir_files = sorted(ir_output_dir.glob("*.mlir"))
    if mlir_files:
        print(f"[{module_name}] MLIR files written:")
        for f in mlir_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}  ({size_kb:.1f} KB)")
    else:
        print(f"[{module_name}] WARNING: no .mlir files found in {ir_output_dir}")

    return compiled


def _validate(
    model_path: Path,
    module_name: str,
    compiled,
    inputs: List[torch.Tensor],
) -> None:
    """Compare Forge compiled output against ONNX Runtime reference (pcc=0.99)."""
    onnx_model = onnx.load(str(model_path))
    framework_model = forge.OnnxModule(module_name, onnx_model)
    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=_PCC))
    verify(inputs, framework_model, compiled, verify_cfg=verify_cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_slice_conv_block_A():
    """
    Block A cam0: AvgPool → Slice×2 → Conv2d×2(groups=1) → Clip×2 → Concat
    input=[1,64,384,384]  output=[1,64,192,192]

    The FuseSliceConvClipConcatToGroupedConv TVM pass should fuse the two
    Conv2d branches into a single Conv2d(groups=2, channels=64).
    """
    module_name = "slice_conv_block_A_cam0"
    ir_dir      = _IR_ROOT / "block_A"
    sample_inp  = torch.randn(1, 64, 384, 384)

    compiled = _compile_onnx(_MODEL_BLOCK_A, [sample_inp], module_name, ir_dir)

    _configure_device()
    output = compiled(sample_inp)
    print(f"\n[{module_name}] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_BLOCK_A, module_name, compiled, [sample_inp])
    print(f"[{module_name}] PASSED  pcc >= {_PCC}")


def test_slice_conv_block_C():
    """
    Block C: Slice×2 → Conv2d×2(groups=1) → Clip×2 → Concat
    input=[1,64,320,576]  output=[1,64,320,576]

    The FuseSliceConvClipConcatToGroupedConv TVM pass should fuse the two
    Conv2d branches into a single Conv2d(groups=2, channels=64).
    """
    module_name = "slice_conv_block_C"
    ir_dir      = _IR_ROOT / "block_C"
    sample_inp  = torch.randn(1, 64, 320, 576)

    compiled = _compile_onnx(_MODEL_BLOCK_C, [sample_inp], module_name, ir_dir)

    _configure_device()
    output = compiled(sample_inp)
    print(f"\n[{module_name}] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_BLOCK_C, module_name, compiled, [sample_inp])
    print(f"[{module_name}] PASSED  pcc >= {_PCC}")
