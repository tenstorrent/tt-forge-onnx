# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FPS benchmark for Block C slice-conv pattern: with vs. without
FuseSliceConvClipConcatToGroupedConv.

Model : BEV_model/split_models/block_C_slice_conv_pattern.onnx
Input : [1, 64, 320, 576]

Compiler flags
--------------
  opt_level=2 | HiFi3 | Float16_b | fp32_dest_acc
  enable_consteval | enable_trace | enable_memreconfig | enable_fusing

Runtime flags
-------------
  program_cache=True

Two parametrized cases:
  with_fusion    — FuseSliceConvClipConcatToGroupedConv active
                   → groups=2 conv2d replaces 2×slice+conv+clip+concat
  without_fusion — callback no-op, original 2-branch graph compiled as-is

Usage
-----
    pytest forge/test/models/onnx/vision/bev/test_bev_slice_conv_block_C_fps.py -s -v \\
        2>&1 | tee SLICE_CONV_GROUPED_IR/block_C_fps.log
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
import torch

import forge
import forge._C
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.tvm_calls.relay.op.forge_passes import FuseSliceConvClipConcatToGroupedConv

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT   = Path(__file__).resolve().parents[6]
_BLOCK_C_ONNX = _REPO_ROOT / "BEV_model" / "split_models" / "block_C_slice_conv_pattern.onnx"

INPUT_SHAPE = (1, 64, 320, 576)
WARMUP_ITERS = 3
TIMED_ITERS  = 10

# ---------------------------------------------------------------------------
# Compiler config
# ---------------------------------------------------------------------------

def _make_cfg(module_name: str) -> CompilerConfig:
    """
    Full production compiler config:
      opt_level=2, HiFi3, Float16_b, fp32_dest_acc
      enable_trace       — tt-metal trace for fast repeated inference
      enable_memreconfig — activation memory reconfiguration
      enable_fusing      — op fusing / double-buffering
    """
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(True)
        .set_enable_memreconfig(True)
        .set_enable_fusing(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg


def _configure_device() -> None:
    """Enable program cache on the TT device."""
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


# ---------------------------------------------------------------------------
# No-op stub used for the "without_fusion" case
# ---------------------------------------------------------------------------

class _DisabledFuseSliceConv(FuseSliceConvClipConcatToGroupedConv):
    """Drop-in stub that leaves the graph unchanged (callback is a no-op)."""
    def callback(self, pre, post, node_map):
        return post


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------

def _compile(module_name: str, use_fusion: bool) -> forge.CompiledModel:
    model = onnx.load(str(_BLOCK_C_ONNX))
    onnx.checker.check_model(model)
    sample_inp = torch.randn(*INPUT_SHAPE)

    # Pop profiler env vars that can interfere with compile
    _dispatch = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync     = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        if use_fusion:
            compiled = forge.compile(
                model,
                sample_inputs=[sample_inp],
                compiler_cfg=_make_cfg(module_name),
                module_name=module_name,
            )
        else:
            # Patch the callback class in forge_passes so run_forge_compile_passes
            # instantiates the no-op stub instead of the real fusion callback.
            target = "forge.tvm_calls.relay.op.forge_passes.FuseSliceConvClipConcatToGroupedConv"
            with patch(target, _DisabledFuseSliceConv):
                compiled = forge.compile(
                    model,
                    sample_inputs=[sample_inp],
                    compiler_cfg=_make_cfg(module_name),
                    module_name=module_name,
                )
    finally:
        if _dispatch is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
        if _sync is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync

    return compiled


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def _benchmark(compiled: forge.CompiledModel, label: str) -> float:
    """Warm up then time TIMED_ITERS inference calls. Returns average FPS."""
    print(f"\n  [{label}] warming up ({WARMUP_ITERS} iters) ...", flush=True)
    for _ in range(WARMUP_ITERS):
        compiled(torch.randn(*INPUT_SHAPE))

    print(f"  [{label}] benchmarking ({TIMED_ITERS} iters) ...", flush=True)
    latencies_ms: list[float] = []
    for i in range(TIMED_ITERS):
        t0 = time.perf_counter()
        compiled(torch.randn(*INPUT_SHAPE))
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3
        latencies_ms.append(ms)
        print(f"    iter {i+1:2d}/{TIMED_ITERS}: {ms:.2f} ms", flush=True)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    fps    = 1000.0 / avg_ms

    print(f"\n  ┌─ {label} ─────────────────────────────────────")
    print(f"  │  avg latency : {avg_ms:.2f} ms")
    print(f"  │  min latency : {min_ms:.2f} ms")
    print(f"  │  max latency : {max_ms:.2f} ms")
    print(f"  │  FPS         : {fps:.2f}")
    print(f"  └{'─' * (len(label) + 22)}")

    return fps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def teardown_devices():
    """Close all TT devices after each test to prevent MetalContext leaks."""
    yield
    try:
        from forge._C import runtime as forge_runtime
        system = forge_runtime.experimental.TTSystem.get_system()
        if system.is_initialized():
            system.close_devices()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "use_fusion",
    [False, True],
    ids=["without_fusion", "with_fusion"],
)
def test_block_C_slice_conv_fps(use_fusion: bool):
    """
    Compile and benchmark block_C_slice_conv_pattern.onnx with the full
    production compiler config (trace + program_cache + memreconfig + fusing).

    without_fusion: FuseSliceConvClipConcatToGroupedConv is a no-op.
                    Graph: 2× (strided_slice → conv2d → clip) → concat
    with_fusion   : Fusion active.
                    Graph: 1× conv2d(groups=2) → clip
    """
    tag        = "WITH fusion" if use_fusion else "WITHOUT fusion"
    label      = f"block_C [{tag}]"
    module_name = f"block_C_slice_conv_{'fused' if use_fusion else 'unfused'}"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  input : {list(INPUT_SHAPE)}")
    print(f"  flags : opt_level=2 | HiFi3 | Float16_b | fp32_dest_acc")
    print(f"          enable_trace | enable_memreconfig | enable_fusing")
    print(f"          program_cache=True")
    print(f"{'=' * 60}")

    os.environ["TT_METAL_FORCE_REINIT"] = "1"

    compiled = _compile(module_name, use_fusion=use_fusion)
    _configure_device()
    fps = _benchmark(compiled, label)

    print(f"\n  RESULT [{tag}]: {fps:.2f} FPS")
