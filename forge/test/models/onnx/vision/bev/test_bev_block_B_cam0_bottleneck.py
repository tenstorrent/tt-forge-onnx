# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Block B camera-0 BEV transform bottleneck test — extracted from block_B ONNX.

Subgraph extracted from block_B_deformed_bev_transform.onnx, camera 0 of 4,
using scripts/extract_block_B_cam0.py (onnx.utils.extract_model).

  Input feat (1, 192, 96, 96)     camera feature map
  Input LUT  (1, 128, 64, 8, 2)   BEV grid-sample lookup table
    │                   │
    │                   ├─ Gather × 8   decompose LUT into 8 subsampled views
    │                   │
    ├─ Conv(IC=192, OC=64, k=1×1)  reduce channels (ReLU6)
    │       → (1, 64, 96, 96)
    │
    ├─ GridSample × 8  (mode=nearest, align_corners=True)
    │       → (1, 128, 64, 512) each   8192 BEV positions × 64 ch
    │
    ├─ Concat(axis=1)              → (1, 512, 128, 64)
    │
    └─ reduce_conv(IC=512, OC=64, k=1×1)
            → (1, 64, 128, 64)

ONNX model (extracted from block_B, 20 nodes):
  BLOCKBD_BOTTLENECK/onnx/block_B_cam0_bev_transform.onnx

Expected bottlenecks (from block_BD_tracy_bottleneck_analysis.md, per camera):
  ReshapeViewDeviceOperation : LUT 5D→4D  (1,128,64,8,2)→(1,128,64,16)
      0.293 ms avg   FPU=0.000%   23.7% of per-cam FW
  GridSampleOperation        : 8192 scatter-gather DRAM reads
      0.230 ms avg   FPU=0.000%   18.6% of per-cam FW
  TilizeDeviceOperation      : ROW_MAJOR→TILE 8 MB conversion
      0.087 ms avg   FPU=0.000%   14.1% of per-cam FW
  InterleavedToShardedDevice : DRAM→L1 scatter to 43 cores
      0.051 ms avg   FPU=0.010%   12.5% of per-cam FW
  PermuteDeviceOperation     : NCHW→NHWC (1,192,96,96) DRAM TILE
      0.146 ms avg   FPU=9.79%    11.8% of per-cam FW

Usage
-----
# Benchmark:
pytest forge/test/models/onnx/vision/bev/test_bev_block_B_cam0_bottleneck.py \\
    --timeout=600 -vss

# With IR dump:
TTMLIR_DUMP_PIPELINE_IR=1 \\
pytest forge/test/models/onnx/vision/bev/test_bev_block_B_cam0_bottleneck.py \\
    --timeout=600 -vss

# Tracy profiling:
bash scripts/tracy_run.sh \\
    -o BLOCKBD_BOTTLENECK/tracy \\
    -n block_B_cam0 \\
    --no-device-trace -p --dispatch-cores --op-count 5000 \\
    --no-check-exit-code \\
    -- python3 -m pytest \\
       forge/test/models/onnx/vision/bev/test_bev_block_B_cam0_bottleneck.py \\
       --timeout=600 -vss
"""
from __future__ import annotations

import os
import time

import onnx
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig

_profiling = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
if _profiling:
    try:
        from tracy import signpost as _tracy_signpost
    except Exception:
        _tracy_signpost = lambda _: None
else:
    _tracy_signpost = lambda _: None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_ITERS = 1
TIMED_ITERS  = 1

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..", "..")
)
_MODEL_DIR = os.path.join(_REPO_ROOT, "BLOCKBD_BOTTLENECK", "onnx")
_IR_DIR    = os.path.join(_REPO_ROOT, "BLOCKBD_BOTTLENECK", "ir")

# Extracted from block_B_deformed_bev_transform.onnx, camera 0
# Script: scripts/extract_block_B_cam0.py
ONNX_PATH = os.path.join(_MODEL_DIR, "block_B_cam0_bev_transform.onnx")

# Two inputs: feature map + LUT
FEAT_SHAPE = (1, 192, 96,  96)
LUT_SHAPE  = (1, 128, 64,   8, 2)

# ---------------------------------------------------------------------------
# Compiler config — identical to full BEV block_B benchmark
# ---------------------------------------------------------------------------

def _get_cfg() -> CompilerConfig:
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


def _configure_device_program_cache() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def _make_inputs():
    return [
        torch.randn(*FEAT_SHAPE),
        torch.randn(*LUT_SHAPE),
    ]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _run(onnx_path: str, module_name: str, ir_out_dir: str):
    # Pop all profiler env vars during compile so OpModel device init does not
    # trigger the Tracy-enabled-build assertion in the runtime.
    # All three are restored before the timed section so inference IS captured.
    _profiler  = os.environ.pop("TT_METAL_DEVICE_PROFILER", None)
    _dispatch  = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync      = os.environ.pop("TT_METAL_PROFILER_SYNC", None)

    ir_dump = bool(os.environ.get("TTMLIR_DUMP_PIPELINE_IR"))
    if ir_dump:
        os.makedirs(ir_out_dir, exist_ok=True)
        os.environ["TTMLIR_DUMP_PIPELINE_IR_DIR"] = ir_out_dir

    try:
        model    = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        compiled = forge.compile(
            model,
            sample_inputs=_make_inputs(),
            compiler_cfg=_get_cfg(),
            module_name=module_name,
        )
        print(f"\n  [{module_name}] compiled OK")
    finally:
        if ir_dump:
            os.environ.pop("TTMLIR_DUMP_PIPELINE_IR_DIR", None)
        # NOTE: _dispatch and _sync are intentionally NOT restored here.
        # They will be restored after warmup so only the timed section is profiled.

    _configure_device_program_cache()

    # Warmup — profiler dispatch still disabled, not captured by Tracy
    print(f"  [{module_name}] warmup ({WARMUP_ITERS}) ...", flush=True)
    for _ in range(WARMUP_ITERS):
        compiled(*_make_inputs())

    # Restore all profiler env vars only before timed iterations
    if _profiler is not None:
        os.environ["TT_METAL_DEVICE_PROFILER"] = _profiler
    if _dispatch is not None:
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
    if _sync is not None:
        os.environ["TT_METAL_PROFILER_SYNC"] = _sync

    # Timed iterations — Tracy captures only this section
    print(f"  [{module_name}] timed  ({TIMED_ITERS}) ...", flush=True)
    _tracy_signpost(f"{module_name}-timed-section-start")
    latencies = []
    for i in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _tracy_signpost(f"{module_name}-iter{i}-start")
        out = compiled(*_make_inputs())
        _tracy_signpost(f"{module_name}-iter{i}-end")
        latencies.append((time.perf_counter() - t0) * 1e3)
        print(f"    iter {i+1}/{TIMED_ITERS}: {latencies[-1]:.2f} ms", flush=True)
    _tracy_signpost(f"{module_name}-timed-section-end")

    avg = sum(latencies) / len(latencies)
    feat_str = "x".join(str(d) for d in FEAT_SHAPE)
    lut_str  = "x".join(str(d) for d in LUT_SHAPE)
    print(f"\n  {'Module':<50} {'Feat shape':<20} {'LUT shape':<22} {'Avg ms':>10} {'FPS':>8}")
    print("  " + "-"*115)
    print(f"  {module_name:<50} {feat_str:<20} {lut_str:<22} {avg:>10.2f} {1000/avg:>8.2f}")
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_block_B_cam0_bev_transform():
    """
    Camera 0 subgraph from block_B_deformed_bev_transform.onnx — BEV transform.

    Exact 20-node subgraph extracted with onnx.utils.extract_model:
      Gather × 8                         decompose LUT into 8 grid views
      Conv(IC=192, OC=64, k=1×1, ReLU6)  reduce camera features
      Clip                               ReLU6 clamp
      GridSample × 8                     nearest-neighbour BEV sampling
      Concat(axis=1)                     → (1, 512, 128, 64)
      reduce_conv(IC=512, OC=64, k=1×1) → (1, 64, 128, 64)

    Inputs:
      feat (1, 192, 96, 96)    camera feature map
      LUT  (1, 128, 64, 8, 2)  BEV grid-sample lookup table

    Expected bottlenecks (from BLOCK_BD/block_BD_tracy_bottleneck_analysis.md):
      ReshapeViewDeviceOperation  0.293 ms  23.7%  LUT 5D→4D stride-change copy
      GridSampleOperation         0.230 ms  18.6%  8192 non-sequential DRAM reads
      TilizeDeviceOperation       0.087 ms  14.1%  ROW_MAJOR→TILE 8 MB conversion
      InterleavedToSharded        0.051 ms  12.5%  DRAM→L1 scatter 43 cores
      PermuteDeviceOperation      0.146 ms  11.8%  NCHW→NHWC (1,192,96,96) DRAM
    """
    os.makedirs(_IR_DIR, exist_ok=True)
    _run(
        onnx_path   = ONNX_PATH,
        module_name = "block_B_cam0_bev_transform",
        ir_out_dir  = _IR_DIR,
    )
