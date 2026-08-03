# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal YUV → Concat → Conv2d bottleneck test — extracted directly from block_A ONNX.

Subgraph extracted from block_A_deformed_backbone.onnx, camera 0.
Covers the exact op chain from camera input through the first backbone conv2d:

  Input (1, 3, 1536, 1536)
    │
    ├─ Conv(1×1, IC=3, OC=3)         YUV adapter with real trained weights
    │     → (1, 3, 1536, 1536)
    │
    ├─ Slice [ch 0]   → (1, 1, 1536, 1536)   Y channel
    │     Reshape [1,1,384,4,384,4]
    │     Transpose {0,3,5,1,2,4}            pixel_unshuffle r=4
    │     Reshape [1,16,384,384]
    │
    ├─ Slice [ch 1:3] → (1, 2, 1536, 1536)   UV channels
    │     AveragePool(kernel=[2,1], stride=[2,2], count_include_pad=1)
    │     Reshape [1,2,384,2,384,2]
    │     Transpose {0,3,5,1,2,4}            pixel_unshuffle r=2
    │     Reshape [1,8,384,384]
    │
    ├─ Concat(axis=1) → (1, 24, 384, 384)
    │
    └─ Conv(1×1, IC=24, OC=64)       first backbone conv2d (real weights)
          → (1, 64, 384, 384)

ONNX model (extracted from actual block_A, 12 nodes):
  BEV_MODEL_LOGS/minimal_dram_bw_models/yuv_concat_conv_cam0_block_A_1536x1536.onnx

TTNN IR bottlenecks reproduced (from block_A_june25 Tracy report):
  ReshapeViewDeviceOperation #1: (1,1,1536,1536) DRAM TILE → [384,4,384,4]
      18.12ms / 12 calls = 1.51ms  OUTPUT_0_X_PAD=32[4] = 87.5% waste
  ReshapeViewDeviceOperation #2: (1,32,48,768) DRAM TILE → [2,384,2,384,2]
      13.73ms / 12 calls = 1.14ms  OUTPUT_0_X_PAD=32[2] = 93.8% waste

Usage
-----
# Benchmark:
pytest forge/test/models/onnx/vision/bev/test_bev_yuv_concat_bottleneck.py \\
    --timeout=300 -vss

# With IR dump:
TTMLIR_DUMP_PIPELINE_IR=1 pytest ... --timeout=300 -vss

# Tracy profiling:
bash scripts/tracy_run.sh \\
    -o BLOCK_A_AND_C/yuv_concat_minimal_tracy \\
    -n yuv_concat_bottleneck \\
    --no-device-trace -p --dispatch-cores --op-count 5000 \\
    --no-check-exit-code \\
    -- python3 -m pytest \\
       forge/test/models/onnx/vision/bev/test_bev_yuv_concat_bottleneck.py \\
       --timeout=300 -vss
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
_MODEL_DIR = os.path.join(_REPO_ROOT, "BEV_MODEL_LOGS", "minimal_dram_bw_models")
_IR_DIR    = os.path.join(_REPO_ROOT, "BLOCK_A_AND_C", "yuv_concat_minimal_ir")

# Extracted from block_A_deformed_backbone.onnx, camera 0
# Includes downstream Conv2d (IC=24→OC=64, kernel=1×1)
ONNX_PATH = os.path.join(_MODEL_DIR, "yuv_concat_conv_cam0_block_A_1536x1536.onnx")

# ---------------------------------------------------------------------------
# Compiler config — identical to full BEV block_A benchmark
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _run(onnx_path: str, input_shape: tuple, module_name: str, ir_out_dir: str):
    # Pop all profiler env vars during compile so OpModel device init does not
    # trigger the Tracy-enabled-build assertion in the runtime.
    # All three are restored before the timed section so inference IS captured.
    _profiler = os.environ.pop("TT_METAL_DEVICE_PROFILER", None)
    _dispatch = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync     = os.environ.pop("TT_METAL_PROFILER_SYNC", None)

    ir_dump = bool(os.environ.get("TTMLIR_DUMP_PIPELINE_IR"))
    if ir_dump:
        os.makedirs(ir_out_dir, exist_ok=True)
        os.environ["TTMLIR_DUMP_PIPELINE_IR_DIR"] = ir_out_dir

    try:
        model    = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        inp      = torch.randn(*input_shape)
        compiled = forge.compile(
            model,
            sample_inputs=[inp],
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
        compiled(torch.randn(*input_shape))

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
        out = compiled(torch.randn(*input_shape))
        _tracy_signpost(f"{module_name}-iter{i}-end")
        latencies.append((time.perf_counter() - t0) * 1e3)
        print(f"    iter {i+1}/{TIMED_ITERS}: {latencies[-1]:.2f} ms", flush=True)
    _tracy_signpost(f"{module_name}-timed-section-end")

    avg = sum(latencies) / len(latencies)
    shape_str = "x".join(str(d) for d in input_shape)
    print(f"\n  {'Module':<55} {'Shape':<22} {'Avg ms':>10} {'FPS':>8}")
    print("  " + "-"*100)
    print(f"  {module_name:<55} {shape_str:<22} {avg:>10.2f} {1000/avg:>8.2f}")
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_yuv_concat_conv_cam0_block_A():
    """
    Camera 0 subgraph from block_A_deformed_backbone.onnx — YUV→Concat→Conv2d.

    Exact 12-node subgraph (includes downstream backbone Conv2d):
      Conv(1×1, IC=3→OC=3, real weights)      YUV adapter
      Slice×2  →  Y(1ch) + UV(2ch)
      Reshape→Transpose{0,3,5,1,2,4}→Reshape  Y pixel_unshuffle r=4 → (1,16,384,384)
      AveragePool(k=[2,1],s=[2,2])             UV downsample
      Reshape→Transpose{0,3,5,1,2,4}→Reshape  UV pixel_unshuffle r=2 → (1,8,384,384)
      Concat(axis=1)                           → (1,24,384,384)
      Conv(1×1, IC=24→OC=64, real weights)    first backbone conv2d → (1,64,384,384)

    Bottlenecks reproduced (block_A_june25 Tracy):
      Y  6D reshape: (1,1,1536,1536) DRAM TILE → DRAM TILE 18432×1  1.51ms  87.5% waste
      UV 6D reshape: (1,32,48,768)   DRAM TILE → DRAM TILE 18432×1  1.14ms  93.8% waste
    """
    os.makedirs(_IR_DIR, exist_ok=True)
    _run(
        onnx_path    = ONNX_PATH,
        input_shape  = (1, 3, 1536, 1536),
        module_name  = "yuv_concat_conv_cam0_block_A",
        ir_out_dir   = _IR_DIR,
    )
