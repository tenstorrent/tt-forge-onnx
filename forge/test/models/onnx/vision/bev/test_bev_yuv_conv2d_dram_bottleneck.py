# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal YUV conv2d DRAM-bandwidth bottleneck tests for Block A and Block C.

Each test compiles and runs a single-node Conv2D ONNX model that reproduces
the exact in_channels=3, kernel=1x1 configuration from the BEV YUV adapter,
at the resolution of Block A (1x3x1536x1536) and Block C (1x3x1280x2304).

Purpose
-------
Confirm that the DRAM bandwidth bottleneck (permute + matmul, FPU ~0%,
PM_REQ_BW ~277 GB/s) is reproducible in isolation from the full model,
using the same compiler config as the production BEV runs.

Usage
-----
# Both tests with Tracy profiler:
bash scripts/tracy_run.sh \\
    -o BEV_BLOCKBC_CONV/tracy \\
    -n yuv_conv2d_dram_bottleneck \\
    --no-device-trace -p --dispatch-cores --op-count 2000 \\
    -- python3 -m pytest \\
       forge/test/models/onnx/vision/bev/test_bev_yuv_conv2d_dram_bottleneck.py \\
       --timeout=300 -vss

# Single test:
pytest forge/test/models/onnx/vision/bev/test_bev_yuv_conv2d_dram_bottleneck.py \\
    -k "block_A" --timeout=300 -vss
"""
from __future__ import annotations

import os
import time

import onnx
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig

try:
    from tracy import signpost as _tracy_signpost
except ImportError:
    _tracy_signpost = lambda _: None

# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

WARMUP_ITERS  = 1   # warm-up inference calls (JIT cache, kernel launch)
TIMED_ITERS   = 1   # timed inference calls for latency measurement (1 = single Tracy-profiled iter)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..", "..")
)
_MODEL_DIR = os.path.join(_REPO_ROOT, "BEV_MODEL_LOGS", "minimal_dram_bw_models")

BLOCK_A_ONNX = os.path.join(_MODEL_DIR, "yuv_conv_block_A_1536x1536.onnx")
BLOCK_C_ONNX = os.path.join(_MODEL_DIR, "yuv_conv_block_C_1280x2304.onnx")

# ---------------------------------------------------------------------------
# Compiler config — matches production BEV benchmark
# ---------------------------------------------------------------------------

def get_cfg() -> CompilerConfig:
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

# ---------------------------------------------------------------------------
# Program cache helper
# ---------------------------------------------------------------------------

def _configure_device_program_cache() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)

# ---------------------------------------------------------------------------
# Compile + benchmark helper
# ---------------------------------------------------------------------------

def _compile_and_benchmark(onnx_path: str, input_shape: tuple, module_name: str):
    """Compile the model, run warmup, then timed benchmark iterations.

    Prints a table row:
      Config | Input shape | Warmup | Avg latency (ms) | FPS
    """
    _dispatch = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync     = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        model    = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        inp      = torch.randn(*input_shape)
        compiled = forge.compile(
            model,
            sample_inputs=[inp],
            compiler_cfg=get_cfg(),
            module_name=module_name,
        )
        # file_path = f"{module_name}.cpp"
        # compiled.export_to_cpp(file_path)
    finally:
        if _dispatch is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
        if _sync is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync

    _configure_device_program_cache()

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n  [{module_name}] warming up ({WARMUP_ITERS} iter) ...", flush=True)
    for _ in range(WARMUP_ITERS):
        compiled(torch.randn(*input_shape))

    # ── Timed benchmark ───────────────────────────────────────────────────────
    print(f"  [{module_name}] benchmarking ({TIMED_ITERS} iter) ...", flush=True)
    latencies_ms = []
    for i in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _tracy_signpost(f"{module_name}-start")
        out = compiled(torch.randn(*input_shape))
        _tracy_signpost(f"{module_name}-end")
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)
        print(f"    iter {i+1:2d}/{TIMED_ITERS}: {latencies_ms[-1]:.2f} ms", flush=True)

    avg_ms  = sum(latencies_ms) / len(latencies_ms)
    min_ms  = min(latencies_ms)
    max_ms  = max(latencies_ms)
    fps     = 1000.0 / avg_ms

    shape_str = "x".join(str(d) for d in input_shape)
    print(
        f"\n  {'Config':<45} {'Shape':<20} {'Warmup':>8} {'Avg (ms)':>10}"
        f" {'Min (ms)':>10} {'Max (ms)':>10} {'FPS':>8}"
    )
    print("  " + "-" * 115)
    print(
        f"  {module_name:<45} {shape_str:<20} {WARMUP_ITERS:>8d}"
        f" {avg_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f} {fps:>8.2f}"
    )

    return out

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_yuv_conv2d_dram_bottleneck_block_A():
    """
    Block A YUV conv2d  in=(1,3,1536,1536)  kernel=1x1  out=(1,3,1536,1536)
    Spatial packing: C*K=96 fills tile rows → 17.7 MB permute (vs 151 MB baseline).
    """
    _compile_and_benchmark(
        onnx_path   = BLOCK_A_ONNX,
        input_shape = (1, 3, 1536, 1536),
        module_name = "yuv_conv2d_block_A_minimal",
    )


def test_yuv_conv2d_dram_bottleneck_block_C():
    """
    Block C YUV conv2d  in=(1,3,1280,2304)  kernel=1x1  out=(1,3,1280,2304)
    Spatial packing: C*K=96 fills tile rows → 17.7 MB permute (vs 188.7 MB baseline).
    """
    _compile_and_benchmark(
        onnx_path   = BLOCK_C_ONNX,
        input_shape = (1, 3, 1280, 2304),
        module_name = "yuv_conv2d_block_C_minimal",
    )
