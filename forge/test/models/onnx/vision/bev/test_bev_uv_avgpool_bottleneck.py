# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal UV AveragePool DRAM-bandwidth bottleneck test.

Reproduces the exact AveragePool op extracted from block_A_deformed_backbone.onnx
and block_C_cylinder_backbone.onnx:

  ONNX AveragePool:
    kernel_shape = [2, 1]      <- exact attrs from BEV model (NOT [2,2])
    strides      = [2, 2]
    pads         = [0, 0, 0, 0]
    count_include_pad = 1
    ceil_mode    = 0

Block A: input (1,2,1536,1536) -> output (1,2,768,768)   (x4 cameras)
Block C: input (1,2,1280,2304) -> output (1,2,640,1152)   (x1 camera)

Expected TTNN IR lowering (matches block_A IR lines 2260-2274):
  to_layout(tile)
  permute{0,2,3,1}   (1,2,H,W) -> (1,H,W,2) TILE DRAM   K=2->32, 93.8% waste
  reshape             -> (1,1,H*W,2) TILE DRAM
  conv2d (depthwise)  groups=2, in=2, out=2, kernel=[2,1], stride=[2,2]
  reshape             -> (1,H/2,W/2,2) TILE
  permute{0,3,1,2}    -> (1,2,H/2,W/2) TILE DRAM

Usage
-----
# Benchmark only:
pytest forge/test/models/onnx/vision/bev/test_bev_uv_avgpool_bottleneck.py \\
    --timeout=300 -vss

# Dump TTNN IR:
TTMLIR_DUMP_PIPELINE_IR=1 pytest ... -k "block_C"

# Tracy profiling:
bash scripts/tracy_run.sh \\
    -o BLOCK_A_AND_C/uv_avgpool_tracy \\
    -n uv_avgpool_bottleneck \\
    --no-device-trace -p --dispatch-cores --op-count 2000 \\
    --no-check-exit-code \\
    -- python3 -m pytest \\
       forge/test/models/onnx/vision/bev/test_bev_uv_avgpool_bottleneck.py \\
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
_IR_DIR_A  = os.path.join(_REPO_ROOT, "BLOCK_A_AND_C", "uv_avgpool_block_A_ir")
_IR_DIR_C  = os.path.join(_REPO_ROOT, "BLOCK_A_AND_C", "uv_avgpool_block_C_ir")

BLOCK_A_ONNX = os.path.join(_MODEL_DIR, "uv_avgpool_block_A_1536x1536.onnx")
BLOCK_C_ONNX = os.path.join(_MODEL_DIR, "uv_avgpool_block_C_1280x2304.onnx")

# ---------------------------------------------------------------------------
# Compiler config — identical to full BEV benchmark
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
# Compile + IR dump + benchmark
# ---------------------------------------------------------------------------

def _run(onnx_path: str, input_shape: tuple, module_name: str, ir_out_dir: str):
    # Disable profiler dispatch during compilation.
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
        if _dispatch is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
        if _sync is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync
        if ir_dump:
            os.environ.pop("TTMLIR_DUMP_PIPELINE_IR_DIR", None)

    _configure_device_program_cache()

    print(f"  [{module_name}] warmup ({WARMUP_ITERS}) ...", flush=True)
    for _ in range(WARMUP_ITERS):
        compiled(torch.randn(*input_shape))

    print(f"  [{module_name}] timed  ({TIMED_ITERS}) ...", flush=True)
    latencies = []
    for i in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _tracy_signpost(f"{module_name}-start")
        out = compiled(torch.randn(*input_shape))
        _tracy_signpost(f"{module_name}-end")
        latencies.append((time.perf_counter() - t0) * 1e3)
        print(f"    iter {i+1}/{TIMED_ITERS}: {latencies[-1]:.2f} ms", flush=True)

    avg = sum(latencies) / len(latencies)
    print(f"\n  {'Module':<45} {'Shape':<22} {'Avg ms':>10} {'FPS':>8}")
    print("  " + "-"*90)
    shape_str = "x".join(str(d) for d in input_shape)
    print(f"  {module_name:<45} {shape_str:<22} {avg:>10.2f} {1000/avg:>8.2f}")
    return out

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_uv_avgpool_bottleneck_block_A():
    """
    Block A: AveragePool(kernel=[2,1], stride=[2,2]) on (1,2,1536,1536).

    Exact attrs extracted from:
      block_A_deformed_backbone.onnx node
      '.../_yuv_420_input_adapter.1/_avg_pool/AveragePool'
      kernel_shape=[2,1]  strides=[2,2]  pads=[0,0,0,0]
      count_include_pad=1  ceil_mode=0

    Expected TTNN IR (block_A_deformed_backbone.mlir lines 2260-2274):
      permute{0,2,3,1}: (1,2,1536,1536) -> (1,1536,1536,2) TILE DRAM
        layout: memref<73728x1x!ttcore.tile<32x32,bf16>, #dram>  (X_PAD=32[2])
      reshape: -> (1,1,2359296,2) TILE DRAM
      conv2d: groups=2 in=2 out=2 kernel=[2,1] stride=[2,2] input_h=1536 input_w=1536
        weight: tensor<1x1x4x2xbf16>
      reshape: -> (1,768,768,2)
      permute{0,3,1,2}: -> (1,2,768,768)

    Expected Tracy ops-perf:
      PermuteDeviceOperation  OUT X_PAD=32[2]  PM_BW=545058ns  FPU~8%  ~6.5ms
    """
    _run(BLOCK_A_ONNX, (1, 2, 1536, 1536), "uv_avgpool_block_A", _IR_DIR_A)


def test_uv_avgpool_bottleneck_block_C():
    """
    Block C: AveragePool(kernel=[2,1], stride=[2,2]) on (1,2,1280,2304).

    Exact attrs extracted from:
      block_C_cylinder_backbone.onnx node
      '.../_yuv_420_input_adapter.1/_avg_pool/AveragePool'
      kernel_shape=[2,1]  strides=[2,2]  pads=[0,0,0,0]
      count_include_pad=1  ceil_mode=0

    Expected TTNN IR (block_C_cylinder_backbone.mlir lines 834-845):
      permute{0,2,3,1}: (1,2,1280,2304) -> (1,1280,2304,2) TILE DRAM
        layout: memref<92160x1x!ttcore.tile<32x32,bf16>, #dram>  (X_PAD=32[2])
      reshape: -> (1,1,2949120,2) TILE DRAM
      conv2d: groups=2 in=2 out=2 kernel=[2,1] stride=[2,2] input_h=1280 input_w=2304
        weight: tensor<1x1x4x2xbf16>
      reshape: -> (1,640,1152,2)
      permute{0,3,1,2}: -> (1,2,640,1152)

    Expected Tracy ops-perf:
      PermuteDeviceOperation  OUT X_PAD=32[2]  PM_BW=681322ns  FPU~9.4%  ~7ms
    """
    _run(BLOCK_C_ONNX, (1, 2, 1280, 2304), "uv_avgpool_block_C", _IR_DIR_C)
