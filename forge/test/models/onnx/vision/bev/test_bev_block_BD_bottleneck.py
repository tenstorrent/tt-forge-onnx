# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tracy profiling test for BEV Block B (deformed BEV transform) and
Block D (cylinder BEV transform) bottleneck analysis.

Block B — block_B_deformed_bev_transform.onnx
  Inputs:
    4 × (1, 192, 96, 96)   camera feature maps (cam0–cam3)
    4 × (1, 128, 64, 8, 2) grid sample LUTs    (lut0–lut3)
  Output:
    4 × (1, 64, 128, 64)   BEV feature maps

Block D — block_D_cylinder_bev_transform.onnx
  Inputs:
    1 × (1, 192, 80, 144)  camera feature map
    1 × (1, 128, 64, 8, 2) grid sample LUT
  Output:
    1 × (1, 64, 128, 64)   BEV feature map

Compiler config: opt_level_2 + HiFi3 + fp32_acc + Float16_b
  (matches the best-performing config from block A/C benchmarks)

Usage
-----
# IR dump only (no profiler):
TTMLIR_DUMP_PIPELINE_IR=1 \
TTMLIR_DUMP_PIPELINE_IR_DIR=BLOCK_BD/block_B_ir \
pytest forge/test/models/onnx/vision/bev/test_bev_block_BD_bottleneck.py \
    -k "block_B" --timeout=600 -vss

# Tracy profiling:
bash scripts/tracy_run.sh \\
    -o BLOCK_BD/block_B_tracy \\
    -n block_B_bottleneck \\
    --no-device-trace -p --dispatch-cores --op-count 10000 \\
    --no-check-exit-code \\
    -- python3 -m pytest \\
       forge/test/models/onnx/vision/bev/test_bev_block_BD_bottleneck.py \\
       -k "block_B" --timeout=600 -vss
"""
from __future__ import annotations

import os
import time

import onnx
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig

# ---------------------------------------------------------------------------
# Tracy signpost
# ---------------------------------------------------------------------------

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
TIMED_ITERS  = 3

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..", "..")
)
_MODELS_DIR = os.path.join(_REPO_ROOT, "BEV_model", "split_models")

BLOCK_B_ONNX = os.path.join(_MODELS_DIR, "block_B_deformed_bev_transform.onnx")
BLOCK_D_ONNX = os.path.join(_MODELS_DIR, "block_D_cylinder_bev_transform.onnx")

# ---------------------------------------------------------------------------
# Compiler config  (matches best BEV benchmark config)
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
# Synthetic inputs
# ---------------------------------------------------------------------------

def _block_B_inputs():
    """4 camera feature maps + 4 LUT tensors."""
    feat = [torch.randn(1, 192, 96, 96,  dtype=torch.float32) for _ in range(4)]
    luts = [torch.randn(1, 128, 64, 8, 2, dtype=torch.float32) for _ in range(4)]
    return feat + luts   # 8 tensors


def _block_D_inputs():
    """1 camera feature map + 1 LUT tensor."""
    feat = torch.randn(1, 192, 80, 144, dtype=torch.float32)
    lut  = torch.randn(1, 128, 64, 8, 2, dtype=torch.float32)
    return [feat, lut]   # 2 tensors

# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _run(onnx_path: str, make_inputs, module_name: str, ir_out_dir: str):
    # Pop profiler env vars — kept disabled during compile AND warmup.
    # Only the timed section gets captured by Tracy.
    _dispatch = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync     = os.environ.pop("TT_METAL_PROFILER_SYNC", None)

    ir_dump = bool(os.environ.get("TTMLIR_DUMP_PIPELINE_IR"))
    if ir_dump:
        os.makedirs(ir_out_dir, exist_ok=True)
        os.environ["TTMLIR_DUMP_PIPELINE_IR_DIR"] = ir_out_dir

    try:
        model    = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        sample   = make_inputs()
        compiled = forge.compile(
            model,
            sample_inputs=sample,
            compiler_cfg=_get_cfg(),
            module_name=module_name,
        )
        print(f"\n  [{module_name}] compiled OK")
    finally:
        if ir_dump:
            os.environ.pop("TTMLIR_DUMP_PIPELINE_IR_DIR", None)
        # NOTE: _dispatch and _sync intentionally NOT restored here.
        # They are restored only before the timed section.

    _configure_device_program_cache()

    # Warmup — profiler dispatch still disabled, not captured by Tracy
    print(f"  [{module_name}] warmup ({WARMUP_ITERS} iter) ...", flush=True)
    for _ in range(WARMUP_ITERS):
        compiled(*make_inputs())

    # Restore profiler dispatch env vars so only timed section is captured
    if _dispatch is not None:
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch
    if _sync is not None:
        os.environ["TT_METAL_PROFILER_SYNC"] = _sync

    # Timed iterations — Tracy captures only this section
    print(f"  [{module_name}] timed  ({TIMED_ITERS} iter) ...", flush=True)
    _tracy_signpost(f"{module_name}-timed-section-start")
    latencies = []
    for i in range(TIMED_ITERS):
        inputs = make_inputs()
        t0 = time.perf_counter()
        _tracy_signpost(f"{module_name}-iter{i}-start")
        out = compiled(*inputs)
        _tracy_signpost(f"{module_name}-iter{i}-end")
        latencies.append((time.perf_counter() - t0) * 1e3)
        print(f"    iter {i+1}/{TIMED_ITERS}: {latencies[-1]:.2f} ms", flush=True)
    _tracy_signpost(f"{module_name}-timed-section-end")

    avg = sum(latencies) / len(latencies)
    print(f"\n  {'Module':<55} {'Avg ms':>10} {'FPS':>8}")
    print("  " + "-"*80)
    print(f"  {module_name:<55} {avg:>10.2f} {1000/avg:>8.2f}")
    return out

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_block_B_deformed_bev_transform():
    """
    Block B — deformed BEV transform (4 cameras).

    ONNX: block_B_deformed_bev_transform.onnx  (80 nodes)
    Inputs:
      4 × (1, 192, 96, 96)    camera feature maps  [cam0–cam3]
      4 × (1, 128, 64, 8, 2)  grid-sample LUTs     [lut0–lut3]
    Output:
      4 × (1, 64, 128, 64)    BEV feature maps

    Key ops expected: GridSample, Gather/Slice, Conv2d (reduce_conv),
    Reshape, Permute chains.
    """
    ir_out_dir = os.path.join(_REPO_ROOT, "BLOCK_BD", "block_B_ir")
    _run(
        onnx_path  = BLOCK_B_ONNX,
        make_inputs = _block_B_inputs,
        module_name = "block_B_deformed_bev_transform",
        ir_out_dir  = ir_out_dir,
    )


def test_block_D_cylinder_bev_transform():
    """
    Block D — cylinder BEV transform (1 camera).

    ONNX: block_D_cylinder_bev_transform.onnx  (20 nodes)
    Inputs:
      1 × (1, 192, 80, 144)   camera feature map
      1 × (1, 128, 64, 8, 2)  grid-sample LUT
    Output:
      1 × (1, 64, 128, 64)    BEV feature map

    Key ops expected: GridSample, Gather/Slice, Conv2d (reduce_conv),
    Reshape, Permute chains.
    """
    ir_out_dir = os.path.join(_REPO_ROOT, "BLOCK_BD", "block_D_ir")
    _run(
        onnx_path  = BLOCK_D_ONNX,
        make_inputs = _block_D_inputs,
        module_name = "block_D_cylinder_bev_transform",
        ir_out_dir  = ir_out_dir,
    )
