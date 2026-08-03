# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark for the YUV-420 input adapter conv2d from block_A and block_C,
with spatial packing enabled or disabled.

Four test cases:

    yuv420_block_A-packing_on  – [1,3,1536,1536] OC=3 1×1 conv, packing ON
    yuv420_block_A-packing_off – [1,3,1536,1536] OC=3 1×1 conv, packing OFF
    yuv420_block_C-packing_on  – [1,3,1280,2304] OC=3 1×1 conv, packing ON
    yuv420_block_C-packing_off – [1,3,1280,2304] OC=3 1×1 conv, packing OFF

    Source nodes:
      block_A_single_cam0.onnx:
        /model/_backbone/CameraDeformedCylinderEncoder/.../_yuv_420_input_adapter.0/Conv
      block_C_cylinder_backbone.onnx:
        /model/_backbone/CameraCylinderEncoder/.../_yuv_420_input_adapter.0/Conv

Models (weights embedded, activation is the lone graph input):
    BEV_model/split_models/yuv420_block_A_input_adapter_conv2d.onnx
    BEV_model/split_models/yuv420_block_C_input_adapter_conv2d.onnx

Examples
--------
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -s -v
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -k yuv420_block_A -s -v
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -k yuv420_block_C -s -v
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -k packing_on -s -v
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -k packing_off -s -v

    # Tracy profiling:
    TT_METAL_DEVICE_PROFILER=1 \\
    pytest forge/test/models/onnx/vision/bev/test_bev_conv2d_benchmark.py -s -v
"""
from __future__ import annotations

import gc
import os
import statistics
import time
from pathlib import Path
from typing import List, Tuple

import onnx
import pytest
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig

_tracy_signpost = lambda _: None
if os.environ.get("TT_METAL_DEVICE_PROFILER"):
    try:
        from tracy import signpost as _tracy_signpost
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _split_models_dir() -> Path:
    assets = os.environ.get("BEV_ASSETS_DIR", "BEV_model")
    return Path(assets) / "split_models"


_MODELS = {
    "yuv420_block_A": "yuv420_block_A_input_adapter_conv2d.onnx",
    "yuv420_block_C": "yuv420_block_C_input_adapter_conv2d.onnx",
}

# ---------------------------------------------------------------------------
# Compiler configs
# ---------------------------------------------------------------------------

def _cfg(spatial_packing: bool) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(False)
        .set_enable_conv2d_search_extensions(True)
        .set_experimental_conv2d_weight_dtype(forge._C.DataFormat.Bfp8_b)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRACY_N_WARMUP = 1
_TRACY_N_TIMED  = 1
_BENCH_N_WARMUP = 3
_BENCH_N_TIMED  = 10


def _configure_device() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def _compile(model_path: Path, activation: torch.Tensor,
             compiler_cfg: CompilerConfig, module_name: str):
    _d = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _s = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    _m = os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        compiled = forge.compile(onnx_model, sample_inputs=[activation],
                                 compiler_cfg=compiler_cfg, module_name=module_name)
    finally:
        if _d: os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _d
        if _s: os.environ["TT_METAL_PROFILER_SYNC"] = _s
        if _m: os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = _m
    _configure_device()
    return compiled


def _run_benchmark(compiled, activation: torch.Tensor, label: str,
                   n_warmup: int, n_timed: int) -> Tuple[float, float, float]:
    for _ in range(n_warmup):
        compiled(activation)

    infer_ms: List[float] = []
    _tracy_signpost(f"{label}-start")
    for i in range(n_timed):
        t0 = time.perf_counter()
        compiled(activation)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        print(f"\r  [{label}] {i+1}/{n_timed}", end="", flush=True)
    _tracy_signpost(f"{label}-end")
    print()

    mean_inf = statistics.mean(infer_ms)
    std_inf  = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0
    fps      = 1000.0 / mean_inf if mean_inf > 0 else float("inf")
    return mean_inf, std_inf, fps


def _print_result(block: str, spatial_packing: bool,
                  in_shape: list, mean_inf: float,
                  std_inf: float, fps: float) -> None:
    W = 72
    mode = "spatial-packing ON  (reshape → ttnn.linear → reshape)"
    if not spatial_packing:
        mode = "spatial-packing OFF (reshape → permute → reshape → ttnn.linear)"
    ic = in_shape[1] if len(in_shape) > 1 else 0
    print(f"\n  {block} — {mode}")
    if spatial_packing and ic >= 4:
        packed = [in_shape[0], ic * 4, in_shape[2] // 4, in_shape[3]]
        print(f"  input : {in_shape}  →  packed {packed}  (K=4)")
    else:
        print(f"  input : {in_shape}  (no packing)")
    print("-" * W)
    print(f"  Inference (H2D+run+D2H) : {mean_inf:.2f} ± {std_inf:.2f} ms")
    print(f"  Throughput              : {fps:.2f} fps")
    print("-" * W)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def split_models_dir():
    d = _split_models_dir()
    for fname in _MODELS.values():
        if not (d / fname).exists():
            pytest.skip(
                f"{fname} not found in {d}. "
                "Run: python forge/test/models/onnx/vision/bev/create_bev_conv2d_splits.py"
            )
    return d

# ---------------------------------------------------------------------------
# Tests (4 cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "block_short,spatial_packing",
    [
        ("yuv420_block_A", True),
        ("yuv420_block_A", False),
        ("yuv420_block_C", True),
        ("yuv420_block_C", False),
    ],
    ids=[
        "yuv420_block_A-packing_on",
        "yuv420_block_A-packing_off",
        "yuv420_block_C-packing_on",
        "yuv420_block_C-packing_off",
    ],
)
def test_bev_ic24_conv2d_search_extensions_bf8(
    block_short: str,
    spatial_packing: bool,
    split_models_dir: Path,
):
    """
    Compile and benchmark the IC24 1×1 conv2d with spatial packing on or off.

    Config  : conv2d_search_extensions_bf8_no_trace + enable_spatial_packing
    Input   : random fp32 activation matching the original conv2d input shape
    Cache   : program cache enabled
    Warmup  : 3 passes  (1 in Tracy profiling mode)
    Timed   : 10 passes (2 in Tracy profiling mode)
    """
    _profiling = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
    n_warmup = _TRACY_N_WARMUP if _profiling else _BENCH_N_WARMUP
    n_timed  = _TRACY_N_TIMED  if _profiling else _BENCH_N_TIMED

    packing_tag = "packing_on" if spatial_packing else "packing_off"
    model_path = split_models_dir / _MODELS[block_short]
    onnx_model = onnx.load(str(model_path))
    in_shape   = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    activation = torch.randn(*in_shape, dtype=torch.float32)

    out_dir = os.environ.get(
        "TTMLIR_DUMP_DIR",
        os.path.join(os.getcwd(), f"BEV_IC24_{block_short.upper()}_{packing_tag.upper()}"),
    )
    os.makedirs(out_dir, exist_ok=True)
    os.environ["TTMLIR_DUMP_PIPELINE_IR"] = "1"

    ic = in_shape[1] if len(in_shape) > 1 else 0
    mode_str = ("reshape → ttnn.linear → reshape"
                if spatial_packing else
                "reshape → permute → reshape → ttnn.linear")
    print(f"\n{'='*70}")
    print(f"  block          : {block_short}")
    print(f"  model          : {_MODELS[block_short]}")
    print(f"  input          : {in_shape}")
    if spatial_packing and ic >= 4:
        packed = [in_shape[0], ic * 4, in_shape[2] // 4, in_shape[3]]
        print(f"  packed (K=4)   : {packed}  (IC×4={ic*4} → ttnn.linear)")
    print(f"  spatial packing: {'ON' if spatial_packing else 'OFF'}")
    print(f"  pipeline       : {mode_str}")
    print(f"  profiling      : {'ON' if _profiling else 'OFF'}")
    print(f"  warmup/timed   : {n_warmup}/{n_timed}")
    print(f"  IR dump dir    : {out_dir}")
    print(f"{'='*70}")

    compiler_cfg = _cfg(spatial_packing)
    module_name  = f"{block_short}_{packing_tag}"
    compiled = _compile(model_path, activation, compiler_cfg, module_name=module_name)

    label = f"{block_short}-{packing_tag}"
    mi, si, fps = _run_benchmark(compiled, activation, label=label,
                                 n_warmup=n_warmup, n_timed=n_timed)
    _print_result(block_short, spatial_packing, in_shape, mi, si, fps)
