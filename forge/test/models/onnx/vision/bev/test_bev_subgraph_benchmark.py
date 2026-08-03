# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tracy profiling benchmark for BEV subgraph models.

Current target: block_A_single_cam0_to_conv_blocks3_clip.onnx
  - Extracted subgraph: model input → conv_blocks_3.0/_net/_net.2/Clip
  - Input  : input_0 (camera-0 feature map)
  - Output : [1, 96, 192, 192]
  - Nodes  : 23

Usage (Tracy profiling via tracy_run.sh):
    bash scripts/tracy_run.sh -o BEV_TRACY_CLIP_<timestamp>/ \\
        -n block_A_conv_blocks3_clip -p --mid-run-dump --no-device-trace \\
        -- pytest forge/test/models/onnx/vision/bev/test_bev_subgraph_benchmark.py \\
               ::test_tracy_conv_blocks3_clip_bf8 -s -v

    Set TTMLIR_DUMP_DIR to redirect IR dumps to a custom directory.
"""
from __future__ import annotations

import contextlib
import os
import signal
import statistics
import time
from typing import List, Tuple

import numpy as np
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

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
)
from test.models.onnx.vision.bev.model_utils.bev_split_utils import (
    BLOCK_DEFS,
    load_block_inputs,
    load_block_inputs_pool,
    split_models_dir,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 1

_TRACY_N_WARMUP = 1
_TRACY_N_TIMED = 1

_BLOCK_A_NAME = "block_A_deformed_backbone"
_MODULE_NAME = "block_A_conv_blocks3_clip"
_MODEL_FILE = "block_A_single_cam0_to_conv_blocks3_clip.onnx"
_NODE_COUNT = 23

_C0, _C1, _C2, _C3 = 44, 30, 14, 10
_TW = _C0 + _C1 + _C2 + _C3 + 5

# ---------------------------------------------------------------------------
# Compiler config
# ---------------------------------------------------------------------------


def _cfg_conv2d_search_extensions_bf8_no_trace() -> CompilerConfig:
    """opt_level_2 + HiFi3 + FP32 acc + extended search + BFP8 weights, NO reshard, trace OFF."""
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


def _configure_device() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def _enable_ir_dumps(module_name: str, dump_dir: str | None = None) -> None:
    if dump_dir is not None:
        os.environ["TTMLIR_DUMP_DIR"] = dump_dir
    out_dir = os.environ.get("TTMLIR_DUMP_DIR", os.getcwd())
    os.environ["TTMLIR_DUMP_PIPELINE_IR"] = "1"
    os.environ["TTMLIR_DUMP_MEMORY_OPS"] = "1"
    print(f"\n[IR dumps enabled] -> {out_dir}/")
    print(f"  Pipeline stages : {{01-ttir-passes, 03-lowering, ..., 13-final-after-dealloc}}.mlir")
    print(f"  TTIR (pre-lower): ttir_{module_name}.mlir")
    print(f"  TTNN (final)    : ttnn_{module_name}.mlir")
    print(f"  Memory checkpts : stderr (grep [MEMDUMP])")


def _compile_onnx(
    model_path,
    sample_inputs: List[torch.Tensor],
    compiler_cfg: CompilerConfig,
    enable_program_cache: bool,
    module_name: str,
):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    _dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync_prof = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    _mid_run_dump = os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        compiled = forge.compile(
            onnx_model,
            sample_inputs=sample_inputs,
            compiler_cfg=compiler_cfg,
            module_name=module_name,
        )
    finally:
        if _dispatch_prof is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
        if _sync_prof is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync_prof
        if _mid_run_dump is not None:
            os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = _mid_run_dump
    if enable_program_cache:
        _configure_device()
    return compiled


def _run_benchmark(
    compiled,
    frames_pool: List[List[torch.Tensor]],
    label: str = "",
    n_warmup: int = _TRACY_N_WARMUP,
    n_timed: int = _TRACY_N_TIMED,
) -> Tuple[float, float, float, float]:
    """Returns (mean_infer_ms, std_infer_ms, mean_total_ms, fps)."""
    for i in range(n_warmup):
        out = compiled(*frames_pool[i % len(frames_pool)])
        _ = [
            np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy()
            for o in (out if isinstance(out, (list, tuple)) else [out])
        ]

    prep_ms: List[float] = []
    infer_ms: List[float] = []
    collect_ms: List[float] = []

    _tracy_signpost(f"{label}-start")
    for i in range(n_timed):
        frame = frames_pool[(n_warmup + i) % len(frames_pool)]

        t0 = time.perf_counter()
        inputs = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in frame]
        prep_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        out = compiled(*inputs)
        infer_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        outputs = out if isinstance(out, (list, tuple)) else [out]
        _ = [
            np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy()
            for o in outputs
        ]
        collect_ms.append((time.perf_counter() - t0) * 1e3)

        print(f"\r  [{label}] {i + 1:4d}/{n_timed}  ({int((i+1)/n_timed*100):3d}%)", end="", flush=True)
    _tracy_signpost(f"{label}-end")

    print()
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_total = statistics.mean(total_ms)
    fps = BATCH_SIZE * 1000.0 / avg_total if avg_total > 0 else float("inf")
    mean_infer = statistics.mean(infer_ms)
    std_infer = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0
    return mean_infer, std_infer, avg_total, fps


def _print_result(
    label: str,
    node_count: int,
    cfg_name: str,
    program_cache: bool,
    mean_infer: float,
    std_infer: float,
    mean_total: float,
    fps: float,
) -> None:
    cache_str = "cache=ON" if program_cache else "cache=OFF"
    title = f"{label}  ({node_count} nodes)"
    row_label = f"{cfg_name}  [{cache_str}]"
    infer_str = f"{mean_infer:.2f} ± {std_infer:.2f} ms"
    sep = "-" * _TW
    print(f"\n  {title}")
    print(sep)
    print(
        f"| {'Config':<{_C0}}"
        f"| {'Inference (H2D+run+D2H)':<{_C1}}"
        f"| {'Total/frame':<{_C2}}"
        f"| {'FPS':<{_C3}}|"
    )
    print(sep)
    print(
        f"| {row_label:<{_C0}}"
        f"| {infer_str:<{_C1}}"
        f"| {mean_total:.2f} ms{'':<{_C2 - len(f'{mean_total:.2f} ms')}}"
        f"| {fps:<{_C3}.2f}|"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bev_assets():
    if not assets_available():
        paths = bev_paths()
        pytest.skip(
            f"BEV assets not found under {paths['root']}. "
            "Set BEV_ASSETS_DIR or populate model/input_samples/output_samples."
        )
    return bev_paths()


@pytest.fixture(scope="session")
def sequences(bev_assets):
    return list_sequences()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_tracy_conv_blocks3_clip_bf8(sequences):
    """
    Tracy profiling for block_A_single_cam0_to_conv_blocks3_clip.onnx.

    Config  : conv2d_search_extensions_bf8_no_trace
              (extended search, BFP8 conv2d weights, NO reshard, TT-trace OFF)
    Cache   : program cache enabled
    Validation: disabled (profiling mode)
    Warmup  : 1 pass
    Timed   : 1 pass (Tracy signpost brackets timed loop only)

    Model   : block_A_single_cam0_to_conv_blocks3_clip.onnx  (23 nodes)
              Subgraph extracted from block_A_single_cam0.onnx:
              input_0 → conv_blocks_3.0/_net/_net.2/Clip  [1, 96, 192, 192]

    IR dumps: TTMLIR_DUMP_DIR env var, or ./BEV_TRACY_CLIP_<block_short>/ by default.
    Ops perf: generated by tracy_run.sh -o <out_dir> wrapper.

    Launch via tracy_run.sh so TT_METAL_DEVICE_PROFILER=1 is set and the
    post-run CSV report is generated automatically.
    """
    model_path = split_models_dir() / _MODEL_FILE
    if not model_path.is_file():
        pytest.skip(
            f"Subgraph model not found: {model_path}\n"
            "Extract it with onnx.utils.extract_model() from block_A_single_cam0.onnx."
        )

    out_dir = os.environ.get(
        "TTMLIR_DUMP_DIR",
        os.path.join(os.getcwd(), "BEV_TRACY_CONV3_CLIP"),
    )
    os.makedirs(out_dir, exist_ok=True)
    _enable_ir_dumps(_MODULE_NAME, dump_dir=out_dir)

    compiler_cfg = _cfg_conv2d_search_extensions_bf8_no_trace()

    seq_id = sequences[0]
    all_inputs = load_block_inputs(_BLOCK_A_NAME, seq_id)
    sample_inputs = [all_inputs[0]]

    print(f"\n{'=' * 70}")
    print(f"  model        : {_MODEL_FILE}  ({_NODE_COUNT} nodes)")
    print(f"  module_name  : {_MODULE_NAME}")
    print(f"  compiler_cfg : conv2d_search_extensions_bf8_no_trace")
    print(f"  program_cache: enabled")
    print(f"  validation   : DISABLED (profiling mode)")
    print(f"  warmup/timed : {_TRACY_N_WARMUP}/{_TRACY_N_TIMED}")
    print(f"  IR dump dir  : {out_dir}")
    print(f"{'=' * 70}")

    compiled = _compile_onnx(
        model_path,
        sample_inputs,
        compiler_cfg,
        enable_program_cache=True,
        module_name=_MODULE_NAME,
    )

    run_label = f"{_MODULE_NAME} | conv2d_search_extensions_bf8_no_trace | cache=ON"
    pool_size = min(_TRACY_N_TIMED + _TRACY_N_WARMUP, max(len(sequences), 4))
    frames = [[f[0]] for f in load_block_inputs_pool(_BLOCK_A_NAME, sequences, pool_size)]

    mi, si, mt, fps = _run_benchmark(
        compiled, frames, label=run_label,
        n_warmup=_TRACY_N_WARMUP, n_timed=_TRACY_N_TIMED,
    )
    _print_result(
        f"block_A subgraph → conv_blocks_3 Clip",
        _NODE_COUNT,
        "conv2d_search_extensions_bf8_no_trace",
        True, mi, si, mt, fps,
    )
