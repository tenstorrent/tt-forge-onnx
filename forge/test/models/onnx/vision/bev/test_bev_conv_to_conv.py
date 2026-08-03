# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import statistics
import time

import numpy as np
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

try:
    from tracy import signpost as _tracy_signpost
except ImportError:
    _tracy_signpost = lambda _: None  # noqa: E731

from pathlib import Path

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
)

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SPLIT_DIR = _REPO_ROOT / "BEV_model/split_models"

BATCH_SIZE = 1
N_WARMUP = 0
N_TIMED = 1
PCC = 0.99

# input_name: the .npy filename (without extension) in the sequence directory
# that provides the real camera input for this model.
_MODELS = {
    "block_A_single_cam": {
        "path":        _SPLIT_DIR / "block_A_single_cam_conv_to_conv.onnx",
        "module_name": "block_A_single_cam_conv_to_conv",
        "input_name":  "input_0",   # camera 0, shape (1,3,1536,1536)
        "label":       "Block A — single-cam conv→conv (12 nodes)",
    },
    "block_C": {
        "path":        _SPLIT_DIR / "block_C_conv_to_conv.onnx",
        "module_name": "block_C_conv_to_conv",
        "input_name":  "input_4",   # cylinder camera, shape (1,3,1280,2304)
        "label":       "Block C — conv→conv (12 nodes)",
    },
}


def _load_sample_input(input_name: str) -> torch.Tensor:
    """Load the first available sequence's .npy file for *input_name*."""
    seq_id = list_sequences()[0]
    npy_path = bev_paths()["input_samples"] / seq_id / f"{input_name}.npy"
    return torch.from_numpy(np.load(str(npy_path)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flush_device_profiler() -> None:
    """Flush on-device DRAM profiler buffer to host after timed inference."""
    try:
        import ttnn
        device = ttnn.GetDevice(0)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def _fmt(vals: list[float]) -> str:
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:8.2f} ± {std:6.2f} ms"


def _print_results_table(
    n_timed: int,
    n_warmup: int,
    prep_ms: list[float],
    infer_ms: list[float],
    collect_ms: list[float],
    batch_size: int = 1,
) -> tuple[float, float, float, float]:
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_total_ms = statistics.mean(total_ms)
    samples_per_sec = batch_size * 1000.0 / avg_total_ms if avg_total_ms > 0 else float("inf")

    mean_infer_ms = statistics.mean(infer_ms)
    std_infer_ms = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0

    print()
    print("=" * 58)
    print(f"  Benchmark results  ({n_timed} iters, {n_warmup} warmup, batch={batch_size})")
    print("=" * 58)
    print(f"  Data prep      (assemble input tensor)    :  {_fmt(prep_ms)}")
    print(f"  Inference      (H2D + run + D2H)          :  {_fmt(infer_ms)}")
    print(f"  Output collect (numpy realise)            :  {_fmt(collect_ms)}")
    print("-" * 58)
    print(f"  Total per frame                           :  {_fmt(total_ms)}")
    print(f"  Average samples/sec                       :  {samples_per_sec:8.2f}")
    print("=" * 58)

    return mean_infer_ms, std_infer_ms, avg_total_ms, samples_per_sec


def _run_benchmark(
    compiled,
    input_tensor: torch.Tensor,
    n_warmup: int,
    n_timed: int,
    label: str,
    batch_size: int = 1,
) -> tuple[float, float, float, float]:
    if n_warmup > 0:
        print(f"[benchmark] Warmup ({n_warmup} iters) …")
        for _ in range(n_warmup):
            out = compiled(input_tensor)
            _ = [
                np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy()
                for o in (out if isinstance(out, (list, tuple)) else [out])
            ]
        print("[benchmark] Warmup done.")

    print(f"[benchmark] Timed run ({n_timed} iters) …")
    prep_ms: list[float] = []
    infer_ms: list[float] = []
    collect_ms: list[float] = []

    for i in range(n_timed):
        t0 = time.perf_counter()
        inp = input_tensor if isinstance(input_tensor, torch.Tensor) else torch.from_numpy(input_tensor)
        prep_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        _tracy_signpost(f"{label}-start")
        out = compiled(inp)
        _tracy_signpost(f"{label}-end")
        _flush_device_profiler()
        infer_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        outputs = out if isinstance(out, (list, tuple)) else [out]
        _ = [
            np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy()
            for o in outputs
        ]
        collect_ms.append((time.perf_counter() - t0) * 1e3)

        pct = int((i + 1) / n_timed * 100)
        print(f"\r[benchmark] {i + 1:4d}/{n_timed}  ({pct:3d}%)", end="", flush=True)

    print()
    return _print_results_table(n_timed, n_warmup, prep_ms, infer_ms, collect_ms, batch_size=batch_size)


def _get_compiler_cfg(trace: bool) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(trace)
    )
    compiler_cfg = CompilerConfig(mlir_config=mlir_config)
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.enable_optimization_passes = True
    return compiler_cfg


def _configure_device_settings(enable_program_cache: bool) -> None:
    from forge._C import runtime as forge_runtime
    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = enable_program_cache
    forge_runtime.experimental.configure_devices(device_settings)


def _compile_model(
    onnx_model: onnx.ModelProto,
    sample_inputs: list[torch.Tensor],
    module_name: str,
    trace: bool,
    enable_program_cache: bool,
):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    _dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync_prof = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        compiled = forge.compile(
            onnx_model,
            sample_inputs=sample_inputs,
            compiler_cfg=_get_compiler_cfg(trace=trace),
            module_name=module_name,
        )
    finally:
        if _dispatch_prof is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
        if _sync_prof is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync_prof
    _configure_device_settings(enable_program_cache)
    return compiled


_COL_CFG   = 44
_COL_INFER = 36
_COL_TOTAL = 18
_COL_FPS   = 12
_TABLE_WIDTH = _COL_CFG + _COL_INFER + _COL_TOTAL + _COL_FPS + 5


def _print_table(
    case_label: str,
    rows: list[tuple[str, float, float, float, float]],
) -> None:
    sep = "-" * _TABLE_WIDTH
    print()
    print(f"  {case_label}")
    print(sep)
    header = (
        f"| {'Configuration':<{_COL_CFG}}"
        f"| {'Inference Time (H2D + run + D2H, per Frame)':<{_COL_INFER}}"
        f"| {'Total per Frame':<{_COL_TOTAL}}"
        f"| {'Samples/sec':<{_COL_FPS}}|"
    )
    print(header)
    print(sep)
    for cfg_label, mean_infer, std_infer, mean_total, samples_per_sec in rows:
        infer_str = f"{mean_infer:.2f} ± {std_infer:.2f} ms"
        total_str = f"{mean_total:.2f} ms"
        fps_str   = f"{samples_per_sec:.2f}"
        row = (
            f"| {cfg_label:<{_COL_CFG}}"
            f"| {infer_str:<{_COL_INFER}}"
            f"| {total_str:<{_COL_TOTAL}}"
            f"| {fps_str:<{_COL_FPS}}|"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", list(_MODELS.keys()))
@pytest.mark.parametrize(
    "program_cache",
    [True, False],
    ids=["enable_program_cache", "disable_program_cache"],
)
@pytest.mark.parametrize(
    "mlir_trace",
    [True, False],
    ids=["trace_enabled", "trace_disabled"],
)
def test_conv_to_conv_benchmark(model_key: str, program_cache: bool, mlir_trace: bool) -> None:
    spec = _MODELS[model_key]

    if not spec["path"].exists():
        pytest.skip(f"Model not found: {spec['path']}")
    if not assets_available():
        pytest.skip(f"BEV assets not found under {bev_paths()['root']}")

    cache_str = "enable_program_cache" if program_cache else "disable_program_cache"
    trace_str = "trace_enabled" if mlir_trace else "trace_disabled"
    cfg_label = f"{model_key}  {cache_str}  {trace_str}"

    sample_input = _load_sample_input(spec["input_name"])

    print(f"\n[conv_to_conv] model         : {spec['label']}")
    print(f"[conv_to_conv] input_shape   : {tuple(sample_input.shape)}")
    print(f"[conv_to_conv] program_cache : {cache_str}")
    print(f"[conv_to_conv] mlir_trace    : {trace_str}")
    print(f"[conv_to_conv] warmup/timed  : {N_WARMUP}/{N_TIMED}")

    onnx_model = onnx.load(str(spec["path"]))
    onnx.checker.check_model(onnx_model)
    print(f"[conv_to_conv] {len(onnx_model.graph.node)} nodes loaded from {spec['path'].name}")

    compiled = _compile_model(
        onnx_model,
        sample_inputs=[sample_input],
        module_name=spec["module_name"],
        trace=mlir_trace,
        enable_program_cache=program_cache,
    )

    print(f"\n[conv_to_conv] Running verification (pcc >= {PCC}) …")
    fw_model = forge.OnnxModule(spec["module_name"], onnx_model)
    verify(
        [sample_input],
        fw_model,
        compiled,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=PCC)),
    )
    print(f"[conv_to_conv] Verification PASSED (pcc >= {PCC})")

    mean_infer, std_infer, mean_total, samples_per_sec = _run_benchmark(
        compiled,
        sample_input,
        n_warmup=N_WARMUP,
        n_timed=N_TIMED,
        label=cfg_label,
        batch_size=BATCH_SIZE,
    )

    _print_table(
        f"{spec['label']}  (warmup={N_WARMUP}, timed={N_TIMED})",
        [(cfg_label, mean_infer, std_infer, mean_total, samples_per_sec)],
    )
