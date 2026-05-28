# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import statistics
import time

import numpy as np
import onnx
import pytest
import torch
from loguru import logger

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.compare import calculate_atol, calculate_pcc, compare_with_golden

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
    load_ground_truth_outputs,
    load_inputs,
)

BATCH_SIZE = 1
N_WARMUP = 3
N_TIMED = 10
PCC = 0.99


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
    print(f"  Data prep      (assemble input list)      :  {_fmt(prep_ms)}")
    print(f"  Inference      (H2D + run + D2H)          :  {_fmt(infer_ms)}")
    print(f"  Output collect (numpy realise)            :  {_fmt(collect_ms)}")
    print("-" * 58)
    print(f"  Total per frame                           :  {_fmt(total_ms)}")
    print(f"  Average samples/sec                       :  {samples_per_sec:8.2f}")
    print("=" * 58)

    return mean_infer_ms, std_infer_ms, avg_total_ms, samples_per_sec


def _run_benchmark(
    compiled,
    frames_pool: list[list[torch.Tensor]],
    n_warmup: int,
    n_timed: int,
    batch_size: int = 1,
) -> tuple[float, float, float, float]:
    if not frames_pool:
        raise ValueError("frames_pool must not be empty")

    if n_warmup > 0:
        print(f"[benchmark] Warmup ({n_warmup} frames) …")
        for i in range(n_warmup):
            inputs = frames_pool[i % len(frames_pool)]
            out = compiled(*inputs)
            _ = [
                np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy()
                for o in (out if isinstance(out, (list, tuple)) else [out])
            ]
        print("[benchmark] Warmup done.")

    print(f"[benchmark] Timed run ({n_timed} frames) …")
    prep_ms: list[float] = []
    infer_ms: list[float] = []
    collect_ms: list[float] = []

    for i in range(n_timed):
        frame_inputs = frames_pool[(n_warmup + i) % len(frames_pool)]

        t0 = time.perf_counter()
        inputs = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in frame_inputs]
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

        pct = int((i + 1) / n_timed * 100)
        print(f"\r[benchmark] {i + 1:4d}/{n_timed}  ({pct:3d}%)", end="", flush=True)

    print()
    return _print_results_table(n_timed, n_warmup, prep_ms, infer_ms, collect_ms, batch_size=batch_size)


def _get_compiler_cfg() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(True)
        .set_enable_ttnn_perf_metrics(True)
        .set_enable_ttnn_perf_metrics_verbose(True) 
        .set_ttnn_perf_metrics_output_file("bev_benchmark_perf_metrics.json")
    )
    compiler_cfg = CompilerConfig(
        mlir_config=mlir_config,
    )
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.enable_optimization_passes = True
    return compiler_cfg


def _configure_device_settings() -> None:
    from forge._C import runtime as forge_runtime

    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = True
    forge_runtime.experimental.configure_devices(device_settings)


def _compile_model(onnx_model: onnx.ModelProto, sample_inputs: list[torch.Tensor]):
    import os

    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    compiled = forge.compile(
        onnx_model,
        sample_inputs=sample_inputs,
        compiler_cfg=_get_compiler_cfg(),
        module_name="bev_onnx_full_model_new",
    )
    _configure_device_settings()
    return compiled


_COL_CFG = 40
_COL_INFER = 36
_COL_TOTAL = 18
_COL_FPS = 12
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
        fps_str = f"{samples_per_sec:.2f}"
        row = (
            f"| {cfg_label:<{_COL_CFG}}"
            f"| {infer_str:<{_COL_INFER}}"
            f"| {total_str:<{_COL_TOTAL}}"
            f"| {fps_str:<{_COL_FPS}}|"
        )
        print(row)
    print(sep)


def _validate_against_ground_truth(
    compiled, onnx_model: onnx.ModelProto, input_tensors: list[torch.Tensor], seq_id: str
) -> None:
    print("\n[bev_benchmark] Running ground-truth validation …")

    output_names = [o.name for o in onnx_model.graph.output]
    forge_outputs = compiled(*input_tensors)
    if not isinstance(forge_outputs, (list, tuple)):
        forge_outputs = [forge_outputs]

    gt = load_ground_truth_outputs(seq_id)

    failures = []
    for name, out in zip(output_names, forge_outputs):
        golden = torch.from_numpy(gt[name])
        calculated = out.detach().cpu() if hasattr(out, "detach") else torch.tensor(out)

        if golden.dtype != calculated.dtype and calculated.dtype == torch.bfloat16:
            calculated = calculated.to(golden.dtype)

        pcc_value = calculate_pcc(golden, calculated) if golden.numel() > 1 else None
        atol_value = calculate_atol(golden, calculated)
        passed = compare_with_golden(golden=golden, calculated=calculated, pcc=PCC)

        if pcc_value is not None:
            logger.info(
                "Output '{}': PCC={:.6f} (required={:.2f}), max|Δ|={:.3e} — {}",
                name,
                pcc_value,
                PCC,
                atol_value,
                "PASS" if passed else "FAIL",
            )
        else:
            logger.info(
                "Output '{}' (scalar): max|Δ|={:.3e} — {}",
                name,
                atol_value,
                "PASS" if passed else "FAIL",
            )

        if not passed:
            failures.append((name, pcc_value, atol_value))

    if failures:
        summary = "\n".join(
            f"  {name}: PCC={pcc:.6f}, max|Δ|={atol:.3e}" if pcc is not None else f"  {name}: max|Δ|={atol:.3e}"
            for name, pcc, atol in failures
        )
        raise AssertionError(
            f"Ground-truth validation failed: "
            f"{len(failures)}/{len(output_names)} output(s) did not meet PCC={PCC}:\n{summary}"
        )

    print(f"[bev_benchmark] Ground-truth validation passed ({len(output_names)} output(s)).")


@pytest.fixture(scope="module")
def bev_assets():
    if not assets_available():
        paths = bev_paths()
        pytest.skip(
            f"BEV assets not found under {paths['root']}. "
            "Set BEV_ASSETS_DIR or populate model/input_samples/output_samples."
        )
    return bev_paths()


@pytest.mark.push
def test_bev_onnx_benchmark(bev_assets):
    onnx_model = onnx.load(str(bev_assets["model"]))
    onnx.checker.check_model(onnx_model)

    sequences = list_sequences()
    seq_id = sequences[0]
    sample_inputs = load_inputs(seq_id)

    pool_size = min(N_TIMED + N_WARMUP, max(len(sequences), 4))
    frames_pool = [load_inputs(sequences[i % len(sequences)]) for i in range(pool_size)]

    n_inputs = len(sample_inputs)
    print(f"\n[bev_benchmark] Compiling model ({n_inputs} inputs, seq={seq_id}) …")
    compiled = _compile_model(onnx_model, sample_inputs)
    print("[bev_benchmark] Compilation done.")

    _validate_against_ground_truth(compiled, onnx_model, sample_inputs, seq_id)

    mean_infer, std_infer, mean_total, samples_per_sec = _run_benchmark(
        compiled,
        frames_pool,
        n_warmup=N_WARMUP,
        n_timed=N_TIMED,
        batch_size=BATCH_SIZE,
    )
    _print_table(
        "BEV ONNX Benchmark (batch_size=1)",
        [("BEV simple_bev_prep", mean_infer, std_infer, mean_total, samples_per_sec)],
    )
