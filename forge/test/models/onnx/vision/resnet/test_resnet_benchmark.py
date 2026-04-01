# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import statistics
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Union


import numpy as np
import onnx
import pytest
import torch

import forge
from forge._C import MLIRConfig

from forge.config import CompilerConfig

from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


# Constants for the high-throughput ONNX benchmark (batch_size=8, more iterations).
_ONNX_BENCH_BATCH_SIZE = 8
_ONNX_BENCH_N_TIMED = 128
_ONNX_BENCH_N_WARMUP = 32
# Separate file to avoid overwriting the batch_size=1 ONNX used by other tests.
ONNX_B8_PATH = _HERE / "models" / "resnet50_imagenet1k_b8.onnx"


# ---------------------------------------------------------------------------
# Run logging (inlined from helper.py)
# ---------------------------------------------------------------------------


class _Tee:
    """Proxy that writes to *original* and *log_file* simultaneously."""

    def __init__(self, original: IO[str], log_file: IO[str]) -> None:
        self._orig = original
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._orig.write(data)
        self._log.write(data)
        return n

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


@contextlib.contextmanager
def _run_logging(script_name: str, log_dir: Union[str, Path] = "logs"):
    """Context manager: tee stdout + stderr to a timestamped log file.

    Args:
        script_name: Short identifier included in the log file name.
        log_dir:     Directory to write log files into, resolved relative to
                     the current working directory (default: ``"logs"``).

    Yields:
        :class:`pathlib.Path` — absolute path of the opened log file.
    """
    log_dir_path = Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir_path / f"{timestamp}_{script_name}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _Tee(orig_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(orig_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"[logging] Run log: {log_path}")
            yield log_path
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.flush()

    print(f"[logging] Log saved → {log_path}")


# ---------------------------------------------------------------------------
# Benchmark harness (inlined from helper.py)
# ---------------------------------------------------------------------------


def _fmt(vals: list[float]) -> str:
    """Format a list of millisecond measurements as ``'  mean ± std ms'``."""
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
    """Print the canonical benchmark results table.

    Args:
        batch_size: Number of samples per inference call.  Used to convert
            per-batch latency into samples/sec (FPS).  Default is 1 so that
            existing callers are unaffected.

    Returns:
        ``(mean_infer_ms, std_infer_ms, mean_total_ms, samples_per_sec)``
    """
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_total_ms = statistics.mean(total_ms)
    # samples_per_sec = (samples per batch) / (seconds per batch)
    samples_per_sec = batch_size * 1000.0 / avg_total_ms if avg_total_ms > 0 else float("inf")

    mean_infer = statistics.mean(infer_ms)
    std_infer = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0

    mean_infer_ms = statistics.mean(infer_ms)

    print()
    print("=" * 58)
    print(f"  Benchmark results  ({n_timed} iters, {n_warmup} warmup, batch={batch_size})")
    print("=" * 58)
    print(f"  Data prep      (torch.from_numpy) :  {_fmt(prep_ms)}")
    print(f"  Inference      (H2D + run + D2H)  :  {_fmt(infer_ms)}")
    if batch_size > 1:
        print(f"  Inference      per image          :  {mean_infer_ms / batch_size:8.3f} ms")
    print(f"  Output collect (numpy realise)    :  {_fmt(collect_ms)}")
    print("-" * 58)
    print(f"  Total per {'batch' if batch_size > 1 else 'frame'}              " f"         :  {_fmt(total_ms)}")
    print(f"  Average samples/sec               :  {samples_per_sec:8.2f}")
    print("=" * 58)

    return mean_infer, std_infer, avg_total_ms, samples_per_sec


def _run_benchmark(
    compiled,
    frames: list[Union[torch.Tensor, np.ndarray]],
    n_warmup: int,
    n_timed: int,
    batch_size: int = 1,
) -> tuple[float, float, float, float]:
    """Run a compiled Forge model through warmup and timed iterations.

    Each timed frame passes through three measured phases:

    1. **data_prep**  — ensure the frame is a :class:`torch.Tensor`.
    2. **inference**  — ``compiled(tensor)``  (H2D + device run + D2H).
    3. **collect**    — materialise every output tensor via ``np.asarray()``.

    Args:
        compiled:   A Forge-compiled model (callable).
        frames:     Input frame pool (``torch.Tensor`` or ``np.ndarray``).
        n_warmup:   Number of untimed warmup iterations.
        n_timed:    Number of timed iterations.
        batch_size: Samples per inference call; used for correct FPS reporting.

    Returns:
        ``(mean_infer_ms, std_infer_ms, mean_total_ms, samples_per_sec)``
    """
    if not frames:
        raise ValueError("frames pool must not be empty")

    if n_warmup > 0:
        print(f"[benchmark] Warmup ({n_warmup} frames) …")
        for i in range(n_warmup):
            frame = frames[i % len(frames)]
            t = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)
            out = compiled(t)
            _ = [np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy() for o in out]
        print("[benchmark] Warmup done.")

    print(f"[benchmark] Timed run ({n_timed} frames) …")
    prep_ms: list[float] = []
    infer_ms: list[float] = []
    collect_ms: list[float] = []

    for i in range(n_timed):
        frame = frames[(n_warmup + i) % len(frames)]

        t0 = time.perf_counter()
        inp = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)
        prep_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        out = compiled(inp)
        infer_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        _ = [np.asarray(o) if o.dtype != torch.bfloat16 else o.detach().cpu().float().numpy() for o in out]
        collect_ms.append((time.perf_counter() - t0) * 1e3)

        pct = int((i + 1) / n_timed * 100)
        print(f"\r[benchmark] {i+1:4d}/{n_timed}  ({pct:3d}%)", end="", flush=True)

    print()

    return _print_results_table(n_timed, n_warmup, prep_ms, infer_ms, collect_ms, batch_size=batch_size)


def _export_onnx(
    variant: ModelVariant = ModelVariant.RESNET_50,
    batch_size: int = 1,
    out_path: Path = ONNX_B8_PATH,
) -> None:
    """Export ResNet-50 to ONNX opset 17.

    Args:
        variant:    Which ResNet variant to export.
        batch_size: Batch size baked into the exported ONNX graph.
        out_path:   Destination ONNX file.  Defaults to ``ONNX_PATH``
                    (batch_size=1).  Pass ``ONNX_B8_PATH`` for batch_size=8.
    """
    model_loader = ModelLoader(variant=variant)
    model = model_loader.load_model()
    input_shape = (batch_size, 3, 224, 224)
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)


def _random_input(batch_size: int = 1) -> torch.Tensor:
    return torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)


def _get_compiler_cfg():
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_enable_trace(True)
        .set_enable_l1_interleaved_fallback_analysis(True)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        .set_enable_remove_dead_values(True)
    )

    default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg = CompilerConfig(mlir_config=mlir_config, default_df_override=default_df_override)
    compiler_cfg.enable_optimization_passes = True
    return compiler_cfg


def _configure_device_settings():
    from forge._C import runtime as forge_runtime

    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = True
    forge_runtime.experimental.configure_devices(device_settings)


def compile_model(
    onnx_model: onnx.ModelProto,
    sample_input: torch.Tensor,
):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"

    compiled = forge.compile(
        onnx_model,
        sample_inputs=[sample_input],
        compiler_cfg=_get_compiler_cfg(),
    )

    _configure_device_settings()
    return compiled


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

_COL_CFG = 52
_COL_INFER = 36
_COL_TOTAL = 18
_COL_FPS = 12
_TABLE_WIDTH = _COL_CFG + _COL_INFER + _COL_TOTAL + _COL_FPS + 5


def _print_table(
    case_label: str,
    rows: list[tuple[str, float, float, float, float]],
    batch_size: int = 1,
) -> None:
    """Print a formatted comparison table for one benchmark case.

    Args:
        case_label:  Human-readable case heading (e.g. ``"ResNet-50 Benchmark"``).
        rows:        List of ``(config_label, mean_infer_ms, std_infer_ms,
                     mean_total_ms, samples_per_sec)`` tuples.
        batch_size:  Samples per inference call.  When > 1, column headers are
                     updated to say "per batch" instead of "per frame" so that
                     the timing numbers are not misread as per-image latency.
    """
    sep = "-" * _TABLE_WIDTH
    unit = f"per Batch (bs={batch_size})" if batch_size > 1 else "per Frame"

    print()
    print(f"  {case_label}")
    print(sep)
    header = (
        f"| {'Configuration':<{_COL_CFG}}"
        f"| {f'Inference Time (H2D + run + D2H, {unit})':<{_COL_INFER}}"
        f"| {f'Total {unit}':<{_COL_TOTAL}}"
        f"| {'Samples/sec':<{_COL_FPS}}|"
    )
    print(header)
    print(sep)
    for cfg_label, mean_infer, std_infer, mean_total, samples_per_sec in rows:
        row = (
            f"| {cfg_label:<{_COL_CFG}}"
            f"| {mean_infer:.2f} ± {std_infer:.2f} ms{'':<{_COL_INFER - len(f'{mean_infer:.2f} ± {std_infer:.2f} ms')}}"
            f"| {mean_total:.2f} ms{'':<{_COL_TOTAL - len(f'{mean_total:.2f} ms')}}"
            f"| {samples_per_sec:.2f}{'':<{_COL_FPS - len(f'{samples_per_sec:.2f}')}}|"
        )
        print(row)
    print(sep)


@pytest.mark.push
def test_resnet50_onnx_benchmark():
    """ResNet-50 ONNX benchmark compiled and run with batch_size=8.

    Root cause of the old 385 FPS (vs ~888 in test_resnet_vision_benchmark.py):

    1. batch_size=1 → hardware underutilised; per-sample overhead dominated by
       fixed H2D/kernel-launch costs that don't scale with batch size.
    2. Only 5 warmup iterations → first timed runs still slow.
    3. Only 10 timed iterations → noisy, statistically unstable.
    4. FPS = 1000/avg_ms counted 1 sample/batch instead of 8 samples/batch.

    Fixes applied:
      - Export and compile with batch_size=8 (saved to ONNX_B8_PATH).
      - 32 warmup + 128 timed iterations (matches test_resnet_vision_benchmark.py).
      - batch_size threaded through _run_benchmark → _print_results_table so that
        FPS = batch_size * 1000 / avg_ms = samples per second.
    """
    bs = _ONNX_BENCH_BATCH_SIZE  # 8
    n_timed = _ONNX_BENCH_N_TIMED  # 128
    n_warmup = _ONNX_BENCH_N_WARMUP  # 32

    # Export with batch_size=8 to a dedicated ONNX file (does not clobber the
    # batch_size=1 file used by test_resnet50_benchmark and test_resnet50_onnx_vs_pytorch_benchmark).
    _export_onnx(batch_size=bs, out_path=ONNX_B8_PATH)
    onnx_model = onnx.load(str(ONNX_B8_PATH))
    onnx.checker.check_model(onnx_model)

    sample_input = _random_input(batch_size=bs)
    pool_size = min(n_timed + n_warmup, 16)
    tensors = [_random_input(batch_size=bs) for _ in range(pool_size)]

    compiled = compile_model(onnx_model, sample_input)

    with _run_logging("resnet50_onnx_benchmark_b8"):
        mean_infer, std_infer, mean_total, samples_per_sec = _run_benchmark(
            compiled, tensors, n_warmup=n_warmup, n_timed=n_timed, batch_size=bs
        )
        _print_table(
            f"ResNet-50 ONNX Benchmark (batch_size={bs})",
            [("ResNet-50 ONNX", mean_infer, std_infer, mean_total, samples_per_sec)],
            batch_size=bs,
        )
