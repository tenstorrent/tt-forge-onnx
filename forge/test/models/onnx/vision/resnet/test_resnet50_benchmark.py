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
ONNX_PATH = _HERE / "models" / "resnet50_imagenet1k.onnx"
INPUT_SHAPE = (1, 3, 224, 224)

_N_TIMED_DEFAULT = 10
_N_WARMUP_DEFAULT = 5


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
) -> tuple[float, float, float, float]:
    """Print the canonical benchmark results table.

    Returns:
        ``(mean_infer_ms, std_infer_ms, mean_total_ms, fps)``
    """
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_total_ms = statistics.mean(total_ms)
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else float("inf")

    mean_infer = statistics.mean(infer_ms)
    std_infer = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0

    print()
    print("=" * 58)
    print(f"  Benchmark results  ({n_timed} frames, {n_warmup} warmup)")
    print("=" * 58)
    print(f"  Data prep      (torch.from_numpy) :  {_fmt(prep_ms)}")
    print(f"  Inference      (H2D + run + D2H)  :  {_fmt(infer_ms)}")
    print(f"  Output collect (numpy realise)    :  {_fmt(collect_ms)}")
    print("-" * 58)
    print(f"  Total per frame                   :  {_fmt(total_ms)}")
    print(f"  Average FPS                       :  {fps:8.2f} fps")
    print("=" * 58)

    return mean_infer, std_infer, avg_total_ms, fps


def _run_benchmark(
    compiled,
    frames: list[Union[torch.Tensor, np.ndarray]],
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, float, float]:
    """Run a compiled Forge model through warmup and timed iterations.

    Each timed frame passes through three measured phases:

    1. **data_prep**  — ensure the frame is a :class:`torch.Tensor`.
    2. **inference**  — ``compiled(tensor)``  (H2D + device run + D2H).
    3. **collect**    — materialise every output tensor via ``np.asarray()``.

    Args:
        compiled:  A Forge-compiled model (callable).
        frames:    Input frame pool (``torch.Tensor`` or ``np.ndarray``).
        n_warmup:  Number of untimed warmup iterations.
        n_timed:   Number of timed iterations.

    Returns:
        ``(mean_infer_ms, std_infer_ms, mean_total_ms, fps)``
    """
    if not frames:
        raise ValueError("frames pool must not be empty")

    if n_warmup > 0:
        print(f"[benchmark] Warmup ({n_warmup} frames) …")
        for i in range(n_warmup):
            frame = frames[i % len(frames)]
            t = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)
            out = compiled(t)
            _ = [np.asarray(o) for o in out]
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
        _ = [np.asarray(o) for o in out]
        collect_ms.append((time.perf_counter() - t0) * 1e3)

        pct = int((i + 1) / n_timed * 100)
        print(f"\r[benchmark] {i+1:4d}/{n_timed}  ({pct:3d}%)", end="", flush=True)

    print()

    return _print_results_table(n_timed, n_warmup, prep_ms, infer_ms, collect_ms)


# ---------------------------------------------------------------------------
# ResNet-50 helpers
# ---------------------------------------------------------------------------



def _export_onnx(
    variant: ModelVariant = ModelVariant.RESNET_50,
    force: bool = False,
) -> None:
    """Export ResNet-50 to ONNX opset 17, using the cached file when available."""
    if ONNX_PATH.exists() and not force:
        return

    model_loader = ModelLoader(variant=variant)
    model = model_loader.load_model()
    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(ONNX_PATH),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )

    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)


def _random_input() -> torch.Tensor:
    return torch.randn(*INPUT_SHAPE, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Compiler configuration helpers
# ---------------------------------------------------------------------------


def _compile_case1(
    onnx_model,
    sample_input: torch.Tensor,
    disable_prepare_conv2d_weights_and_bias: bool,
):
    """Case 1: Full Forge + MLIR compiler configuration with Program Cache.

    Enables the complete optimizer pipeline (consteval, memory-layout analysis,
    fusing, L1-interleaved fallback) together with ``enable_optimization_passes``
    and the device-side program cache.

    Args:
        onnx_model: Loaded ONNX model proto.
        sample_input: Representative input tensor used for shape inference.
        disable_prepare_conv2d_weights_and_bias: When *True* the
            ``TTNNPrepareConv2dWeightsAndBias`` MLIR pass is disabled via
            ``enable-prepare-conv2d-weights-and-bias=False``.

    Returns:
        Forge-compiled model ready for inference.
    """
    os.environ["TT_METAL_FORCE_REINIT"] = "1"

    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_enable_optimizer(True)
        .set_enable_memory_layout_analysis(True)
        .set_enable_fusing(True)
        .set_enable_fusing_conv2d_with_multiply_pattern(True)
    )

    if disable_prepare_conv2d_weights_and_bias:
        mlir_config.set_custom_config(
            "l1-interleaved-fallback-analysis-enabled=1"
            " enable-prepare-conv2d-weights-and-bias=False"
        )
    else:
        mlir_config.set_custom_config(
            "l1-interleaved-fallback-analysis-enabled=1"
        )

    compiler_cfg = CompilerConfig(mlir_config=mlir_config)
    compiler_cfg.enable_optimization_passes = True

    compiled = forge.compile(
        onnx_model,
        sample_inputs=[sample_input],
        compiler_cfg=compiler_cfg,
    )

    from forge._C import runtime as forge_runtime

    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = True
    forge_runtime.experimental.configure_devices(device_settings)

    return compiled


def _compile_case2(
    onnx_model,
    sample_input: torch.Tensor,
    disable_prepare_conv2d_weights_and_bias: bool,
):
    """Case 2: Only MLIR optimizer enabled (minimal configuration).

    All other MLIR passes (consteval, memory-layout analysis, fusing, …) and
    ``enable_optimization_passes`` are left at their defaults.  No program
    cache is configured.

    Args:
        onnx_model: Loaded ONNX model proto.
        sample_input: Representative input tensor used for shape inference.
        disable_prepare_conv2d_weights_and_bias: When *True* the
            ``TTNNPrepareConv2dWeightsAndBias`` MLIR pass is disabled via
            ``enable-prepare-conv2d-weights-and-bias=False``.

    Returns:
        Forge-compiled model ready for inference.
    """
    mlir_config = MLIRConfig().set_enable_optimizer(True)

    if disable_prepare_conv2d_weights_and_bias:
        mlir_config.set_custom_config("enable-prepare-conv2d-weights-and-bias=False")

    compiler_cfg = CompilerConfig(mlir_config=mlir_config)

    compiled = forge.compile(
        onnx_model,
        sample_inputs=[sample_input],
        compiler_cfg=compiler_cfg,
    )

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
) -> None:
    """Print a formatted comparison table for one benchmark case.

    Args:
        case_label: Human-readable case heading (e.g. ``"Case 1: …"``).
        rows: List of ``(config_label, mean_infer_ms, std_infer_ms,
              mean_total_ms, fps)`` tuples.
    """
    sep = "-" * _TABLE_WIDTH

    print()
    print(f"  {case_label}")
    print(sep)
    header = (
        f"| {'Configuration':<{_COL_CFG}}"
        f"| {'Inference Time (H2D + run + D2H)':<{_COL_INFER}}"
        f"| {'Total per Frame':<{_COL_TOTAL}}"
        f"| {'FPS':<{_COL_FPS}}|"
    )
    print(header)
    print(sep)
    for cfg_label, mean_infer, std_infer, mean_total, fps in rows:
        row = (
            f"| {cfg_label:<{_COL_CFG}}"
            f"| {mean_infer:.2f} ± {std_infer:.2f} ms{'':<{_COL_INFER - len(f'{mean_infer:.2f} ± {std_infer:.2f} ms')}}"
            f"| {mean_total:.2f} ms{'':<{_COL_TOTAL - len(f'{mean_total:.2f} ms')}}"
            f"| {fps:.2f} fps{'':<{_COL_FPS - len(f'{fps:.2f} fps')}}|"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Parameterized per-case tests
# ---------------------------------------------------------------------------

_DISABLE_IDS = ["TTNNPrepareConv2dWeightsAndBias_enabled", "TTNNPrepareConv2dWeightsAndBias_disabled"]


@pytest.mark.parametrize("disable_prepare_conv2d_weights_and_bias", [False, True], ids=_DISABLE_IDS)
def test_resnet50_case1_benchmark(disable_prepare_conv2d_weights_and_bias: bool):
    """Case 1: Full Forge + MLIR compiler configuration with Program Cache.

    Benchmarks ResNet-50 with the complete optimizer pipeline active.
    The ``disable_prepare_conv2d_weights_and_bias`` parameter controls
    whether the ``TTNNPrepareConv2dWeightsAndBias`` MLIR pass is active.
    """
    _export_onnx()
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    sample_input = _random_input()

    n_timed = _N_TIMED_DEFAULT
    n_warmup = _N_WARMUP_DEFAULT
    pool_size = min(n_timed + n_warmup, 8)
    tensors = [_random_input() for _ in range(pool_size)]

    pass_state = "Disabled" if disable_prepare_conv2d_weights_and_bias else "Enabled"
    script = f"resnet50_case1_TTNNPrepareConv2dWeightsAndBias_{pass_state.lower()}"

    compiled = _compile_case1(onnx_model, sample_input, disable_prepare_conv2d_weights_and_bias)

    with _run_logging(script):
        print(f"\nCase 1: Full Forge + MLIR Compiler Configs Enabled with Program Cache")
        print(f"TTNNPrepareConv2dWeightsAndBias: {pass_state}")
        mean_infer, std_infer, mean_total, fps = _run_benchmark(
            compiled, tensors, n_warmup=n_warmup, n_timed=n_timed
        )
        _print_table(
            f"Case 1 — TTNNPrepareConv2dWeightsAndBias {pass_state}",
            [(f"{pass_state} TTNNPrepareConv2dWeightsAndBias", mean_infer, std_infer, mean_total, fps)],
        )


@pytest.mark.parametrize("disable_prepare_conv2d_weights_and_bias", [False, True], ids=_DISABLE_IDS)
def test_resnet50_case2_benchmark(disable_prepare_conv2d_weights_and_bias: bool):
    """Case 2: Only MLIR optimizer enabled.

    Benchmarks ResNet-50 with only ``set_enable_optimizer(True)`` set.
    The ``disable_prepare_conv2d_weights_and_bias`` parameter controls
    whether the ``TTNNPrepareConv2dWeightsAndBias`` MLIR pass is active.
    """
    _export_onnx()
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    sample_input = _random_input()

    n_timed = _N_TIMED_DEFAULT
    n_warmup = _N_WARMUP_DEFAULT
    pool_size = min(n_timed + n_warmup, 8)
    tensors = [_random_input() for _ in range(pool_size)]

    pass_state = "Disabled" if disable_prepare_conv2d_weights_and_bias else "Enabled"
    script = f"resnet50_case2_TTNNPrepareConv2dWeightsAndBias_{pass_state.lower()}"

    compiled = _compile_case2(onnx_model, sample_input, disable_prepare_conv2d_weights_and_bias)

    with _run_logging(script):
        print(f"\nCase 2: Only MLIR Optimizer Enabled")
        print(f"TTNNPrepareConv2dWeightsAndBias: {pass_state}")
        mean_infer, std_infer, mean_total, fps = _run_benchmark(
            compiled, tensors, n_warmup=n_warmup, n_timed=n_timed
        )
        _print_table(
            f"Case 2 — TTNNPrepareConv2dWeightsAndBias {pass_state}",
            [(f"{pass_state} TTNNPrepareConv2dWeightsAndBias", mean_infer, std_infer, mean_total, fps)],
        )

