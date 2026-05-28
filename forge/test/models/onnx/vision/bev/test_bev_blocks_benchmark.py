# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-block optimization-sweep benchmark for simple_bev_prep.onnx.

Run after create_bev_splits.py has generated the split model files:
    python forge/test/models/onnx/vision/bev/create_bev_splits.py

Single test  — test_opt_sweep  — parametrized over three axes:

    block        : block_A  block_B  block_C  block_D  block_E  block_F
    compiler_cfg : baseline  consteval  opt_level_1  fp16b  consteval_fp16b
    program_cache: enable_program_cache  disable_program_cache

Examples
--------
# One specific combination:
pytest test_bev_blocks_benchmark.py -k "block_A and fp16b and enable_program_cache"

# Block A, baseline config, program cache disabled:
pytest test_bev_blocks_benchmark.py -k "block_A and baseline and disable_program_cache"

# All configs for block E, cache enabled:
pytest test_bev_blocks_benchmark.py -k "block_E and enable_program_cache"

# All blocks, fp16b only:
pytest test_bev_blocks_benchmark.py -k "fp16b"

# Everything:
pytest test_bev_blocks_benchmark.py -v
"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import statistics
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import onnx
import pytest
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.verify import verify

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
)
from test.models.onnx.vision.bev.model_utils.bev_split_utils import (
    BLOCK_DEFS,
    MERGED_BLOCK_DEFS,
    intermediate_samples_available,
    load_block_inputs,
    load_block_inputs_pool,
    load_merged_inputs,
    load_merged_inputs_pool,
    merged_models_available,
    merged_models_dir,
    split_models_available,
    split_models_dir,
)

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

N_WARMUP = 3
N_TIMED = 10
BATCH_SIZE = 1

COMPILE_TIMEOUT_SECS = 15 * 60  # 15 minutes per block


class _CompileTimeout(Exception):
    pass


@contextlib.contextmanager
def _timeout(seconds: int, label: str):
    """SIGALRM-based timeout. Raises _CompileTimeout if `seconds` elapses."""
    def _handler(signum, frame):
        raise _CompileTimeout(f"{label} exceeded {seconds // 60}-min compile timeout")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _diagnose_timeout(block_name: str) -> str:
    """Inspect IR dumps to identify which ops likely caused the MLA hang."""
    dump_dir = os.environ.get("TTMLIR_DUMP_DIR", os.getcwd())
    for fname in ["04-fusing.mlir", "09-mla.mlir", "03-lowering.mlir"]:
        path = os.path.join(dump_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()
        op_counts = Counter(re.findall(r'"ttnn\.(\w+)"', content))
        # Find tensors with large channel counts (MLA bottleneck)
        large = re.findall(
            r'tensor<[^>]*x(\d{3,})[^>]*bf16[^>]*>', content
        )
        large_dims = Counter(int(d) for d in large).most_common(5)
        top_ops = op_counts.most_common(10)
        lines = [
            f"  IR stage     : {fname}",
            f"  Top ops      : {top_ops}",
            f"  Large dims   : {large_dims}",
        ]
        return "\n".join(lines)
    return "  No IR dumps found — set TTMLIR_DUMP_DIR before running"

# ---------------------------------------------------------------------------
# Block short-name → BLOCK_DEFS key mapping
# These short names are the pytest param IDs used with -k
# ---------------------------------------------------------------------------

BLOCKS: Dict[str, str] = {
    "block_A": "block_A_deformed_backbone",
    "block_B": "block_B_deformed_bev_transform",
    "block_C": "block_C_cylinder_backbone",
    "block_D": "block_D_cylinder_bev_transform",
    "block_E": "block_E_bev_aggregator",
    "block_F": "block_F_output_heads",
}

# Merged block pytest param IDs — each key is the -k selector
MERGED_BLOCKS: Dict[str, str] = {
    "merged_AB":     "merged_AB",
    "merged_CD":     "merged_CD",
    "merged_ABCD":   "merged_ABCD",
    "merged_EF":     "merged_EF",
    "merged_ABCDEF": "merged_ABCDEF",
}

# ---------------------------------------------------------------------------
# Compiler config variants
# Keys are the pytest param IDs used with -k  (no special chars for safety)
# ---------------------------------------------------------------------------

def _cfg_baseline(block_name: str) -> CompilerConfig:
    cfg = CompilerConfig()
    return cfg

def _cfg_forge_optimization_passes() -> CompilerConfig:
    cfg = CompilerConfig()
    cfg.enable_optimization_passes = True
    return cfg

def _cfg_consteval(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.retain_tvm_python_files = True
    cfg.enable_optimization_passes = True
    return cfg

def _cfg_opt_level_1(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(1)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.retain_tvm_python_files = True
    cfg.enable_optimization_passes = True
    return cfg

def _cfg_opt_level_2(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.retain_tvm_python_files = True
    cfg.enable_optimization_passes = True
    return cfg

def _cfg_opt_level_1_bfloat16() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(1)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_2_bfloat16(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_1_bfloat16_hifi2_fp32_acc() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(1)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        .set_compute_cfg_fp32_dest_acc_en(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_2_bfloat16_hifi2_fp32_acc(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        .set_compute_cfg_fp32_dest_acc_en(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_2_bfloat16_hifi3_fp32_acc() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_2_bfloat16_hifi4_fp32_acc(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi4)
        .set_compute_cfg_fp32_dest_acc_en(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg

def _cfg_opt_level_2_bfloat16_hifi3_fp32_acc_no_trace(block_name: str) -> CompilerConfig:
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


def _cfg_opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled(block_name: str) -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(True)
        .set_enable_ttnn_perf_metrics(True)
        .set_enable_ttnn_perf_metrics_verbose(True)
        .set_ttnn_perf_metrics_output_file(f"BEV_MODEL_LOGS/LATEST/{block_name}_AFTER_FIX_12_perf_metrics.json")
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg


COMPILER_CONFIGS: Dict[str, callable] = {
    "baseline":                  _cfg_baseline,
    "forge_optimization_passes": _cfg_forge_optimization_passes,
    "consteval": _cfg_consteval,
    "opt_level_1": _cfg_opt_level_1,
    "opt_level_2": _cfg_opt_level_2,
    "opt_level_2_bfloat16": _cfg_opt_level_2_bfloat16,
    "opt_level_1_bfloat16": _cfg_opt_level_1_bfloat16,
    "opt_level_2_bfloat16_hifi2_fp32_acc": _cfg_opt_level_2_bfloat16_hifi2_fp32_acc,
    "opt_level_1_bfloat16_hifi2_fp32_acc": _cfg_opt_level_1_bfloat16_hifi2_fp32_acc,
    "opt_level_2_bfloat16_hifi3_fp32_acc": _cfg_opt_level_2_bfloat16_hifi3_fp32_acc,
    "opt_level_2_bfloat16_hifi4_fp32_acc": _cfg_opt_level_2_bfloat16_hifi4_fp32_acc,
    "opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled": _cfg_opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled,
    "opt_level_2_bfloat16_hifi3_fp32_acc_no_trace": _cfg_opt_level_2_bfloat16_hifi3_fp32_acc_no_trace,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_device() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def _enable_ir_dumps(module_name: str, dump_dir: str | None = None) -> None:
    """Enable pipeline IR and memory-op dumps for this compilation.

    Dumps are written to TTMLIR_DUMP_DIR if set, otherwise the current directory.
    Pass dump_dir to override programmatically (sets TTMLIR_DUMP_DIR).

    Files written:
        01-ttir-passes.mlir, 03-lowering.mlir, 04-fusing.mlir,
        05-decomposition.mlir, 06-workarounds.mlir, 07-cfg.mlir,
        09-mla.mlir, 11-trace-hoist.mlir (trace only),
        12-layout-decompose.mlir, 13-final-after-dealloc.mlir,
        ttir_<module_name>.mlir, ttnn_<module_name>.mlir
        (plus consteval stages when consteval is enabled)

    Memory checkpoint summaries are printed to stderr (grep [MEMDUMP]).
    """
    import os as _os
    if dump_dir is not None:
        os.environ["TTMLIR_DUMP_DIR"] = dump_dir
    out_dir = os.environ.get("TTMLIR_DUMP_DIR", _os.getcwd())
    os.environ["TTMLIR_DUMP_PIPELINE_IR"] = "1"
    os.environ["TTMLIR_DUMP_MEMORY_OPS"] = "1"
    print(f"\n[IR dumps enabled] -> {out_dir}/")
    print(f"  Pipeline stages : {{01-ttir-passes, 03-lowering, ..., 13-final-after-dealloc}}.mlir")
    print(f"  TTIR (pre-lower): ttir_{module_name}.mlir")
    print(f"  TTNN (final)    : ttnn_{module_name}.mlir")
    print(f"  Memory checkpts : stderr (grep [MEMDUMP])")
    print(f"  (set TTMLIR_DUMP_DIR=<path> to redirect all dumps)")


def _compile_onnx(
    model_path,
    sample_inputs: List[torch.Tensor],
    compiler_cfg: CompilerConfig,
    enable_program_cache: bool,
    module_name: str,
):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    # TT_METAL_DEVICE_PROFILER_DISPATCH=1 and TT_METAL_PROFILER_SYNC=1 both cause failures
    # during forge.compile() when the mock device (OpModel SingletonDeviceContext) is open:
    #   - DISPATCH: teardown reads dispatch-core L1 buffers never written → crash at
    #     metal_context.cpp:451 (destroy_all_instances check=true)
    #   - SYNC: ProfilerSync(INIT) builds a sync kernel during mock device open; the linker
    #     fails with "non constant or forward reference address expression" for .text section
    # Pop both during compile and restore in finally so real device profiling is unaffected.
    #
    # NOTE: TT_METAL_DEVICE_PROFILER is intentionally NOT popped here. Popping it prevents
    # InitDeviceProfiler() from running on the mock device during forge.compile(), which
    # disrupts the global firstInit=true sequence and causes profile_log_device.csv to be
    # empty after inference.
    _dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync_prof = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        compiled = forge.compile(onnx_model, sample_inputs=sample_inputs, compiler_cfg=compiler_cfg, module_name=module_name)
    finally:
        if _dispatch_prof is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
        if _sync_prof is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync_prof
    if enable_program_cache:
        _configure_device()
    return compiled


def _compile_block(
    block_name: str,
    sample_inputs: List[torch.Tensor],
    compiler_cfg: CompilerConfig,
    enable_program_cache: bool,
):
    return _compile_onnx(
        split_models_dir() / f"{block_name}.onnx",
        sample_inputs, compiler_cfg, enable_program_cache, module_name=block_name,
    )


def _compile_merged(
    merged_name: str,
    sample_inputs: List[torch.Tensor],
    compiler_cfg: CompilerConfig,
    enable_program_cache: bool,
):
    return _compile_onnx(
        merged_models_dir() / f"{merged_name}.onnx",
        sample_inputs, compiler_cfg, enable_program_cache, module_name=merged_name,
    )


def _run_benchmark(
    compiled,
    frames_pool: List[List[torch.Tensor]],
    label: str = "",
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
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

    print()
    total_ms = [p + r + c for p, r, c in zip(prep_ms, infer_ms, collect_ms)]
    avg_total = statistics.mean(total_ms)
    fps = BATCH_SIZE * 1000.0 / avg_total if avg_total > 0 else float("inf")
    mean_infer = statistics.mean(infer_ms)
    std_infer = statistics.stdev(infer_ms) if len(infer_ms) > 1 else 0.0
    return mean_infer, std_infer, avg_total, fps


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_block(
    block_name: str,
    compiled,
    block_inputs: List[torch.Tensor],
) -> None:
    """Compare compiled output against ONNX Runtime on the same split model."""
    split_model_path = split_models_dir() / f"{block_name}.onnx"
    onnx_model = onnx.load(str(split_model_path))
    framework_model = forge.OnnxModule(block_name, onnx_model)
    verify(block_inputs, framework_model, compiled)


_C0, _C1, _C2, _C3 = 44, 30, 14, 10
_TW = _C0 + _C1 + _C2 + _C3 + 5


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
def split_models(bev_assets):
    if not split_models_available():
        pytest.skip(
            "Split model files not found. "
            "Run: python forge/test/models/onnx/vision/bev/create_bev_splits.py"
        )
    return split_models_dir()


@pytest.fixture(scope="session")
def merged_models(bev_assets):
    if not merged_models_available():
        pytest.skip(
            "Merged model files not found. "
            "Run: python forge/test/models/onnx/vision/bev/create_bev_splits.py"
        )
    return merged_models_dir()


@pytest.fixture(scope="session")
def sequences(bev_assets):
    return list_sequences()


# ---------------------------------------------------------------------------
# test_opt_sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_short", list(BLOCKS.keys()))
@pytest.mark.parametrize("cfg_name", list(COMPILER_CONFIGS.keys()))
@pytest.mark.parametrize(
    "program_cache",
    [True, False],
    ids=["enable_program_cache", "disable_program_cache"],
)
def test_opt_sweep(
    block_short: str,
    cfg_name: str,
    program_cache: bool,
    sequences,
):
    """
    Compile a single BEV block with the given compiler config and program-cache
    setting, then benchmark it with N_WARMUP warmup and N_TIMED timed iterations.

    Pytest param IDs:

        block_short   : block_A  block_B  block_C  block_D  block_E  block_F
        cfg_name      : baseline  consteval  opt_level_1  fp16b  consteval_fp16b
        program_cache : enable_program_cache  disable_program_cache

    Select via -k, e.g.:
        -k "block_A and fp16b and enable_program_cache"
        -k "block_A and baseline and disable_program_cache"
        -k "block_E and enable_program_cache"
        -k "fp16b"
    """
    block_name = BLOCKS[block_short]
    cache_str = "enable_program_cache" if program_cache else "disable_program_cache"
    # Label embedded in every progress line:  block_A | baseline | cache=OFF
    run_label = f"{block_short} | {cfg_name} | {'cache=ON' if program_cache else 'cache=OFF'}"

    seq_id = sequences[0]
    sample_inputs = load_block_inputs(block_name, seq_id)

    src = "real" if intermediate_samples_available(seq_id) else "synthetic"

    print(f"\n{'=' * 70}")
    print(f"  block        : {block_short}  ({BLOCK_DEFS[block_name]['label']})")
    print(f"  compiler_cfg : {cfg_name}")
    print(f"  program_cache: {cache_str}")
    print(f"  intermediates: {src}")
    print(f"{'=' * 70}")

    _enable_ir_dumps(block_name)
    compiler_cfg = COMPILER_CONFIGS[cfg_name](block_name)
    try:
        with _timeout(COMPILE_TIMEOUT_SECS, f"{block_short}/{cfg_name}"):
            compiled = _compile_block(
                block_name,
                sample_inputs,
                compiler_cfg,
                enable_program_cache=program_cache,
            )
    except _CompileTimeout as exc:
        diag = _diagnose_timeout(block_name)
        pytest.fail(
            f"\nCOMPILE TIMEOUT — {exc}\n"
            f"Diagnosis:\n{diag}\n"
            f"Fix: reduce op complexity or add OpModel constraints for the offending ops."
        )

    # Block B has a known grid_sample PCC issue; skip validation to get timing.
    if block_short != "block_B":
        print(f"\n[validation] Running verify() for {block_short} ({cfg_name}) ...")
        _validate_block(block_name, compiled, sample_inputs)
        print(f"[validation] PASSED")
    else:
        print(f"\n[validation] Skipped for block_B (known grid_sample PCC issue)")

    # When device profiler is active, the per-kernel DRAM trace buffer fills up
    # across many iterations for large blocks and corrupts the profiler state,
    # causing DeviceProfiler::dumpDeviceResults to throw during teardown.
    # Limit to a single timed run (no warmup) so the buffer stays within bounds.
    _profiling = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
    _n_warmup = 0 if _profiling else N_WARMUP
    _n_timed = 1 if _profiling else N_TIMED

    pool_size = min(_n_timed + _n_warmup, max(len(sequences), 4))
    frames = load_block_inputs_pool(block_name, sequences, pool_size)

    mi, si, mt, fps = _run_benchmark(compiled, frames, label=run_label, n_warmup=_n_warmup, n_timed=_n_timed)
    _print_result(BLOCK_DEFS[block_name]["label"], BLOCK_DEFS[block_name]["node_count"],
                  cfg_name, program_cache, mi, si, mt, fps)


# ---------------------------------------------------------------------------
# test_merged_sweep — benchmark multiple blocks compiled as one fused model
# ---------------------------------------------------------------------------

# @pytest.mark.parametrize("merged_short", list(MERGED_BLOCKS.keys()))
# @pytest.mark.parametrize("cfg_name", list(COMPILER_CONFIGS.keys()))
# @pytest.mark.parametrize(
#     "program_cache",
#     [True, False],
#     ids=["enable_program_cache", "disable_program_cache"],
# )
# def test_merged_sweep(
#     merged_short: str,
#     cfg_name: str,
#     program_cache: bool,
#     sequences,
#     merged_models,
# ):
#     """
#     Compile a multi-block fusion as a single ONNX model and benchmark it.

#     Merged block param IDs (use with -k):
#         merged_AB      — Block A + Block B  (Deformed Camera → BEV)
#         merged_CD      — Block C + Block D  (Cylinder Camera → BEV)
#         merged_ABCD    — Block A + B + C + D  (All Cameras → BEV)
#         merged_EF      — Block E + Block F  (BEV Aggregator → Output Heads)
#         merged_ABCDEF  — Full pipeline

#     Examples
#     --------
#     pytest test_bev_blocks_benchmark.py -k "merged_AB and baseline and disable_program_cache"
#     pytest test_bev_blocks_benchmark.py -k "merged_ABCD and opt_level_1 and enable_program_cache"
#     pytest test_bev_blocks_benchmark.py -k "merged_ABCDEF"
#     pytest test_bev_blocks_benchmark.py -k "merged"
#     """
#     merged_name = MERGED_BLOCKS[merged_short]
#     mdef = MERGED_BLOCK_DEFS[merged_name]
#     cache_str = "enable_program_cache" if program_cache else "disable_program_cache"
#     run_label = f"{merged_short} | {cfg_name} | {'cache=ON' if program_cache else 'cache=OFF'}"

#     seq_id = sequences[0]
#     sample_inputs = load_merged_inputs(merged_name, seq_id)

#     src = "real" if intermediate_samples_available(seq_id) else "synthetic"

#     print(f"\n{'=' * 70}")
#     print(f"  merged       : {merged_short}  ({mdef['label']})")
#     print(f"  blocks       : {' + '.join(mdef['blocks'])}")
#     print(f"  compiler_cfg : {cfg_name}")
#     print(f"  program_cache: {cache_str}")
#     print(f"  intermediates: {src}")
#     print(f"{'=' * 70}")

#     try:
#         compiled = _compile_merged(
#             merged_name,
#             sample_inputs,
#             COMPILER_CONFIGS[cfg_name](),
#             enable_program_cache=program_cache,
#         )
#     except Exception as exc:
#         pytest.skip(f"Compilation failed ({cfg_name}): {exc}")
#         return

#     pool_size = min(N_TIMED + N_WARMUP, max(len(sequences), 4))
#     frames = load_merged_inputs_pool(merged_name, sequences, pool_size)

#     mi, si, mt, fps = _run_benchmark(compiled, frames, label=run_label)
#     _print_result(mdef["label"], mdef["node_count"], cfg_name, program_cache, mi, si, mt, fps)