# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import onnx
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from third_party.tt_forge_models.resnet.pytorch.loader import ModelLoader, ModelVariant
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# Separate ONNX file for batch_size=8 to avoid colliding with the batch_size=1 file
# used by test_resnet_benchmark.py.
ONNX_PATH = _HERE / "models" / "resnet50_imagenet1k.onnx"

# ---------------------------------------------------------------------------
# Defaults (mirrors tt-xla test_vision.py)
# ---------------------------------------------------------------------------
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOOP_COUNT = 128
DEFAULT_INPUT_SIZE = (3, 224, 224)  # (channels, height, width)
# ONNX model is exported in float32; the compiler's default_df_override handles
# device-side precision (Float16_b / bfloat16).
DEFAULT_DATA_FORMAT = torch.float32
DEFAULT_REQUIRED_PCC = 0.99

WARMUP_STEPS = 32


# ---------------------------------------------------------------------------
# PCC helpers
# ---------------------------------------------------------------------------
def _compute_pcc_single(golden_flat: torch.Tensor, device_flat: torch.Tensor) -> float:
    g_c = golden_flat - golden_flat.mean()
    d_c = device_flat - device_flat.mean()
    denom = g_c.norm() * d_c.norm()
    if denom == 0:
        return 1.0 if torch.allclose(golden_flat, device_flat, rtol=1e-2, atol=1e-2) else 0.0
    pcc = (g_c @ d_c) / denom
    return float(max(-1.0, min(1.0, pcc.item())))


def compute_pcc(
    golden_output: torch.Tensor,
    device_output: torch.Tensor,
    required_pcc: float = 0.99,
) -> float:
    """Compute PCC between CPU golden output and Forge device output."""
    golden_flat = golden_output.to(torch.float32).flatten()
    device_flat = device_output.to(torch.float32).flatten()
    pcc_value = _compute_pcc_single(golden_flat, device_flat)
    logger.info(f"PCC check: Calculated PCC={pcc_value:.6f}, Required PCC={required_pcc}")
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. Calculated: pcc={pcc_value:.6f}. Required: pcc={required_pcc}"
    return pcc_value


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------
def _export_onnx(pytorch_model: torch.nn.Module, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Export a ResNet-50 PyTorch model to ONNX opset 17.

    Re-exports unconditionally so the ONNX file always matches the requested
    batch_size.  The caller is responsible for loading and checking the
    resulting file with ``onnx.load`` / ``onnx.checker.check_model``.
    """
    input_shape = (batch_size, *DEFAULT_INPUT_SIZE)
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            dummy,
            str(ONNX_PATH),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )

    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)


# ---------------------------------------------------------------------------
# Forge compile + device helpers
# ---------------------------------------------------------------------------
def _get_compiler_cfg() -> CompilerConfig:
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


def _configure_device_settings() -> None:
    from forge._C import runtime as forge_runtime

    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = True
    forge_runtime.experimental.configure_devices(device_settings)


def _compile_forge_model(
    onnx_model: onnx.ModelProto,
    sample_input: torch.Tensor,
):
    """Compile a model with Forge and configure device settings."""
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    compiled = forge.compile(
        onnx_model,
        sample_inputs=[sample_input],
        compiler_cfg=_get_compiler_cfg(),
    )
    _configure_device_settings()
    return compiled


# ---------------------------------------------------------------------------
# Benchmark execution (mirrors execute_and_measure_fps in tt-xla)
# ---------------------------------------------------------------------------
def _execute_and_measure(
    compiled,
    inputs: list[torch.Tensor],
    loop_count: int,
) -> tuple[list[torch.Tensor], float]:
    """Run compiled Forge model for loop_count iterations.

    Mirrors tt-xla's execute_and_measure_fps: all iterations run in sequence,
    total wall-clock time (H2D + inference + D2H) is measured end-to-end.

    Returns:
        (predictions, total_time_seconds)
    """
    predictions: list[torch.Tensor] = []

    start_time = time.perf_counter_ns()
    for i in range(loop_count):
        t0 = time.perf_counter_ns()
        outputs = compiled(inputs[i % len(inputs)])
        # Forge compiled model returns a list of tensors; first tensor = logits
        predictions.append(outputs[0])
        t1 = time.perf_counter_ns()
        logger.info(f"Iteration {i} took {(t1 - t0) / 1e6:.4f} ms")
    end_time = time.perf_counter_ns()

    total_time = (end_time - start_time) / 1e9
    logger.info(f"Total time: {total_time:.4f}s for {loop_count} iterations")
    return predictions, total_time


# ---------------------------------------------------------------------------
# Core benchmark function (mirrors benchmark_vision_torch_xla)
# ---------------------------------------------------------------------------
def benchmark_vision_forge(
    model,
    model_info_name: str,
    load_inputs_fn: Callable,
    extract_golden_output_fn: Callable,
    batch_size: int = DEFAULT_BATCH_SIZE,
    loop_count: int = DEFAULT_LOOP_COUNT,
    input_size: tuple = DEFAULT_INPUT_SIZE,
    data_format: torch.dtype = DEFAULT_DATA_FORMAT,
    required_pcc: float = DEFAULT_REQUIRED_PCC,
    onnx_model: Optional["onnx.ModelProto"] = None,
) -> dict:

    # Generate input pool (loop_count batches, each of shape (batch_size, *input_size))
    logger.info(f"Generating {loop_count} input batches (batch_size={batch_size})...")
    inputs = [load_inputs_fn(batch_size, data_format) for _ in range(loop_count)]

    # CPU golden output for PCC validation (always runs the PyTorch model on CPU).
    logger.info("Computing CPU golden output...")
    golden_output = extract_golden_output_fn(model.cpu_eval_forward(inputs[0]))

    # Compile with Forge — prefer ONNX model when provided, otherwise use PyTorch model.
    logger.info(f"Compiling {'ONNX' if onnx_model is not None else 'PyTorch'} model with Forge...")
    compiled = _compile_forge_model(onnx_model, inputs[0])

    # Warmup (mirrors tt-xla: min(WARMUP_STEPS, loop_count) iterations)
    warmup_count = min(WARMUP_STEPS, loop_count)
    logger.info(f"Starting warmup ({warmup_count} iterations)...")
    _execute_and_measure(compiled, inputs[:warmup_count], warmup_count)
    logger.info("Warmup completed.")

    # Timed benchmark run
    logger.info(f"Starting benchmark ({loop_count} iterations)...")
    predictions, total_time = _execute_and_measure(compiled, inputs, loop_count)
    logger.info("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time if total_time > 0 else float("inf")

    # PCC verification: compare first device prediction against CPU golden
    pcc_value = compute_pcc(
        golden_output[0],
        predictions[0],
        required_pcc=required_pcc,
    )
    logger.info(f"PCC verification passed with PCC={pcc_value:.6f}")

    date = datetime.now().strftime("%d-%m-%Y")

    # Print formatted summary (mirrors print_benchmark_results in tt-xla utils.py)
    print("====================================================================")
    print(f"| {model_info_name} Benchmark Results:".ljust(67) + "|")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_info_name}")
    print(f"| Model type: Vision, Random Input Data")
    print(f"| Dataset name: Random Data")
    print(f"| Date: {date}")
    print(f"| Total execution time: {total_time:.4f}s")
    print(f"| Total samples: {total_samples}")
    print(f"| Samples per second: {samples_per_sec:.2f}")
    print(f"| Batch size: {batch_size}")
    print(f"| Input size: {input_size}")
    print("====================================================================")


# ---------------------------------------------------------------------------
# ResNet-50 test (mirrors test_resnet50 in tt-xla test_vision.py)
# ---------------------------------------------------------------------------
def test_resnet50():

    batch_size = 8

    variant = ModelVariant.RESNET_50
    loader = ModelLoader(variant=variant)
    module_name = loader.get_model_info(variant=variant).name
    pytorch_model = loader.load_model()
    pytorch_model.eval()

    # Export to ONNX and reload for validation + compilation.
    logger.info(f"Exporting ResNet-50 to ONNX (batch_size={batch_size})...")
    _export_onnx(pytorch_model, batch_size=batch_size)
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX export verified.")

    framework_model = forge.OnnxModule(module_name, onnx_model)
    framework_model.set_data_format_override(forge._C.DataFormat.Float16_b)

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_golden_output_fn(output):
        # Torchvision ResNet returns a raw classification tensor (no .logits wrapper).
        return output

    benchmark_vision_forge(
        model=framework_model,
        model_info_name=module_name,
        load_inputs_fn=load_inputs_fn,
        extract_golden_output_fn=extract_golden_output_fn,
        batch_size=batch_size,
        onnx_model=onnx_model,
    )
