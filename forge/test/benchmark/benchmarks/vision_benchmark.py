# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Union

from loguru import logger
import onnx
import torch

import forge
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.compare import calculate_pcc, compare_with_golden

from test.benchmark.utils import (
    create_benchmark_result,
    get_benchmark_metadata,
    print_benchmark_results,
)

WARMUP_STEPS = 32


def get_compiler_cfg(
    *,
    optimization_level: int = 2,
    trace_enabled: bool = True,
    data_format: str = "bfloat16",
) -> CompilerConfig:
    """Create the ``CompilerConfig`` used for ONNX vision benchmarks.

    Args:
        optimization_level: MLIR optimization level.
        trace_enabled: Whether tracing is enabled in MLIR config.
        data_format: ``bfloat16`` selects ``Float16_b`` default DF override when set.

    Returns:
        Configured ``CompilerConfig`` instance.
    """
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(optimization_level)
        .set_enable_trace(trace_enabled)
        .set_enable_l1_interleaved_fallback_analysis(True)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        .set_enable_remove_dead_values(True)
    )
    compiler_cfg = CompilerConfig(mlir_config=mlir_config)
    compiler_cfg.enable_optimization_passes = True
    if data_format == "bfloat16":
        compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    return compiler_cfg


def configure_device_settings() -> None:
    from forge._C import runtime as forge_runtime

    device_settings = forge_runtime.experimental.DeviceSettings()
    device_settings.enable_program_cache = True
    forge_runtime.experimental.configure_devices(device_settings)


def compile_forge_onnx(
    model: Union[forge.OnnxModule, onnx.ModelProto],
    sample_input: torch.Tensor,
    module_name: str,
    optimization_level: int = 2,
    trace_enabled: bool = True,
    data_format: str = "bfloat16",
):
    """Compile an ONNX module for inference with benchmark compiler settings.

    Args:
        model: ``OnnxModule`` or ``ModelProto`` to compile.
        sample_input: Example input tensor for shape/dtype.
        module_name: Module name passed to ``forge.compile``.
        optimization_level: MLIR optimization level.
        trace_enabled: Trace flag for MLIR config.
        data_format: Data format label forwarded to ``get_compiler_cfg``.

    Returns:
        Compiled executable returned by ``forge.compile``.
    """
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    compiled = forge.compile(
        model,
        [sample_input],
        module_name=module_name,
        compiler_cfg=get_compiler_cfg(
            optimization_level=optimization_level,
            trace_enabled=trace_enabled,
            data_format=data_format,
        ),
    )
    configure_device_settings()
    return compiled


def execute_and_measure_fps(
    compiled,
    inputs: list[torch.Tensor],
    loop_count: int,
) -> tuple[list[torch.Tensor], float]:
    """Execute the compiled graph repeatedly and measure total wall time.

    Args:
        compiled: Handle returned by ``forge.compile``.
        inputs: Non-empty list of input tensors; indexed with ``i % len(inputs)``.
        loop_count: Number of forward executions.

    Returns:
        Tuple of (list of first output tensor per iteration, total seconds).
    """
    predictions: list[torch.Tensor] = []
    start_time = time.perf_counter_ns()
    for i in range(loop_count):
        start_iteration_time = time.perf_counter_ns()
        outputs = compiled(inputs[i % len(inputs)])
        predictions.append(outputs[0])
        end_iteration_time = time.perf_counter_ns()
        print(f"Iteration {i} took {(end_iteration_time - start_iteration_time) / 1e6:.04f} ms")
    end_time = time.perf_counter_ns()
    total_time = (end_time - start_time) / 1e9
    print(f"Total time: {total_time:.4f}s for {loop_count} iterations")
    return predictions, total_time


def benchmark_vision_forge_onnx(
    model,
    model_name: str,
    load_inputs_fn: Callable[..., torch.Tensor],
    extract_output_tensor_fn: Callable,
    optimization_level: int,
    trace_enabled: bool,
    batch_size: int,
    loop_count: int,
    warmup_count: Optional[int],
    input_size: tuple,
    data_format: str,
    required_pcc: float,
    training: bool,
) -> dict:
    """Compile ONNX vision model, warm up, time execution, verify PCC vs golden.

    Args:
        model: Eager framework module (same object used for golden forward).
        model_name: Name for compile and reporting.
        load_inputs_fn: Produces batch tensors; called as ``load_inputs_fn(batch_size)``.
        extract_output_tensor_fn: Maps model output to tensor(s); first element used for PCC.
        optimization_level: Compiler optimization level.
        trace_enabled: MLIR trace flag.
        batch_size: Batch size for generated inputs and throughput.
        loop_count: Timed iteration count after warmup.
        warmup_count: Warmup iterations, or ``None`` to use ``min(WARMUP_STEPS, loop_count)``.
        input_size: Shape tuple for reporting only (C×H×W).
        data_format: Format string for compiler and reporting.
        required_pcc: Minimum PCC for ``compare_with_golden`` to pass.
        training: If True, raises ``ValueError`` (unsupported).

    Returns:
        Result dict from ``create_benchmark_result`` plus ``pcc`` and extra ``config`` keys.

    Raises:
        ValueError: If ``training`` is True.
        AssertionError: If device output fails golden PCC check.
    """
    if training:
        raise ValueError("Training mode is not supported")

    inputs = [load_inputs_fn(batch_size) for _ in range(loop_count)]

    golden_output = extract_output_tensor_fn(model(inputs[0]))[0]

    print(f"Compiling ONNX model with Forge ({model_name})...")
    compiled = compile_forge_onnx(
        model,
        inputs[0],
        module_name=model_name,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        data_format=data_format,
    )

    if warmup_count is None:
        warmup_loop_count = min(WARMUP_STEPS, loop_count)
    else:
        warmup_loop_count = min(max(0, warmup_count), loop_count)
    warmup_inputs = inputs[:warmup_loop_count]

    logger.info("Starting warmup...")
    execute_and_measure_fps(
        compiled=compiled,
        inputs=warmup_inputs,
        loop_count=warmup_loop_count,
    )
    logger.info("Warmup completed.")

    logger.info("Starting benchmark...")
    predictions, total_time = execute_and_measure_fps(
        compiled=compiled,
        inputs=inputs,
        loop_count=loop_count,
    )
    logger.info("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time if total_time > 0 else float("inf")

    metadata = get_benchmark_metadata()
    full_model_name = model_name
    model_type = "Vision"
    dataset_name = "Random Data"
    title = model_name

    print_benchmark_results(
        model_title=title,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        batch_size=batch_size,
        data_format=data_format,
        input_size=input_size,
    )

    device_out = predictions[0]
    if not compare_with_golden(
        golden_output,
        device_out,
        pcc=required_pcc,
    ):
        pcc_value = float(calculate_pcc(golden_output, device_out))
        raise AssertionError(
            f"Golden comparison failed (compare_with_golden). PCC={pcc_value:.6f}, required={required_pcc}"
        )

    pcc_value = float(calculate_pcc(golden_output, device_out))
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=-1,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_samples,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_name,
        display_name="",
    )
    result["pcc"] = pcc_value
    result["config"]["training"] = training
    result["config"]["warmup_iterations"] = warmup_loop_count
    return result
