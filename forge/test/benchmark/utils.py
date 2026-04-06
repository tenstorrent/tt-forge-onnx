# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
from datetime import datetime
from typing import Any, Dict

from forge.forge_property_utils import get_device_arch, get_device_count, get_device_type


def get_benchmark_metadata() -> Dict[str, str]:
    return {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "machine_name": socket.gethostname(),
    }


def print_benchmark_results(
    model_title: str,
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    date: str,
    machine_name: str,
    total_time: float,
    total_samples: int,
    samples_per_sec: float,
    batch_size: int,
    data_format: str,
    input_size: tuple,
) -> None:
    """Print a formatted benchmark summary to stdout.

    Args:
        model_title: Short title line for the banner.
        full_model_name: Full model identifier string.
        model_type: Category label (e.g. ``Vision``).
        dataset_name: Dataset or input source label.
        date: Date string for the report.
        machine_name: Hostname string.
        total_time: Wall-clock seconds for the timed loop.
        total_samples: ``batch_size * loop_count`` (or equivalent).
        samples_per_sec: Throughput estimate.
        batch_size: Batch size used in the timed loop.
        data_format: Precision / data format label.
        input_size: Input spatial/tensor shape tuple (e.g. C×H×W).

    Returns:
        None
    """
    print("====================================================================")
    print(f"| {model_title} Benchmark Results:".ljust(67) + "|")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time:.4f}s")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec:.2f}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print("====================================================================")


def create_measurement(
    measurement_name: str,
    value: Any,
    step_name: str,
    iteration: int = 1,
    step_warm_up_num_iterations: int = 0,
    target: float = -1,
    device_power: float = -1.0,
    device_temperature: float = -1.0,
) -> Dict[str, Any]:
    """Build one ``measurements`` entry for ``create_benchmark_result``.

    Args:
        measurement_name: Metric name (e.g. ``total_time``).
        value: Metric value.
        step_name: Step or scope label stored in the record.
        iteration: Iteration index for the record.
        step_warm_up_num_iterations: Warmup iteration count metadata.
        target: Optional target value (-1 if unused).
        device_power: Power sample (-1 if unused).
        device_temperature: Temperature sample (-1 if unused).

    Returns:
        Dict suitable for appending to the result ``measurements`` list.
    """
    return {
        "iteration": iteration,
        "step_name": step_name,
        "step_warm_up_num_iterations": step_warm_up_num_iterations,
        "measurement_name": measurement_name,
        "value": value,
        "target": target,
        "device_power": device_power,
        "device_temperature": device_temperature,
    }


def create_benchmark_result(
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    num_layers: int,
    batch_size: int,
    input_size: tuple,
    loop_count: int,
    data_format: str,
    total_time: float,
    total_samples: int,
    optimization_level: int = 2,
    program_cache_enabled: bool = True,
    trace_enabled: bool = False,
    model_info: str = "",
    display_name: str = "",
) -> Dict[str, Any]:
    """Build the top-level JSON-serializable dict for perf ingestion.

    Args:
        full_model_name: Primary model name field.
        model_type: High-level model category.
        dataset_name: Dataset label.
        num_layers: Layer count metadata (-1 if unknown).
        batch_size: Batch size for the run.
        input_size: Input shape tuple (channels, height, width for vision).
        loop_count: Number of timed iterations.
        data_format: Precision / format string stored as ``precision``.
        total_time: Aggregated wall time (seconds).
        total_samples: Total samples processed in the timed region.
        optimization_level: Compiler optimization level in ``config``.
        program_cache_enabled: Whether program cache is reflected in ``config``.
        trace_enabled: Whether tracing is reflected in ``config``.
        model_info: Extra model string in ``config``.
        display_name: Display string in ``config``.

    Returns:
        Dict with ``model``, ``run_type``, ``config``, ``measurements``, ``device_info``, etc.
    """
    measurements = [
        create_measurement("total_samples", total_samples, full_model_name),
        create_measurement("total_time", total_time, full_model_name),
    ]

    config = {
        "model_size": "small",
        "optimization_level": optimization_level,
        "program_cache_enabled": program_cache_enabled,
        "trace_enabled": trace_enabled,
        "model_info": model_info,
        "display_name": display_name,
        "backend": "tt",
    }

    image_dimension = f"{input_size[0]}x{input_size[1]}x{input_size[2]}"
    run_type = f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}"

    arch = get_device_arch()
    device_count = get_device_count()

    return {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": run_type,
        "config": config,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,
        "output_sequence_length": -1,
        "image_dimension": image_dimension,
        "perf_analysis": False,
        "measurements": measurements,
        "device_info": {
            "device_name": socket.gethostname(),
            "arch": arch,
            "device_count": device_count,
            "device_type": get_device_type(arch, device_count),
        },
    }
