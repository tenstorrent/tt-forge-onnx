# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from test.benchmark.options import ForgeBenchmarkOptions


def make_validator_positive_int(option_name):
    """Create a positive integer validator with the option name in error messages."""

    def validate(value):
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError
            return int_value
        except (ValueError, TypeError):
            raise pytest.UsageError(f"Invalid value for {option_name}: '{value}'. Must be a positive integer (> 0).")

    return validate


def pytest_addoption(parser):
    parser.addoption(
        "--output-file",
        action="store",
        default=None,
        help="Path to write benchmark results as JSON. If omitted, no file is written.",
    )
    parser.addoption(
        "--batch-size",
        action="store",
        default=None,
        type=make_validator_positive_int("--batch-size"),
        help="Number of samples per inference call (positive integer). Omit to use each test's default.",
    )
    parser.addoption(
        "--loop-count",
        action="store",
        default=None,
        type=make_validator_positive_int("--loop-count"),
        help="Number of timed iterations after warmup (positive integer). Omit to use each test's default.",
    )
    parser.addoption(
        "--warmup-count",
        action="store",
        default=None,
        type=make_validator_positive_int("--warmup-count"),
        help="Number of warmup iterations before timing begins (positive integer). Omit to use min(32, loop_count).",
    )
    parser.addoption(
        "--data-format",
        action="store",
        default="bfloat16",
        choices=("float32", "bfloat16"),
        help="Data format for model, inputs and compiler config: float32 or bfloat16 (default: bfloat16).",
    )
    parser.addoption(
        "--training",
        action="store_true",
        default=False,
        help="Run in training mode. Not supported by current benchmarks; raises an error if set.",
    )


@pytest.fixture
def forge_benchmark_options(request) -> ForgeBenchmarkOptions:
    return ForgeBenchmarkOptions(
        output_file=request.config.getoption("--output-file"),
        batch_size=request.config.getoption("--batch-size"),
        loop_count=request.config.getoption("--loop-count"),
        warmup_count=request.config.getoption("--warmup-count"),
        data_format=request.config.getoption("--data-format"),
        training=bool(request.config.getoption("--training")),
    )
