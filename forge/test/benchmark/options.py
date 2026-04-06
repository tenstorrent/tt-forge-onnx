# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForgeBenchmarkOptions:
    """Benchmark-related values parsed from pytest CLI (see ``conftest.py``).

    Attributes:
        output_file: Path for JSON results, or ``None`` if ``--output-file`` was not set.
        batch_size: Override from ``--batch-size``, or ``None`` to use test defaults.
        loop_count: Override from ``--loop-count``, or ``None`` to use test defaults.
        warmup_count: Override from ``--warmup-count``, or ``None`` for benchmark logic.
        data_format: ``float32`` or ``bfloat16`` from ``--data-format``.
        training: Whether ``--training`` was passed.
    """

    output_file: Optional[str]
    batch_size: Optional[int]
    loop_count: Optional[int]
    warmup_count: Optional[int]
    data_format: Optional[str]
    training: bool
