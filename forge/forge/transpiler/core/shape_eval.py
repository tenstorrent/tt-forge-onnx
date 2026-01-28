# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shape-evaluation metadata for TIR operations.

This module defines the framework-agnostic data structures (ShapeEvalMeta,
ShapeDependency) and a pluggable registry so that each frontend can register
its own op-type metadata without creating a dependency from core → frontend.

Registration
------------
Frontends call ``register_shape_eval_meta(mapping)`` at import time.  The ONNX
frontend does this in ``frontends/onnx/operations/op_shape_meta.py``, which is
imported by ``frontends/onnx/__init__.py``.

Unknown op types default to ``DEFAULT_SHAPE_EVAL_META`` (VALUE_DEPENDENT,
fake-exec disallowed) — the safest possible assumption.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict


class ShapeDependency(Enum):
    """How an op's output shape depends on inputs."""

    SHAPE_ONLY = auto()
    VALUE_OF_SHAPE_INPUT = auto()
    VALUE_DEPENDENT = auto()


@dataclass(frozen=True)
class ShapeEvalMeta:
    """Metadata describing shape-evaluation behaviour for a TIR op."""

    dependency: ShapeDependency
    fake_exec_allowed: bool
    has_shape_rule: bool


DEFAULT_SHAPE_EVAL_META = ShapeEvalMeta(
    dependency=ShapeDependency.VALUE_DEPENDENT,
    fake_exec_allowed=False,
    has_shape_rule=False,
)

# Pre-built singletons exposed so frontends can reference them directly.
SHAPE_ONLY = ShapeEvalMeta(
    dependency=ShapeDependency.SHAPE_ONLY,
    fake_exec_allowed=True,
    has_shape_rule=True,
)

VALUE_OF_SHAPE_INPUT = ShapeEvalMeta(
    dependency=ShapeDependency.VALUE_OF_SHAPE_INPUT,
    fake_exec_allowed=True,
    has_shape_rule=False,
)

# ---------------------------------------------------------------------------
# Registry — populated by frontends at import time
# ---------------------------------------------------------------------------

_registry: Dict[str, ShapeEvalMeta] = {}


def register_shape_eval_meta(meta_map: Dict[str, ShapeEvalMeta]) -> None:
    """
    Register op-type → ShapeEvalMeta mappings from a frontend.

    May be called multiple times (e.g. when multiple frontends are loaded).
    Later registrations overwrite earlier ones for the same key.

    Args:
        meta_map: Mapping from TIR op-type string to ShapeEvalMeta.
    """
    _registry.update(meta_map)


def get_shape_eval_meta_for_op(op_type: str) -> ShapeEvalMeta:
    """
    Return the ShapeEvalMeta for *op_type*, or DEFAULT_SHAPE_EVAL_META if
    no frontend has registered metadata for that op type.
    """
    return _registry.get(op_type, DEFAULT_SHAPE_EVAL_META)
