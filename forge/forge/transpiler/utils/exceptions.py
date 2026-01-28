# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom exceptions for the transpiler.

This is the single canonical module for all transpiler exception classes.
It covers both core infrastructure errors and ONNX frontend–specific errors.

Hierarchy
---------
TranspilerError (base)
├── ConversionError           – converter failed to produce TIR nodes
├── ValidationError           – validation check failed during transpilation
│   └── DebugValidationError  – debug-mode output mismatch detected
├── UnsupportedOperationError – ONNX op has no registered converter
└── ONNXModelValidationError  – ONNX model structure / schema validation failed
"""
from typing import Any, Dict, List, Optional


class TranspilerError(Exception):
    """
    Base exception for all transpiler errors.

    All custom transpiler exceptions should inherit from this class
    to allow consistent error handling across the codebase.
    """


class ConversionError(TranspilerError):
    """
    Raised when an operation converter fails to produce TIR nodes.

    Carries structured metadata (op type, node name, index, reason) so that
    callers can log or display precise diagnostic information.
    """

    def __init__(self, op_type: str, node_name: str, reason: str, node_index: Optional[int] = None):
        """
        Args:
            op_type: ONNX (or other frontend) operation type that failed.
            node_name: Name of the node that failed to convert.
            reason: Human-readable explanation of the failure.
            node_index: Optional position of the node in the graph.
        """
        self.op_type = op_type
        self.node_name = node_name
        self.reason = reason
        self.node_index = node_index

        msg = f"Failed to convert {op_type} node '{node_name}'"
        if node_index is not None:
            msg += f" at index {node_index}"
        msg += f": {reason}"

        super().__init__(msg)


class ValidationError(TranspilerError):
    """
    Raised when a validation check fails during transpilation or graph execution.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Args:
            message: Human-readable error message.
            details: Optional mapping with additional diagnostic key-value pairs.
        """
        self.details = details or {}
        super().__init__(message)


class DebugValidationError(ValidationError):
    """
    Raised when debug-mode output validation detects a mismatch.

    Emitted when the TIR graph's output tensors diverge from the reference
    runtime (e.g., ONNX Runtime) outputs. This is a hard error in debug mode
    that stops execution immediately so the discrepancy can be investigated.
    """

    def __init__(
        self,
        message: str,
        frontend_node_name: Optional[str] = None,
        tir_nodes: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            message: Human-readable error message.
            frontend_node_name: Name of the frontend node that triggered the mismatch.
            tir_nodes: List of TIR node names involved in the validation.
            details: Optional mapping with additional diagnostic key-value pairs.
        """
        self.frontend_node_name = frontend_node_name
        self.tir_nodes = tir_nodes or []
        super().__init__(message, details=details)


# ---------------------------------------------------------------------------
# ONNX frontend exceptions
# ---------------------------------------------------------------------------


class UnsupportedOperationError(TranspilerError, ValueError):
    """
    Raised when the model contains ONNX operations with no registered converter.

    Inherits from both ``TranspilerError`` and ``ValueError`` so that callers
    catching either base class receive this error.
    """

    def __init__(self, message: str, unsupported_ops: List[Dict[str, Any]]):
        """
        Args:
            message: Human-readable summary of the unsupported operations.
            unsupported_ops: List of dicts describing each unsupported op.
                Each dict typically contains ``op_type``, ``node_name``,
                ``node_index``, ``inputs``, and ``attrs``.
        """
        super().__init__(message)
        self.unsupported_ops = unsupported_ops
        self.unsupported_types = sorted({op["op_type"] for op in unsupported_ops})


class ONNXModelValidationError(TranspilerError, ValueError):
    """
    Raised when ONNX model structure or schema validation fails.

    Provides detailed model metadata in the error message to help diagnose the
    root cause. Inherits from both ``TranspilerError`` and ``ValueError`` for
    backward compatibility.
    """

    def __init__(
        self,
        message: str,
        validation_error: Optional[Exception] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            message: Human-readable error message.
            validation_error: The underlying validation exception, if any.
            model_info: Optional dict with model metadata (opset, inputs,
                outputs, nodes, initializers, ir_version, etc.).
        """
        super().__init__(message)
        self.validation_error = validation_error
        self.model_info = model_info or {}

    def __str__(self) -> str:
        """Return a detailed error message including model metadata."""
        base_msg = super().__str__()
        if not self.model_info:
            return base_msg

        info_lines = []
        for key, label in [
            ("opset", "Opset Version"),
            ("ir_version", "IR Version"),
            ("inputs", "Model Inputs"),
            ("outputs", "Model Outputs"),
            ("nodes", "Total Nodes"),
            ("initializers", "Initializers"),
        ]:
            value = self.model_info.get(key)
            if value is not None:
                info_lines.append(f"  {label}: {value}")

        return base_msg + ("\n\nModel Information:\n" + "\n".join(info_lines) if info_lines else "")
