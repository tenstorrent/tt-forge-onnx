# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom exceptions for the transpiler core.

This module defines base exception classes used across all transpiler components
for better error handling and user experience.
"""
from typing import Optional, Dict, Any


class TranspilerError(Exception):
    """
    Base exception for all transpiler errors.

    All custom transpiler exceptions should inherit from this class
    to allow for consistent error handling across the codebase.
    """


class ConversionError(TranspilerError):
    """
    Exception raised when operation conversion fails.

    This exception is raised when a converter fails to convert an operation
    from the frontend format (e.g., ONNX) to TIR nodes.
    """

    def __init__(self, op_type: str, node_name: str, reason: str, node_index: Optional[int] = None):
        """
        Initialize the conversion error.

        Args:
            op_type: Type of the operation that failed to convert
            node_name: Name of the node that failed
            reason: Reason for the conversion failure
            node_index: Optional index of the node in the graph
        """
        self.op_type = op_type
        self.node_name = node_name
        self.reason = reason
        self.node_index = node_index

        error_msg = f"Failed to convert {op_type} node '{node_name}'"
        if node_index is not None:
            error_msg += f" at index {node_index}"
        error_msg += f": {reason}"

        super().__init__(error_msg)


class ValidationError(TranspilerError):
    """
    Exception raised when validation fails.

    This exception is raised when validation checks fail during transpilation
    or graph execution.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        self.details = details or {}
        super().__init__(message)


class DebugValidationError(ValidationError):
    """
    Exception raised when debug mode validation fails.

    This exception is raised when debug mode detects mismatches between
    TIR graph outputs and frontend runtime outputs (e.g., ONNXRuntime).

    This is a critical error that should stop execution in debug mode.
    """

    def __init__(
        self,
        message: str,
        frontend_node_name: Optional[str] = None,
        tir_nodes: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the debug validation error.

        Args:
            message: Human-readable error message
            frontend_node_name: Optional name of the frontend node that failed validation
            tir_nodes: Optional list of TIR node names involved in the validation
            details: Optional dictionary with additional error details
        """
        self.frontend_node_name = frontend_node_name
        self.tir_nodes = tir_nodes or []
        super().__init__(message, details=details)
