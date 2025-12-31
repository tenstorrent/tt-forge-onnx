# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom exceptions for ONNX frontend transpiler.

This module defines all custom exception classes used by the ONNX transpiler
for better error handling and user experience.
"""
from typing import Dict, List, Any


class UnsupportedOperationError(ValueError):
    """
    Exception raised when unsupported ONNX operations are found during transpilation.

    This exception contains detailed information about all unsupported operations,
    including their types, node names, indices, inputs, and attributes.
    """

    def __init__(self, message: str, unsupported_ops: List[Dict[str, Any]]):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            unsupported_ops: List of dictionaries containing details about unsupported operations
        """
        super().__init__(message)
        self.unsupported_ops = unsupported_ops
        self.unsupported_types = sorted(set([op["op_type"] for op in unsupported_ops]))


class ONNXModelValidationError(ValueError):
    """
    Exception raised when ONNX model validation fails.

    This exception provides detailed information about validation failures,
    including the type of validation error, model metadata, and actionable error messages.
    """

    def __init__(self, message: str, validation_error: Exception = None, model_info: Dict[str, Any] = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            validation_error: The original validation exception (if any)
            model_info: Dictionary containing model metadata (opset, inputs, outputs, etc.)
        """
        super().__init__(message)
        self.validation_error = validation_error
        self.model_info = model_info or {}

    def __str__(self):
        """Return a detailed error message with model information."""
        base_msg = super().__str__()
        if self.model_info:
            info_lines = []
            if "opset" in self.model_info and self.model_info["opset"] is not None:
                info_lines.append(f"  Opset Version: {self.model_info['opset']}")
            if "inputs" in self.model_info:
                info_lines.append(f"  Model Inputs: {self.model_info['inputs']}")
            if "outputs" in self.model_info:
                info_lines.append(f"  Model Outputs: {self.model_info['outputs']}")
            if "nodes" in self.model_info:
                info_lines.append(f"  Total Nodes: {self.model_info['nodes']}")
            if "initializers" in self.model_info:
                info_lines.append(f"  Initializers: {self.model_info['initializers']}")
            if "ir_version" in self.model_info and self.model_info["ir_version"] is not None:
                info_lines.append(f"  IR Version: {self.model_info['ir_version']}")
            if info_lines:
                base_msg += "\n\nModel Information:\n" + "\n".join(info_lines)
        return base_msg
