# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Framework-specific tensor conversion utilities.

Provides functions to convert tensors to framework-specific formats
for debug mode and validation purposes.

Currently supports: ONNX framework only.
"""
from typing import Any, Optional
import torch
import numpy as np


def convert_to_framework_tensor(tensor: Any, framework: str) -> Any:
    """
    Convert tensor to framework-specific format.

    This function handles conversion from various tensor types (PyTorch, numpy, etc.)
    to the format expected by the target framework for debug/validation purposes.

    Currently supports:
    - "onnx": Converts to numpy arrays (ONNX Runtime expects numpy)

    Args:
        tensor: Input tensor (can be torch.Tensor, numpy.ndarray, etc.)
        framework: Target framework name (must be "onnx")

    Returns:
        Tensor in framework-specific format (numpy array for ONNX)

    Raises:
        ValueError: If framework is not "onnx"
    """
    if framework != "onnx":
        raise ValueError(f"Unsupported framework: {framework}. " f"Currently only 'onnx' framework is supported.")

    # ONNX uses numpy arrays for inference
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif hasattr(tensor, "numpy"):  # Other framework tensors (TensorFlow, PaddlePaddle, etc.)
        return tensor.numpy()
    else:
        return np.array(tensor)


def convert_inputs_to_framework(inputs: dict, framework: str, input_order: Optional[list] = None) -> list:
    """
    Convert a dictionary of input tensors to framework-specific format.

    Args:
        inputs: Dictionary mapping input names to tensors
        framework: Target framework name
        input_order: Optional list specifying the order of input names.
                    If provided, inputs will be converted in this order.
                    If None, uses sorted order of input names.

    Returns:
        List of tensors in framework-specific format (ordered by input_order or sorted keys)
    """
    framework_inputs = []

    # Use provided order if available, otherwise sort
    if input_order is not None:
        input_names = [name for name in input_order if name in inputs]
    else:
        input_names = sorted(inputs.keys())

    for input_name in input_names:
        tensor = inputs[input_name]
        converted = convert_to_framework_tensor(tensor, framework)
        framework_inputs.append(converted)
    return framework_inputs
