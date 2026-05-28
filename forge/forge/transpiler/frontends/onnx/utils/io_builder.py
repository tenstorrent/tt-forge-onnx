# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX converter utility functions.

These functions are used by ONNX operation converters to build input/output
dictionaries for TIRNode creation.
"""
from onnx import NodeProto
from typing import List, Tuple, Optional
from collections import OrderedDict

from forge.transpiler.core.types import TensorInfo


def build_input_output_dicts(
    node_proto: Optional[NodeProto],
    input_tensors: OrderedDict[str, TensorInfo],
    output_tensors: OrderedDict[str, TensorInfo],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    check_output_tensors: bool = False,
) -> Tuple[OrderedDict[str, TensorInfo], OrderedDict[str, TensorInfo]]:
    """
    Build OrderedDict for inputs and outputs from node_proto and tensor dictionaries.

    This is a common pattern across all converters to build input_dict and output_dict
    for TIRNode creation. This utility function eliminates code duplication.

    Args:
        node_proto: ONNX node protocol buffer (optional, used only if input_names/output_names not provided)
        input_tensors: Dictionary of input tensor information (original ONNX inputs)
        output_tensors: Dictionary of output tensor information (includes intermediate outputs)
        input_names: Optional list of specific input names to use instead of node_proto.input
        output_names: Optional list of specific output names to use instead of node_proto.output
        check_output_tensors: If True, also check output_tensors for inputs (for intermediate outputs)

    Returns:
        Tuple of (input_dict, output_dict) as OrderedDict[str, TensorInfo]

    Raises:
        ValueError: If neither input_names nor node_proto is provided, or if a required tensor is not found
    """
    # Build input_dict
    input_dict = OrderedDict()
    if input_names is not None:
        input_names_to_check = input_names
    elif node_proto is not None:
        input_names_to_check = node_proto.input
    else:
        raise ValueError("Either input_names or node_proto must be provided")

    for input_name in input_names_to_check:
        if input_name in input_tensors:
            input_dict[input_name] = input_tensors[input_name]
        elif check_output_tensors and input_name in output_tensors:
            # For intermediate outputs (e.g., from transpose nodes in GEMM)
            input_dict[input_name] = output_tensors[input_name]
        else:
            raise ValueError(
                f"Cannot find TensorInfo for input '{input_name}'. Expected in input_tensors or output_tensors."
            )

    # Build output_dict
    output_dict = OrderedDict()
    if output_names is not None:
        output_names_to_check = output_names
    elif node_proto is not None:
        output_names_to_check = node_proto.output
    else:
        raise ValueError("Either output_names or node_proto must be provided")

    for output_name in output_names_to_check:
        if output_name in output_tensors:
            output_dict[output_name] = output_tensors[output_name]

    return input_dict, output_dict
