# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Split operation converter.

Decomposes ONNX Split into one IndexNode (strided slice) per output tensor.
This avoids a dedicated SplitNode and reuses the existing forge.op.Index
infrastructure.

Decomposition strategy:
    ONNX Split [axis=a, split=[s0, s1, ..., s_{n-1}]]
        input (…, D, …)           (D = input.shape[axis])
    →  IndexNode  axis=a  start=0        stop=s0          → out0
    →  IndexNode  axis=a  start=s0       stop=s0+s1       → out1
    …
    →  IndexNode  axis=a  start=D-s_n-1  stop=D           → out_{n-1}

Opset version matrix (from the official ONNX spec):
    v1   axis: INT (no default)
         split: INTS attribute OR optional second input of type T (float only)
         Type constraints: float (double, float, float16) only
         Shape inference: False

    v2   axis: INT (default 0)
         split: INTS attribute only (second input removed)
         Type constraints: all tensor types (no bfloat16)
         Shape inference: True

    v11  axis: INT (default 0), negative values accepted [-rank, rank-1]
         split: INTS attribute only (same as v2, negative axis added)
         Type constraints: all tensor types (no bfloat16)
         Shape inference: True

    v13  axis: INT (default 0), negative values accepted
         split: moved from attribute → optional second input tensor(int64)
         Equal split if 'split' input is absent (must be evenly divisible)
         Type constraints: all tensor types including bfloat16
         Shape inference: True

    v18  axis: INT (default 0), negative values accepted
         split: optional second input tensor(int64)   ┐ mutually exclusive
         num_outputs: optional INT attribute           ┘ per spec
         If neither provided → equal split (number of graph outputs)
         If not evenly divisible → last chunk is smaller (ceiling division)
         Type constraints: all tensor types including bfloat16
         Shape inference: True
"""
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from onnx import NodeProto
from loguru import logger

from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.indexing import IndexNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.utils.exceptions import ConversionError


class SplitConverter(OnnxOpConverter):
    """
    Converter for ONNX Split — decomposes into one IndexNode per output.

    Split-size resolution per opset:
    - v1      : ``attrs["split"]`` (INTS) first; fallback to second input of type T.
    - v2–v12  : ``attrs["split"]`` (INTS attribute) only.
    - v13+    : optional second input tensor(int64); equal split if absent.
    - v18+    : optional second input tensor(int64) OR ``num_outputs`` INT attribute
                (mutually exclusive per spec); ceiling division for uneven splits.
    - All     : negative ``axis`` is normalised to its positive equivalent.
    - v1–v17  : equal split requires the dimension to be evenly divisible.
    - v18+    : last chunk is allowed to be smaller (ceiling division).
    """

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
        tir_graph=None,
    ) -> List:

        node_name = node_proto.name if node_proto.name else f"Split_{node_index}"
        data_input = node_proto.input[0]
        input_info = input_tensors[data_input]

        # --- Axis (negative axis supported from opset 11; apply for all for robustness) ---
        axis = int(attrs.get("axis", 0))
        rank = len(input_info.shape) if input_info.shape else None
        if rank is not None and axis < 0:
            axis = rank + axis

        dim_size = input_info.shape[axis] if (input_info.shape and axis < len(input_info.shape)) else None
        num_outputs = len(node_proto.output)

        # --- Resolve split sizes based on opset ---
        split_sizes = cls._resolve_split_sizes(
            node_proto=node_proto,
            attrs=attrs,
            opset=opset,
            dim_size=dim_size,
            num_outputs=num_outputs,
            graph_proto=graph_proto,
            tir_graph=tir_graph,
            node_name=node_name,
        )

        # --- Build one IndexNode per output ---
        nodes = []
        start = 0
        # Every IndexNode shares the same data input
        input_dict = OrderedDict({data_input: input_tensors[data_input]})

        for i, output_name in enumerate(node_proto.output):
            chunk_size = split_sizes[i]
            stop = start + chunk_size

            output_dict = OrderedDict({output_name: output_tensors[output_name]})

            # Give each slice node a unique name
            slice_name = f"{node_name}_split_{i}" if num_outputs > 1 else node_name

            nodes.append(
                IndexNode.create(
                    name=slice_name,
                    inputs=input_dict,
                    outputs=output_dict,
                    axis=axis,
                    start=start,
                    stop=stop,
                    stride=1,
                )
            )

            start = stop

        return nodes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_split_sizes(
        cls,
        node_proto: NodeProto,
        attrs: Dict[str, Any],
        opset: int,
        dim_size: Optional[int],
        num_outputs: int,
        graph_proto,
        tir_graph,
        node_name: str,
    ) -> List[int]:
        """
        Determine the size of each output chunk in order.

        Returns a list of length ``num_outputs`` summing to ``dim_size``.
        """
        split_sizes = None

        if opset <= 12:
            # v1-v12: split sizes come from the ``split`` attribute (INTS).
            # v1 also allowed a second input of the same type, but this is
            # extremely rare; we handle it via validate_constant_input as fallback.
            raw = attrs.get("split", None)
            if raw is not None:
                split_sizes = [int(x) for x in raw]
            elif opset == 1 and len(node_proto.input) > 1 and node_proto.input[1]:
                # v1-only: optional second input carrying split sizes
                is_valid, value, _ = validate_constant_input(
                    node_proto, input_index=1, graph_proto=graph_proto, tir_graph=tir_graph
                )
                if is_valid and value is not None:
                    if isinstance(value, (list, tuple)):
                        split_sizes = [int(x) for x in value]
                    else:
                        split_sizes = [int(value)]

        else:
            # v13+: split sizes come from the optional second input tensor (int64).
            # Spec (v13): "Lengths of the parts can be specified using input 'split'."
            # Spec (v18): "Either input 'split' or the attribute 'num_outputs' should be
            #              specified, but not both."
            has_split_input = len(node_proto.input) > 1 and node_proto.input[1]

            # v18 mutual-exclusivity check: warn if both split input and num_outputs are present
            if opset >= 18 and has_split_input and attrs.get("num_outputs") is not None:
                logger.warning(
                    f"Split node '{node_name}' (opset {opset}): both 'split' input and "
                    f"'num_outputs' attribute are present, which is invalid per the ONNX v18 "
                    f"spec. The 'split' input takes precedence; 'num_outputs' is ignored."
                )

            if has_split_input:
                is_valid, value, error_msg = validate_constant_input(
                    node_proto, input_index=1, graph_proto=graph_proto, tir_graph=tir_graph
                )
                if not is_valid:
                    raise ConversionError(
                        node_name,
                        f"Split (opset {opset}): 'split' input is not constant and cannot "
                        f"be resolved at compile time. Dynamic split sizes are not supported. "
                        f"Details: {error_msg}",
                    )
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        split_sizes = [int(x) for x in value]
                    else:
                        split_sizes = [int(value)]

        # v18: num_outputs attribute — only consulted when no split input was resolved.
        # Mutually exclusive with split input per spec; the check above already warned.
        if split_sizes is None and opset >= 18:
            num_outputs_attr = attrs.get("num_outputs", None)
            if num_outputs_attr is not None:
                num_outputs = int(num_outputs_attr)

        # Fall back to equal split when no explicit sizes are given
        if split_sizes is None:
            if dim_size is None:
                raise ConversionError(
                    node_name,
                    "Split: cannot compute equal-split sizes because the input dimension "
                    "along the split axis is unknown. Please provide an explicit 'split' "
                    "input or run shape inference before transpilation.",
                )
            # v18 allows the last chunk to be smaller; all other opsets require
            # an even split. We use ceiling division for v18 and exact for others.
            if opset >= 18:
                # ceil division: first (num_outputs-1) chunks get ceil(D/N), last gets remainder.
                # Clamp to 0 so that when num_outputs > dim_size, trailing chunks are 0-sized
                # rather than negative (ONNX allows empty chunks).
                chunk_size = (dim_size + num_outputs - 1) // num_outputs
                last_size = max(0, dim_size - chunk_size * (num_outputs - 1))
                split_sizes = [chunk_size] * (num_outputs - 1) + [last_size]
            else:
                if dim_size % num_outputs != 0:
                    raise ConversionError(
                        node_name,
                        f"Split: input dimension {dim_size} along the split axis is not evenly "
                        f"divisible by the number of outputs {num_outputs}. Provide an explicit "
                        f"'split' attribute/input to specify unequal chunk sizes.",
                    )
                chunk_size = dim_size // num_outputs
                split_sizes = [chunk_size] * num_outputs

        return split_sizes
