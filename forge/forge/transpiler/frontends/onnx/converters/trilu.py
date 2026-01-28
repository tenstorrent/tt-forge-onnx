# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Trilu operation converter.

Trilu (since opset 14) extracts the upper or lower triangular part of a 2-D
matrix or a batch of 2-D matrices.  Because there is no native ``forge.op.Trilu``
/ TTIR ``trilu`` op, this converter *decomposes* the operation into existing
TIR nodes at transpilation time.

Decomposition strategy
----------------------
1. **Compute a binary triangular mask** (``float32``, shape ``[N, M]``) at
   transpile time using ``torch.triu`` / ``torch.tril``.
   The mask is stored in ``tir_graph.computed_constants`` so that the code
   generator can persist it to a ``.pt`` file and
   ``process_framework_parameters()`` can load it at runtime.

2. **Dtype alignment (conditional):** If the input dtype differs from
   ``float32``, insert a ``CastNode`` to convert the ``float32`` mask to the
   input dtype before multiplication.  This ensures ``MulNode.eval()`` does not
   fail with a dtype mismatch.

3. **Element-wise multiplication:** A ``MulNode`` multiplies the original input
   tensor with the (possibly cast) mask.  Values inside the triangle are
   preserved (mask=1), values outside are zeroed (mask=0).

   * ``upper=1`` → keeps elements on and **above** the *k*-th diagonal.
   * ``upper=0`` → keeps elements on and **below** the *k*-th diagonal.
   * ``k=0``     → main diagonal (default when the ``k`` input is absent).

Constant resolution for ``k``
-----------------------------
The optional ``k`` input is resolved at compile time using two strategies
(same pattern as :class:`ExpandConverter`):

1. Constant subgraph evaluation via
   :func:`~forge.transpiler.frontends.onnx.utils.constant_value_extractor.resolve_constant_tensor_value`
   — handles the case where ``k`` is produced by a preceding ``Constant`` node
   whose output is already in ``tir_graph.computed_constants``.
2. Direct lookup in TIR graph stores and raw ONNX initializers via
   :func:`~forge.transpiler.frontends.onnx.utils.validation.validate_constant_input`.

If both strategies fail (i.e. ``k`` is a runtime activation), a
``ConversionError`` is raised because the mask cannot be precomputed.

Limitations
-----------
* ``k`` must be a compile-time constant; dynamic ``k`` is not supported.
* The last two input dimensions (N, M) must be concrete integers after ONNX
  shape inference.

Reference: https://onnx.ai/onnx/operators/onnx__Trilu.html
"""
import torch
import onnx
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from onnx import NodeProto
from loguru import logger

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.operations.arithmetic import MulNode
from forge.transpiler.operations.other import CastNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.validation import (
    validate_constant_input,
    ConverterValidationError,
)
from forge.transpiler.frontends.onnx.utils.constant_value_extractor import (
    resolve_constant_tensor_value,
)
from forge.transpiler.utils.exceptions import ConversionError


class TriluConverter(OnnxOpConverter):
    """
    Converter for ONNX Trilu operation (opset 14+).

    Decomposes Trilu into a precomputed triangular mask stored as a computed
    constant, then performs an element-wise multiplication with the input:

    * ``float32`` input → ``[MulNode]``
    * other dtype input → ``[CastNode, MulNode]``

    The ``k`` diagonal-offset input must be a compile-time constant or absent
    (defaults to ``0``).  If ``k`` depends on a runtime value, a
    ``ConversionError`` is raised.
    """

    # ── Public entry point ────────────────────────────────────────────────────

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
        """
        Convert an ONNX Trilu node into TIR nodes.

        Args:
            node_proto:     ONNX node protocol buffer.
            input_tensors:  Mapping from input name to TensorInfo.  The first
                            entry is the required ``input`` tensor; the second,
                            if present, is the optional ``k`` tensor.
            output_tensors: Mapping from output name to TensorInfo.
            attrs:          Extracted node attributes.  Key: ``"upper"`` (int,
                            default ``1``).
            node_index:     Zero-based position of this node in the graph.
            graph_proto:    ONNX graph proto (used for constant resolution).
            opset:          Opset version from the model.  Must be >= 14.
            tir_graph:      Partially-built TIR graph.  Used to:

                            * Resolve the ``k`` constant via TIR stores.
                            * Store the precomputed mask in
                              ``tir_graph.computed_constants``.

        Returns:
            ``[MulNode]``              — when input dtype is ``float32``.
            ``[CastNode, MulNode]``    — when input dtype is not ``float32``.

        Raises:
            ConversionError: If opset < 14, input shape is unknown / rank < 2,
                or ``k`` cannot be resolved at compile time.
        """
        node_name = node_proto.name if node_proto.name else f"Trilu_{node_index}"

        try:
            return cls._convert_impl(
                node_proto,
                input_tensors,
                output_tensors,
                attrs,
                node_index,
                graph_proto,
                opset,
                tir_graph,
                node_name,
            )
        except ConverterValidationError as exc:
            raise ConversionError("Trilu", node_name, str(exc), node_index=node_index) from exc

    # ── Private implementation ────────────────────────────────────────────────

    @classmethod
    def _convert_impl(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto,
        opset: int,
        tir_graph,
        node_name: str,
    ) -> List:
        """Core conversion logic — raises ``ConverterValidationError`` on failure."""

        # ── 1. Opset guard ────────────────────────────────────────────────────
        cls._validate_opset(opset, node_name)

        # ── 2. Extract and validate the primary (data) input ─────────────────
        input_name, input_info, input_shape = cls._validate_primary_input(node_proto, input_tensors, node_name)

        # ── 3. Extract attributes ─────────────────────────────────────────────
        upper = int(attrs.get("upper", 1))

        # ── 4. Resolve k (diagonal offset) ───────────────────────────────────
        k = cls._extract_k_value(node_proto, graph_proto, tir_graph, node_name)
        logger.trace(f"Trilu '{node_name}': upper={upper}, k={k}, " f"input_shape={input_shape}, opset={opset}")

        # ── 5. Compute N, M (concrete last-two dims) ──────────────────────────
        N, M = cls._resolve_matrix_dims(input_shape, node_name)

        # ── 6. Build triangular mask and store as computed constant ───────────
        mask_const_name, mask_const_info = cls._compute_and_store_mask(node_name, N, M, upper, k, tir_graph)

        # ── 7. Determine input dtype ──────────────────────────────────────────
        input_onnx_dtype, input_torch_dtype = cls._resolve_input_dtype(input_info)

        # ── 8. Wire the nodes ─────────────────────────────────────────────────
        current_tensors = OrderedDict(input_tensors)
        current_tensors[mask_const_name] = mask_const_info

        nodes: List = []
        mul_mask_name, current_tensors = cls._maybe_insert_cast(
            node_proto,
            node_name,
            mask_const_name,
            mask_const_info,
            input_onnx_dtype,
            input_torch_dtype,
            current_tensors,
            nodes,
        )

        cls._build_mul_node(
            node_proto,
            node_name,
            input_name,
            mul_mask_name,
            current_tensors,
            output_tensors,
            input_shape,
            input_onnx_dtype,
            nodes,
        )

        return nodes

    # ── Validation helpers ────────────────────────────────────────────────────

    @classmethod
    def _validate_opset(cls, opset: int, node_name: str) -> None:
        """Raise ConverterValidationError when opset < 14."""
        if opset < 14:
            raise ConverterValidationError(
                f"Node '{node_name}': Trilu is available from opset 14 onwards, " f"got opset {opset}."
            )

    @classmethod
    def _validate_primary_input(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        node_name: str,
    ) -> Tuple[str, TensorInfo, Tuple]:
        """
        Validate that the first input exists and has rank >= 2.

        Returns:
            (input_name, input_info, input_shape)

        Raises:
            ConverterValidationError: On any validation failure.
        """
        if len(node_proto.input) < 1:
            raise ConverterValidationError(
                f"Node '{node_name}': Trilu requires at least 1 input tensor, " f"got {len(node_proto.input)}."
            )

        input_name = node_proto.input[0]
        input_info = input_tensors.get(input_name)
        if input_info is None:
            raise ConverterValidationError(
                f"Node '{node_name}': primary input '{input_name}' not found " f"in input_tensors."
            )

        input_shape = input_info.shape
        if input_shape is None or len(input_shape) < 2:
            raise ConverterValidationError(
                f"Node '{node_name}': input '{input_name}' must have rank >= 2, " f"got shape {input_shape!r}."
            )

        return input_name, input_info, input_shape

    # ── k (diagonal offset) resolution ───────────────────────────────────────

    @classmethod
    def _extract_k_value(
        cls,
        node_proto: NodeProto,
        graph_proto,
        tir_graph,
        node_name: str,
    ) -> int:
        """
        Resolve the optional ``k`` input tensor to a Python integer.

        Resolution order (same two-strategy pattern as :class:`ExpandConverter`):

        1. Constant subgraph evaluation via ``resolve_constant_tensor_value``
           — handles ``k`` produced by a preceding ``Constant`` ONNX node whose
           output lives in ``tir_graph.computed_constants``.
        2. Direct lookup in TIR graph stores (params / constants /
           computed_constants) and raw ONNX graph initializers via
           ``validate_constant_input``.

        When the ``k`` input is absent the ONNX-spec default of ``0`` is used.

        Args:
            node_proto:  ONNX node proto.
            graph_proto: ONNX graph proto (for initializer fallback).
            tir_graph:   Partially-built TIR graph.
            node_name:   Node name for error messages.

        Returns:
            Integer diagonal offset (default ``0``).

        Raises:
            ConverterValidationError: If ``k`` is present but cannot be
                resolved at compile time.
        """
        # k is input index 1 (optional)
        if len(node_proto.input) < 2 or not node_proto.input[1]:
            logger.trace(f"Trilu '{node_name}': no k input — using default k=0.")
            return 0

        k_input_name = node_proto.input[1]

        # Strategy 1: constant subgraph evaluation
        if tir_graph is not None and graph_proto is not None:
            resolved, k_values, _ = resolve_constant_tensor_value(k_input_name, tir_graph, graph_proto)
            if resolved and k_values is not None:
                k_val = k_values[0] if isinstance(k_values, list) else int(k_values)
                logger.trace(f"Trilu '{node_name}': resolved k={k_val} via constant subgraph evaluation.")
                return int(k_val)

        # Strategy 2: direct initializer / TIR-graph constant lookup
        is_valid, k_value, error_msg = validate_constant_input(
            node_proto,
            input_index=1,
            graph_proto=graph_proto,
            input_name=k_input_name,
            tir_graph=tir_graph,
        )
        if is_valid and k_value is not None:
            k_val = k_value[0] if isinstance(k_value, list) else k_value
            logger.trace(f"Trilu '{node_name}': resolved k={k_val} via direct constant lookup.")
            return int(k_val)

        raise ConverterValidationError(
            f"Node '{node_name}': the 'k' input '{k_input_name}' must be a "
            f"compile-time constant initializer.  Dynamic 'k' (runtime "
            f"activation) is not supported.  {error_msg or ''}"
        )

    # ── Shape resolution ──────────────────────────────────────────────────────

    @classmethod
    def _resolve_matrix_dims(cls, input_shape: Tuple, node_name: str) -> Tuple[int, int]:
        """
        Extract and validate the last two concrete integer dimensions.

        Args:
            input_shape: Full input shape tuple.
            node_name:   Node name for error messages.

        Returns:
            ``(N, M)`` — the row and column counts.

        Raises:
            ConverterValidationError: If either dimension is not a concrete int.
        """
        N, M = input_shape[-2], input_shape[-1]
        if not isinstance(N, int) or not isinstance(M, int):
            raise ConverterValidationError(
                f"Node '{node_name}': the last two input dimensions must be concrete "
                f"integers for static mask precomputation, got N={N!r}, M={M!r} "
                f"(full shape: {input_shape!r}).  "
                f"Use the OnnxRuntime concrete-shape pre-pass "
                f"(pass module_inputs to transpile()) to resolve symbolic dims."
            )
        return N, M

    # ── Mask computation ──────────────────────────────────────────────────────

    @classmethod
    def _compute_and_store_mask(
        cls,
        node_name: str,
        N: int,
        M: int,
        upper: int,
        k: int,
        tir_graph,
    ) -> Tuple[str, TensorInfo]:
        """
        Compute the binary triangular mask and store it in ``computed_constants``.

        The mask is always computed in ``float32`` so that:

        * A single ``MulNode`` suffices for ``float32`` inputs (no cast needed).
        * A single ``CastNode`` converts the mask for all other dtypes.

        Args:
            node_name: Node name (used as part of the constant key).
            N, M:      Last two dimensions of the input tensor.
            upper:     ``1`` for upper triangular, ``0`` for lower triangular.
            k:         Diagonal offset.
            tir_graph: TIR graph whose ``computed_constants`` dict is updated.

        Returns:
            ``(mask_const_name, mask_const_info)`` — the name under which the
            mask is stored and its corresponding ``TensorInfo``.
        """
        if upper:
            mask = torch.triu(torch.ones(N, M, dtype=torch.float32), diagonal=k)
        else:
            mask = torch.tril(torch.ones(N, M, dtype=torch.float32), diagonal=k)

        mask_const_name = f"{node_name}_trilu_mask"
        if tir_graph is not None:
            tir_graph.computed_constants[mask_const_name] = mask
            logger.trace(
                f"Trilu '{node_name}': stored mask '{mask_const_name}' "
                f"shape={tuple(mask.shape)}, upper={upper}, k={k} "
                f"in tir_graph.computed_constants."
            )

        mask_const_info = TensorInfo(
            name=mask_const_name,
            shape=(N, M),
            onnx_dtype=onnx.TensorProto.FLOAT,
        )
        return mask_const_name, mask_const_info

    # ── Dtype helpers ─────────────────────────────────────────────────────────

    @classmethod
    def _resolve_input_dtype(cls, input_info: TensorInfo) -> Tuple[int, "torch.dtype"]:
        """
        Derive the ONNX dtype and PyTorch dtype from *input_info*.

        Falls back to ``float32`` if the dtype is unknown/undefined.

        Returns:
            ``(onnx_dtype, torch_dtype)``
        """
        onnx_dtype = getattr(input_info, "onnx_dtype", None)
        if not onnx_dtype or onnx_dtype == onnx.TensorProto.UNDEFINED:
            return onnx.TensorProto.FLOAT, torch.float32
        return onnx_dtype, onnx_dtype_to_torch_dtype(onnx_dtype)

    # ── Node builders ─────────────────────────────────────────────────────────

    @classmethod
    def _maybe_insert_cast(
        cls,
        node_proto: NodeProto,
        node_name: str,
        mask_const_name: str,
        mask_const_info: TensorInfo,
        input_onnx_dtype: int,
        input_torch_dtype: "torch.dtype",
        current_tensors: OrderedDict,
        nodes: List,
    ) -> Tuple[str, OrderedDict]:
        """
        Conditionally insert a ``CastNode`` to convert the mask to *input_torch_dtype*.

        The mask is always ``float32``; for non-float32 inputs a ``CastNode``
        is prepended so that ``MulNode.eval()`` receives type-matched operands.

        Args:
            node_proto:         ONNX node proto (for ``build_input_output_dicts``).
            node_name:          Trilu node name.
            mask_const_name:    Name of the float32 mask constant.
            mask_const_info:    TensorInfo for the float32 mask constant.
            input_onnx_dtype:   ONNX dtype of the primary input tensor.
            input_torch_dtype:  PyTorch dtype of the primary input tensor.
            current_tensors:    Mutable dict of all currently available tensors.
            nodes:              List to append the ``CastNode`` to (mutated in-place).

        Returns:
            ``(mul_mask_name, updated_current_tensors)``
            *mul_mask_name* is the name of the mask tensor the ``MulNode`` should
            consume — either the original ``float32`` mask (no cast) or the cast
            output.
        """
        if input_torch_dtype == torch.float32:
            # Mask is already float32 — no cast needed.
            logger.trace(f"Trilu '{node_name}': input is float32, skipping CastNode for mask.")
            return mask_const_name, current_tensors

        cast_out_name = f"{node_name}_trilu_cast_mask"
        cast_out_info = TensorInfo(
            name=cast_out_name,
            shape=mask_const_info.shape,
            onnx_dtype=input_onnx_dtype,
        )
        cast_input_dict, cast_output_dict = build_input_output_dicts(
            node_proto,
            current_tensors,
            {cast_out_name: cast_out_info},
            input_names=[mask_const_name],
            output_names=[cast_out_name],
        )
        nodes.append(
            CastNode.create(
                name=f"{node_name}_cast_mask",
                inputs=cast_input_dict,
                outputs=cast_output_dict,
                dtype=input_torch_dtype,
            )
        )
        current_tensors = OrderedDict(current_tensors)
        current_tensors[cast_out_name] = cast_out_info

        logger.trace(
            f"Trilu '{node_name}': inserted CastNode '{node_name}_cast_mask' "
            f"to convert mask float32 → {input_torch_dtype}."
        )
        return cast_out_name, current_tensors

    @classmethod
    def _build_mul_node(
        cls,
        node_proto: NodeProto,
        node_name: str,
        input_name: str,
        mul_mask_name: str,
        current_tensors: OrderedDict,
        output_tensors: OrderedDict,
        input_shape: Tuple,
        input_onnx_dtype: int,
        nodes: List,
    ) -> None:
        """
        Create the ``MulNode(input, mask)`` and append it to *nodes*.

        The output TensorInfo is set to the same shape and dtype as the input
        tensor when it is absent from *output_tensors* (e.g. when ONNX shape
        inference did not populate it).

        Args:
            node_proto:       ONNX node proto.
            node_name:        Trilu node name.
            input_name:       Name of the primary data tensor.
            mul_mask_name:    Name of the mask tensor (possibly cast).
            current_tensors:  All tensors available at this point.
            output_tensors:   Graph-level output tensor info dict (mutated if needed).
            input_shape:      Shape of the primary data tensor.
            input_onnx_dtype: ONNX dtype of the primary data tensor.
            nodes:            List to append the ``MulNode`` to (mutated in-place).
        """
        output_name = node_proto.output[0]
        if output_name not in output_tensors or output_tensors[output_name].shape is None:
            output_tensors[output_name] = TensorInfo(
                name=output_name,
                shape=input_shape,
                onnx_dtype=input_onnx_dtype,
            )

        mul_input_dict, mul_output_dict = build_input_output_dicts(
            node_proto,
            current_tensors,
            output_tensors,
            input_names=[input_name, mul_mask_name],
            output_names=[output_name],
        )
        nodes.append(
            MulNode.create(
                name=f"{node_name}_mul",
                inputs=mul_input_dict,
                outputs=mul_output_dict,
            )
        )
        logger.trace(
            f"Trilu '{node_name}': created MulNode '{node_name}_mul' "
            f"inputs=['{input_name}', '{mul_mask_name}'] → output='{output_name}'."
        )
