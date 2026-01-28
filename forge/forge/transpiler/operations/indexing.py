# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Indexing operations: Embedding, IndexSelect (maps to AdvIndex), Index
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape_mixins import EmbeddingShape, IndexSelectShape, IndexShape


class EmbeddingNode(EmbeddingShape, TIRNode):
    """
    Embedding lookup operation node.

    Performs embedding lookup: output = embedding_table[indices]
    Only supports axis=0 (forge.op.Embedding only works on axis 0).

    Uses ONLY torch.nn.functional.embedding in eval() - no other operations.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
    ) -> "EmbeddingNode":
        """
        Create an EmbeddingNode.

        Args:
            name: Node name
            inputs: OrderedDict with 'data' (embedding_table) and 'indices' inputs
            outputs: OrderedDict with single output
        """
        return EmbeddingNode(
            name=name,
            op_type="Embedding",
            inputs=inputs,
            outputs=outputs,
            attrs={},  # No axis needed - embedding always uses axis 0
            forge_op_name="Embedding",
        )

    def convert_inputs_to_forge_order(self, input_names: list) -> list:
        """
        Convert TIR input order to Forge operation input order.

        TIR EmbeddingNode order: (embedding_table, indices)
        Forge forge.op.Embedding order: (indices, embedding_table) - INVERSE

        Args:
            input_names: List of input names in TIR order

        Returns:
            List of input names in Forge operation order
        """
        # Forge.op.Embedding expects: (indices, embedding_table)
        # TIR has: (embedding_table, indices)
        # So we reverse the order
        if len(input_names) == 2:
            return [input_names[1], input_names[0]]  # Swap: indices first, then embedding_table
        return input_names

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate embedding operation using PyTorch.

        **ONLY uses torch.nn.functional.embedding** - no other operations.
        This matches forge.op.Embedding which only supports axis=0.

        Note: Index normalization, type casting, and clamping should be handled
        by separate converter nodes before this operation.

        Args:
            input_tensors: Dictionary mapping input names to tensors
            - input_tensors[self.input_names[0]]: embedding_table (data)
            - input_tensors[self.input_names[1]]: indices (must be int64, normalized, clamped)

        Returns:
            Dictionary mapping output name to result tensor
        """
        embedding_table = input_tensors[self.input_names[0]]  # embedding_table (data)
        indices = input_tensors[self.input_names[1]]  # indices (pre-processed)

        # Use ONLY torch.nn.functional.embedding
        # All preprocessing (cast, normalize, clamp) should be done by converter
        result = torch.nn.functional.embedding(indices, embedding_table)

        return {self.output_names[0]: result}


class IndexSelectNode(IndexSelectShape, TIRNode):
    """
    Index select operation node (matches torch.index_select API).

    Performs indexing along a specified axis: output = torch.index_select(data, dim, index).
    Supports any axis via the dim parameter.

    Maps to forge.op.AdvIndex which handles any axis.

    Constraints (from forge.op.AdvIndex):
    - indices must be 1D or 2D
    - If 2D, will be reshaped to 1D before operation
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: int = 0,
    ) -> "IndexSelectNode":
        """
        Create an IndexSelectNode.

        Args:
            name: Node name
            inputs: OrderedDict with 'data' and 'index' inputs
            outputs: OrderedDict with single output
            dim: Dimension along which to index (default: 0)
        """
        return IndexSelectNode(
            name=name,
            op_type="IndexSelect",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim": dim},
            forge_op_name="AdvIndex",  # Maps to forge.op.AdvIndex
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge AdvIndex requires:
        - dim: int (dimension to index along)

        Constraints enforced (from forge/csrc/ops/op_adv_index.cpp):
        - indices must be 1D or 2D (validated in converter, enforced via ReshapeNode)
        - If 2D, AdvIndex will reshape to 1D internally before torch.index_select

        Args:
            attrs: Dictionary containing 'dim'

        Returns:
            Dictionary of Forge-specific attributes with 'dim'
        """
        forge_attrs = {"dim": attrs.get("dim", 0)}
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate index_select operation using PyTorch.

        **ONLY uses torch.index_select** - no other operations.
        This matches forge.op.AdvIndex semantics.

        Note: Index normalization, type casting, reshaping, and clamping should be
        handled by separate converter nodes before this operation.

        Constraints (matching forge.op.AdvIndex):
        - indices must be 1D or 2D
        - If 2D, indices will be reshaped to 1D (handled by converter)

        Args:
            input_tensors: Dictionary mapping input names to tensors
            - input_tensors[self.input_names[0]]: data tensor
            - input_tensors[self.input_names[1]]: index tensor (must be 1D int64, normalized, clamped)

        Returns:
            Dictionary mapping output name to result tensor
        """
        data = input_tensors[self.input_names[0]]
        index = input_tensors[self.input_names[1]]  # Pre-processed: 1D int64

        dim = self.attrs.get("dim", 0)

        # Use ONLY torch.index_select
        # All preprocessing (cast, normalize, clamp, reshape) should be done by converter
        result = torch.index_select(data, dim=dim, index=index)

        return {self.output_names[0]: result}


class IndexNode(IndexShape, TIRNode):
    """
    Index (strided slice) operation node.

    Performs slicing along a specified axis: output = data[start:stop:stride] along dim.
    Used when Gather indices form a contiguous ascending sequence.

    Maps to forge.op.Index (strided_slice).
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        axis: int,
        start: int,
        stop: int,
        stride: int = 1,
    ) -> "IndexNode":
        """
        Create an IndexNode.

        Args:
            name: Node name
            inputs: OrderedDict with single 'data' input
            outputs: OrderedDict with single output
            axis: Dimension along which to slice
            start: Starting index (inclusive)
            stop: Stopping index (exclusive)
            stride: Stride amount (default: 1)
        """
        return IndexNode(
            name=name,
            op_type="Index",
            inputs=inputs,
            outputs=outputs,
            attrs={"axis": axis, "start": start, "stop": stop, "stride": stride},
            forge_op_name="Index",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Map TIR attr names to forge.op.Index parameter names.

        forge.op.Index signature:
            Index(name, operandA, dim, start, stop, stride)
        TIR stores the dimension as 'axis'; Forge calls it 'dim'.
        """
        return {
            "dim": attrs["axis"],
            "start": attrs["start"],
            "stop": attrs["stop"],
            "stride": attrs.get("stride", 1),
        }

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Index operation using PyTorch.

        Uses Python slice notation to perform a strided slice along one axis.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        data = input_tensors[self.input_names[0]]

        axis = self.attrs.get("axis", 0)
        start = self.attrs.get("start", 0)
        stop = self.attrs.get("stop", data.shape[axis])
        stride = self.attrs.get("stride", 1)

        # Normalize negative indices
        axis_size = data.shape[axis]
        if start < 0:
            start = axis_size + start
        if stop < 0:
            stop = axis_size + stop

        # Build an N-dimensional index: select all dims except `axis`
        idx = [slice(None)] * data.ndim
        idx[axis] = slice(start, stop, stride)
        result = data[tuple(idx)]

        return {self.output_names[0]: result}
