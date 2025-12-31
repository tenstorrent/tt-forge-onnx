# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reduction operations: ReduceSum, ReduceMean, ReduceMax
"""
import torch
from collections import OrderedDict
from typing import Dict, Union, Tuple

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


class ReduceSumNode(TIRNode):
    """
    PyTorch-like ReduceSum operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Union[int, Tuple[int, ...], None] = None,
        keepdim: bool = False,
    ) -> "ReduceSumNode":
        """Static factory method to create a ReduceSumNode."""
        return ReduceSumNode(
            name=name,
            op_type="ReduceSum",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim": dim, "keepdim": keepdim},
            forge_op_name="ReduceSum",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge ReduceSum takes dim as int (single dimension), not tuple.
        Maps 'keepdim' to 'keep_dim'.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = {}
        if "dim" in attrs:
            dim = attrs["dim"]
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if "keepdim" in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate ReduceSum operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        keepdim = bool(self.attrs.get("keepdim", False))
        return {self.output_names[0]: torch.sum(x, dim=dim, keepdim=keepdim)}


class ReduceMeanNode(TIRNode):
    """
    PyTorch-like ReduceMean operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Union[int, Tuple[int, ...], None] = None,
        keepdim: bool = False,
    ) -> "ReduceMeanNode":
        """Static factory method to create a ReduceMeanNode."""
        return ReduceMeanNode(
            name=name,
            op_type="ReduceMean",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim": dim, "keepdim": keepdim},
            forge_op_name="ReduceAvg",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge ReduceAvg takes dim as int (single dimension), not tuple.
        Maps 'keepdim' to 'keep_dim'.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = {}
        if "dim" in attrs:
            dim = attrs["dim"]
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if "keepdim" in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate ReduceMean operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        keepdim = bool(self.attrs.get("keepdim", False))
        return {self.output_names[0]: torch.mean(x, dim=dim, keepdim=keepdim)}


class ReduceMaxNode(TIRNode):
    """
    PyTorch-like ReduceMax operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dim: Union[int, Tuple[int, ...], None] = None,
        keepdim: bool = False,
    ) -> "ReduceMaxNode":
        """Static factory method to create a ReduceMaxNode."""
        return ReduceMaxNode(
            name=name,
            op_type="ReduceMax",
            inputs=inputs,
            outputs=outputs,
            attrs={"dim": dim, "keepdim": keepdim},
            forge_op_name="ReduceMax",
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch attrs to Forge attrs.

        Forge ReduceMax takes dim as int (single dimension), not tuple.
        Maps 'keepdim' to 'keep_dim'.

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        forge_attrs = {}
        if "dim" in attrs:
            dim = attrs["dim"]
            if isinstance(dim, (tuple, list)):
                forge_attrs["dim"] = dim[0] if len(dim) > 0 else 0
            else:
                forge_attrs["dim"] = dim
        if "keepdim" in attrs:
            forge_attrs["keep_dim"] = attrs["keepdim"]
        return forge_attrs

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate ReduceMax operation using PyTorch.

        Uses torch.amax() which handles all cases including dim=None, dim=int, and dim=tuple.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        dim = self.attrs.get("dim", None)
        keepdim = bool(self.attrs.get("keepdim", False))
        return {self.output_names[0]: torch.amax(x, dim=dim, keepdim=keepdim)}
