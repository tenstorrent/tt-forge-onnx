# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Other operations: Concat, Clip, Cast, Pad, Identity
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo


class ConcatNode(TIRNode):
    @staticmethod
    def create(
        name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo], dim: int
    ) -> "ConcatNode":
        """
        Static factory method to create a ConcatNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
            outputs: OrderedDict mapping output names to TensorInfo
            dim: Dimension to concatenate along (required, no default)
        """
        return ConcatNode(
            name=name, op_type="Concat", inputs=inputs, outputs=outputs, attrs={"dim": dim}, forge_op_name="Concatenate"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Convert PyTorch dim to Forge axis.

        Raises:
            ValueError: If 'dim' is None or not present
        """
        if "dim" not in attrs:
            raise ValueError(f"ConcatNode '{self.name}': 'dim' attribute is required but not found in attrs")
        dim = attrs["dim"]
        if dim is None:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute cannot be None. "
                "ConcatNode is similar to torch.cat() which requires a 'dim' argument."
            )
        if not isinstance(dim, int):
            raise TypeError(f"ConcatNode '{self.name}': 'dim' must be an integer, got {type(dim).__name__}")
        return {"axis": dim}

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Concat operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If 'dim' is None or not present
            TypeError: If 'dim' is not an integer
        """
        inputs = [input_tensors[inp] for inp in self.input_names]
        if "dim" not in self.attrs:
            raise ValueError(f"ConcatNode '{self.name}': 'dim' attribute is required but not found")
        dim = self.attrs["dim"]
        if dim is None:
            raise ValueError(
                f"ConcatNode '{self.name}': 'dim' attribute cannot be None. "
                "ConcatNode is similar to torch.cat() which requires a 'dim' argument."
            )
        if not isinstance(dim, int):
            raise TypeError(f"ConcatNode '{self.name}': 'dim' must be an integer, got {type(dim).__name__}")
        return {self.output_names[0]: torch.cat(inputs, dim=dim)}


class ClipNode(TIRNode):
    """
    PyTorch-like Clip operation.

    Clips tensor values to be within [min_val, max_val] range.
    Similar to torch.clamp().
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        min_val: float = None,
        max_val: float = None,
    ) -> "ClipNode":
        """Static factory method to create a ClipNode."""
        attrs = {}
        if min_val is not None:
            attrs["min"] = min_val
        if max_val is not None:
            attrs["max"] = max_val
        return ClipNode(name=name, op_type="Clip", inputs=inputs, outputs=outputs, attrs=attrs, forge_op_name="Clip")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Clip operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        min_val = self.attrs.get("min", None)
        max_val = self.attrs.get("max", None)
        return {self.output_names[0]: torch.clamp(x, min=min_val, max=max_val)}


class CastNode(TIRNode):
    """
    PyTorch-like Cast operation.

    Casts tensor to a different dtype.
    Similar to torch.Tensor.to(dtype).
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        dtype: Optional[torch.dtype] = None,
    ) -> "CastNode":
        """Static factory method to create a CastNode."""
        attrs = {}
        if dtype is not None:
            attrs["dtype"] = dtype
        return CastNode(name=name, op_type="Cast", inputs=inputs, outputs=outputs, attrs=attrs, forge_op_name="Cast")

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Cast operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        to_dtype = self.attrs.get("dtype", x.dtype)
        return {self.output_names[0]: x.to(dtype=to_dtype)}


class PadNode(TIRNode):
    """
    PyTorch-like Pad operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        pad: Union[List[int], Tuple[int, ...]],
        mode: str = "constant",
        value: float = 0.0,
    ) -> "PadNode":
        """Static factory method to create a PadNode."""
        return PadNode(
            name=name,
            op_type="Pad",
            inputs=inputs,
            outputs=outputs,
            attrs={"pad": pad, "mode": mode, "value": value},
            forge_op_name="Pad",
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Pad operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            TypeError: If pad is not a tuple or list of integers
            ValueError: If mode is not supported
        """
        x = input_tensors[self.input_names[0]]
        pad = self.attrs["pad"]
        mode = self.attrs.get("mode", "constant")
        value = self.attrs.get("value", 0.0)

        if isinstance(pad, list):
            if all(isinstance(p, int) for p in pad):
                pad = tuple(pad)
            else:
                raise TypeError(
                    f"PadNode '{self.name}': pad must be a tuple or list of integers, "
                    f"got list with non-integer elements: {pad}"
                )
        elif isinstance(pad, tuple):
            if not all(isinstance(p, int) for p in pad):
                raise TypeError(f"PadNode '{self.name}': pad tuple must contain only integers, " f"got: {pad}")
        else:
            raise TypeError(
                f"PadNode '{self.name}': pad must be a tuple or list of integers, " f"got {type(pad).__name__}: {pad}"
            )

        supported_modes = {"constant", "reflect", "replicate", "circular"}
        if mode not in supported_modes:
            raise ValueError(
                f"PadNode '{self.name}': unsupported padding mode '{mode}'. " f"Supported modes are: {supported_modes}"
            )

        return {self.output_names[0]: F.pad(x, pad, mode=mode, value=value)}


class IdentityNode(TIRNode):
    """
    Identity operation - returns input tensor unchanged.
    Maps to PyTorch's identity operation (just returns the tensor as-is).
    """

    @staticmethod
    def create(
        name: str, inputs: OrderedDict[str, TensorInfo], outputs: OrderedDict[str, TensorInfo]
    ) -> "IdentityNode":
        """Static factory method to create an IdentityNode."""
        return IdentityNode(
            name=name, op_type="Identity", inputs=inputs, outputs=outputs, attrs={}, forge_op_name="Identity"
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Identity operation using PyTorch.

        Returns the input tensor unchanged (identity function).

        Args:
            input_tensors: Dictionary mapping input names to tensors

        Returns:
            Dictionary mapping output name to result tensor
        """
        x = input_tensors[self.input_names[0]]
        return {self.output_names[0]: x}


class FullNode(TIRNode):
    """
    Full operation - creates a tensor filled with a specified value.
    Maps to PyTorch's torch.full() operation.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        shape: Tuple,
        fill_value: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> "FullNode":
        """
        Static factory method to create a FullNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo (can be empty for constant creation)
            outputs: OrderedDict mapping output names to TensorInfo
            shape: Shape of the tensor to create
            fill_value: Value to fill the tensor with (default: 0.0)
            dtype: Data type of the tensor (if None, inferred from output TensorInfo)
        """
        attrs = {"shape": shape, "fill_value": fill_value}
        if dtype is not None:
            attrs["dtype"] = dtype

        return FullNode(
            name=name,
            op_type="Full",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            forge_op_name=None,  # Full not available in Forge (Constant exists but different API)
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate Full operation using PyTorch.

        Creates a tensor filled with the specified value.

        Args:
            input_tensors: Dictionary mapping input names to tensors (may be empty)

        Returns:
            Dictionary mapping output name to result tensor

        Raises:
            ValueError: If shape is not provided and cannot be inferred
        """
        shape = self.attrs.get("shape", None)
        fill_value = self.attrs.get("fill_value", 0.0)
        dtype = self.attrs.get("dtype", None)

        if shape is None:
            output_info = list(self.outputs.values())[0]
            if output_info and output_info.shape:
                shape = tuple(s if s is not None else 1 for s in output_info.shape)
            else:
                raise ValueError("FullNode requires shape attribute or output shape")

        if dtype is None:
            output_info = list(self.outputs.values())[0]
            if output_info and hasattr(output_info, "onnx_dtype"):
                from forge.transpiler.core.types import onnx_dtype_to_torch_dtype

                dtype = onnx_dtype_to_torch_dtype(output_info.onnx_dtype)
            else:
                dtype = torch.float32

        result = torch.full(shape, fill_value, dtype=dtype)
        return {self.output_names[0]: result}
