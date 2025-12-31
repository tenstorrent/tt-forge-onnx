# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Base node class and operation registry for the transpiler IR.

Framework-agnostic - used by all frontends.
"""
import torch
from typing import Dict, List, Any
from collections import OrderedDict
from forge.transpiler.core.types import TensorInfo


class TIRNode:
    """
    Base class for all Transpiler Intermediate Representation nodes.
    Represents a node in the intermediate representation between ML frameworks
    (e.g., ONNX, PaddlePaddle) and Forge module graphs.
    Framework-agnostic - operations are common across all frontends.

    TIRNodes are created with PyTorch-compatible attributes (attrs).
    The only conversion pipeline is: attrs -> forge_attrs.
    Framework-specific conversions (e.g., ONNX -> PyTorch) happen in the frontend converter.
    """

    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        forge_op_name: str = None,
        src_layer: str = None,
    ):
        """
        Initialize a TIRNode.

        Args:
            name: Unique node name
            op_type: Operation type (e.g., "Conv2d", "Relu", "Add")
            inputs: OrderedDict mapping input names to TensorInfo objects
            outputs: OrderedDict mapping output names to TensorInfo objects
            attrs: PyTorch-compatible attributes dictionary
            forge_op_name: Optional Forge operation name (e.g., "Conv2d", "Add").
                          If None, operation must be decomposed before code generation.
            src_layer: Optional source layer name from original framework for debugging/tracking
        """
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        # Store original output names before sanitization for debug/comparison purposes
        # Output names may be sanitized during graph construction, but we need originals
        # for matching against frontend model outputs
        self.original_outputs = list(outputs.keys())
        self.attrs = attrs
        self.src_layer = src_layer
        # Convert PyTorch-compatible attributes to Forge-specific attributes
        # This allows subclasses to customize attribute transformation (e.g., dim -> axis)
        self.forge_attrs = self.convert_attrs_to_forge_attrs(self.attrs)
        self.forge_op_name = forge_op_name

    def convert_attrs_to_forge_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert PyTorch attributes to Forge-specific attributes for code generation.

        This is the only attribute conversion pipeline in TIRNode.
        Subclasses can override this method to perform attribute transformations.

        Why this conversion exists:
        - PyTorch and Forge may use different attribute names (e.g., 'dim' vs 'axis')
        - Forge may require additional attributes (e.g., 'stable=True' for Softmax)
        - This separation keeps TIR framework-agnostic while allowing Forge-specific codegen

        Args:
            attrs: Dictionary of PyTorch-compatible attributes

        Returns:
            Dictionary of Forge-specific attributes
        """
        return attrs.copy()

    @property
    def input_names(self) -> List[str]:
        """
        Get list of input tensor names.

        Returns:
            List of input tensor names in order
        """
        return list(self.inputs.keys())

    @property
    def output_names(self) -> List[str]:
        """
        Get list of output tensor names.

        Returns:
            List of output tensor names in order
        """
        return list(self.outputs.keys())

    @property
    def input_tensors(self) -> List[TensorInfo]:
        """
        Get input tensor metadata.

        Returns:
            List of TensorInfo objects for inputs in order
        """
        return list(self.inputs.values())

    @property
    def output_tensors(self) -> List[TensorInfo]:
        """
        Get output tensor metadata.

        Returns:
            List of TensorInfo objects for outputs in order
        """
        return list(self.outputs.values())

    @property
    def forge_op_function_name(self) -> str:
        """
        Get full Forge operation function name.

        Returns:
            Full function name in format "forge.op.{forge_op_name}" if forge_op_name is set,
            otherwise None
        """
        if self.forge_op_name is None:
            return None
        return f"forge.op.{self.forge_op_name}"

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute the node operation using PyTorch.

        Args:
            input_tensors: Dictionary mapping input names to PyTorch tensors

        Returns:
            Dictionary mapping output names to result tensors

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement eval")

    def emit(self) -> Dict[str, Any]:
        """
        Generate operation metadata dictionary for code generation.

        Returns a dictionary describing the operation that matches the Operation class structure.
        Used by code generators to produce Forge module code.

        Returns:
            Dictionary with keys:
            - function_name: Forge operation function name (e.g., "forge.op.Conv2d")
            - node_name: Name of the node
            - output_name: Name of the first output tensor (for backward compatibility)
            - output_names: List of all output tensor names
            - input_names: List of input tensor names
            - input_shapes: List of input tensor shapes (empty list if shape is None)
            - input_dtypes: List of input tensor dtypes (None if dtype is unknown)
            - args: Dictionary of Forge-specific operation arguments
            - src_layer: Source layer name from original framework (if available)

        Raises:
            NotImplementedError: If forge_op_name is None (operation has no Forge equivalent)
        """
        # Validate that this operation has a Forge equivalent
        # Operations without Forge equivalents must be decomposed before code generation
        if self.forge_op_name is None:
            raise NotImplementedError(
                f"Operation {self.op_type} (node: {self.name}) has no Forge operation equivalent. "
                f"If this operation has no direct Forge equivalent, it must be decomposed "
                f"using pattern callbacks before code generation."
            )

        # Return metadata dictionary matching ForgeWriter's Operation structure
        # This allows code generators to produce consistent Forge module code
        return {
            "function_name": self.forge_op_function_name,
            "node_name": self.name,
            # output_name is first output for backward compatibility with single-output operations
            "output_name": self.output_names[0] if len(self.outputs) > 0 else None,
            "output_names": self.output_names,
            "input_names": self.input_names,
            # Convert None shapes to empty lists for code generation compatibility
            "input_shapes": [info.shape if info.shape else [] for info in self.inputs.values()],
            "input_dtypes": [info.torch_dtype if info.torch_dtype else None for info in self.inputs.values()],
            "args": self.forge_attrs,
            "src_layer": self.src_layer,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' op_type='{self.op_type}'>"
