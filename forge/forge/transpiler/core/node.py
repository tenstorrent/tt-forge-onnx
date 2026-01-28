# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Base node class and operation registry for the transpiler IR.

Framework-agnostic - used by all frontends.
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.core.shape_eval import get_shape_eval_meta_for_op, ShapeEvalMeta


class TIRNode(ABC):
    """
    Base class for all Transpiler Intermediate Representation nodes.
    Represents a node in the intermediate representation between ML frameworks
    (e.g., ONNX) and Forge module graphs.
    Framework-agnostic — operations are common across all supported frontends.

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
        forge_op_name: Optional[str] = None,
        src_layer: Optional[str] = None,
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
        self.forge_op_name = forge_op_name
        # Shape-evaluation metadata used by inline shape resolver.
        # Subclasses may override by defining class attribute `shape_eval_meta`.
        class_meta = getattr(self.__class__, "shape_eval_meta", None)
        if class_meta is not None and isinstance(class_meta, ShapeEvalMeta):
            self.shape_eval_meta = class_meta
        else:
            self.shape_eval_meta = get_shape_eval_meta_for_op(self.op_type)

    @property
    def forge_attrs(self) -> Dict[str, Any]:
        """
        Forge-specific attributes for code generation, derived from attrs.

        Computed lazily via convert_attrs_to_forge_attrs so that the virtual
        dispatch to a subclass override happens after the object is fully
        constructed — avoiding the CodeQL warning about calling an overridable
        method from __init__.
        """
        return self.convert_attrs_to_forge_attrs(self.attrs)

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
    def forge_op_function_name(self) -> Optional[str]:
        """
        Get full Forge operation function name.

        Returns:
            Full function name in format "forge.op.{forge_op_name}" if forge_op_name is set,
            otherwise None
        """
        if self.forge_op_name is None:
            return None
        return f"forge.op.{self.forge_op_name}"

    @abstractmethod
    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute the node operation using PyTorch.

        Every concrete TIRNode subclass **must** implement this method.  Failing
        to do so raises ``TypeError`` at class-definition time (not at call time),
        making missing implementations an early, hard error rather than a silent
        runtime failure.

        Args:
            input_tensors: Dictionary mapping input names to PyTorch tensors.

        Returns:
            Dictionary mapping output names to result tensors.
        """

    @staticmethod
    def _has_unknown_dimension(shape: Optional[Tuple]) -> bool:
        if shape is None:
            return True
        for dim in shape:
            if dim is None or isinstance(dim, str):
                return True
            if isinstance(dim, int) and dim < 0:
                return True
        return False

    def infer_output_shapes(self, tensor_shapes: Dict[str, Tuple]) -> Optional[Dict[str, Tuple]]:
        """
        Infer output shapes from known input shapes.

        Default behavior is conservative: if output TensorInfo already has a
        fully-known shape, return it; otherwise return None and let fallback
        paths handle it.  Subclasses override this to provide efficient
        formula-based shape inference without executing the operation.
        """
        inferred = {}
        for out_name, out_info in self.outputs.items():
            out_shape = out_info.shape
            if self._has_unknown_dimension(out_shape):
                return None
            inferred[out_name] = tuple(out_shape)
        return inferred if inferred else None

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

        # Convert input order if needed (e.g., EmbeddingNode has inverse order)
        input_names = self.input_names
        if hasattr(self, "convert_inputs_to_forge_order"):
            input_names = self.convert_inputs_to_forge_order(self.input_names)

        # Build input_shapes and input_dtypes lists matching the converted input_names order
        input_shapes_list = []
        input_dtypes_list = []
        for name in input_names:
            if name in self.inputs:
                info = self.inputs[name]
                input_shapes_list.append(info.shape if info.shape else [])
                input_dtypes_list.append(info.torch_dtype if info.torch_dtype else None)
            else:
                # Fallback: use original order if name not found
                input_shapes_list.append([])
                input_dtypes_list.append(None)

        # Return metadata dictionary matching ForgeWriter's Operation structure
        # This allows code generators to produce consistent Forge module code
        return {
            "function_name": self.forge_op_function_name,
            "node_name": self.name,
            # output_name is first output for backward compatibility with single-output operations
            "output_name": self.output_names[0] if len(self.outputs) > 0 else None,
            "output_names": self.output_names,
            "input_names": input_names,  # Use converted input order
            # Convert None shapes to empty lists for code generation compatibility
            # Note: input_shapes and input_dtypes match input_names order
            "input_shapes": input_shapes_list,
            "input_dtypes": input_dtypes_list,
            "args": self.forge_attrs,
            "src_layer": self.src_layer,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' op_type='{self.op_type}'>"
