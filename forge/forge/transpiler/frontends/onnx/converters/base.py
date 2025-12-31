# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Base converter class for ONNX operations with opset version support.

This module provides the OnnxOpConverter base class that all ONNX operation converters
inherit from. It provides a standardized interface for converters and handles opset
version binding through the get_converter() method.
"""
from typing import Callable, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from typing import Dict, Any
    from onnx import NodeProto
    from forge.transpiler.core.types import TensorInfo
    from forge.transpiler.frontends.onnx.converters.converter_result import ConverterResult


class OnnxOpConverter:
    """
    Base class for ONNX operation converters with opset version support.

    Subclasses must implement a single `convert()` method that receives opset as a parameter:

    ```python
    @classmethod
    def convert(cls, node_proto, input_tensors, output_tensors, attrs,
               node_index, graph_proto, opset: int) -> ConverterResult:
        # Implementation with opset-based branching if needed
        ...
    ```

    The `get_converter` classmethod automatically binds the opset and returns a converter function.
    """

    @classmethod
    def convert(
        cls,
        node_proto: "NodeProto",
        input_tensors: "OrderedDict[str, TensorInfo]",
        output_tensors: "OrderedDict[str, TensorInfo]",
        attrs: "Dict[str, Any]",
        node_index: int,
        graph_proto=None,
        opset: int = 1,
    ) -> "ConverterResult":
        """
        Main conversion method. Subclasses must override this.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes
            node_index: Index of the node in the graph
            graph_proto: ONNX graph protocol buffer (optional)
            opset: Opset version (always provided)

        Returns:
            ConverterResult: Either List[TIRNode] or ConstantResult

        Raises:
            NotImplementedError: If subclass doesn't override this method
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement convert() method. "
            f"This method should handle all opset versions using the opset parameter."
        )

    @classmethod
    def get_converter(cls, opset: int) -> Callable:
        """
        Get converter for given opset version.

        Returns a converter function that calls convert() with the opset bound.
        This ensures opset is always available in the convert() method, allowing
        converters to handle version-specific differences.

        The returned function matches the signature expected by the transpiler engine,
        which calls converters without the opset parameter. The opset is bound here
        and passed to convert() internally.

        Args:
            opset: Opset version from the model

        Returns:
            Converter function that takes (node_proto, input_tensors, output_tensors,
            attrs, node_index, graph_proto) and calls convert() with opset bound
        """

        def converter(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto=None):
            # Bind opset version to convert() method call
            # This allows convert() to access opset without it being in the engine's call signature
            return cls.convert(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto, opset)

        return converter
