# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Normalization operations: LayerNorm
"""
import torch
from typing import Dict
from collections import OrderedDict

from forge.transpiler.core.node import TIRNode
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.shape_mixins import LayerNormShape


class LayerNormNode(LayerNormShape, TIRNode):
    """
    Layer Normalization operation node.

    Normalizes input over specified dimensions, then applies scale and bias.
    Maps to PyTorch's torch.nn.LayerNorm and Forge's forge.op.Layernorm.
    """

    @staticmethod
    def create(
        name: str,
        inputs: OrderedDict[str, TensorInfo],
        outputs: OrderedDict[str, TensorInfo],
        axis: int = -1,
        epsilon: float = 1e-5,
    ) -> "LayerNormNode":
        """
        Static factory method to create a LayerNormNode.

        Args:
            name: Node name
            inputs: OrderedDict mapping input names to TensorInfo
                - First input: X (input tensor)
                - Second input: Scale (scale tensor)
                - Third input (optional): Bias (bias tensor)
            outputs: OrderedDict mapping output names to TensorInfo
                - First output: Y (normalized tensor)
                - Second output (optional): Mean (saved mean)
                - Third output (optional): InvStdDev (saved inverse std dev)
            axis: First normalization dimension (default: -1)
            epsilon: Small value for numerical stability (default: 1e-5)

        Returns:
            LayerNormNode instance
        """
        attrs = {"axis": axis, "epsilon": epsilon}

        return LayerNormNode(
            name=name,
            op_type="LayerNorm",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            forge_op_name="Layernorm",  # Maps to forge.op.Layernorm
        )

    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate LayerNorm operation using manual computation.

        Args:
            input_tensors: Dictionary mapping input names to tensors
                - X: Input tensor
                - Scale: Scale tensor (must match normalized_shape)
                - Bias (optional): Bias tensor (must match normalized_shape)

        Returns:
            Dictionary mapping output names to result tensors
                - Y: Normalized tensor
                - Mean (optional): Saved mean tensor
                - InvStdDev (optional): Saved inverse standard deviation tensor
        """
        x = input_tensors[self.input_names[0]]
        scale = input_tensors[self.input_names[1]]
        bias = input_tensors[self.input_names[2]] if len(self.input_names) > 2 else None

        axis = self.attrs.get("axis", -1)
        epsilon = self.attrs.get("epsilon", 1e-5)

        # Handle negative axis
        rank = len(x.shape)
        if axis < 0:
            axis = rank + axis

        # Determine normalized axes
        normalized_axes = list(range(axis, rank))

        # Stage 1: Compute Mean
        # Mean = ReduceMean<axes=normalized_axes>(X)
        mean = x.mean(dim=normalized_axes, keepdim=True)

        # Stage 2: Center the input
        # D = Sub(X, Mean)
        d = x - mean

        # Stage 3: Compute variance
        # DD = Mul(D, D)
        dd = d * d
        # Var = ReduceMean<axes=normalized_axes>(DD)
        var = dd.mean(dim=normalized_axes, keepdim=True)

        # Stage 4: Compute standard deviation with epsilon
        # VarEps = Add(Var, epsilon)
        var_eps = var + epsilon
        # StdDev = Sqrt(VarEps)
        std_dev = torch.sqrt(var_eps)

        # InvStdDev = 1.0 / StdDev
        inv_std_dev = 1.0 / std_dev

        # Normalized = Mul(D, InvStdDev)
        normalized = d * inv_std_dev

        # Stage 5: Scale and Shift
        # NormalizedScaled = Mul(Normalized, Scale)
        normalized_scaled = normalized * scale

        # Y = Add(NormalizedScaled, Bias) or just NormalizedScaled if no bias
        if bias is not None:
            y = normalized_scaled + bias
        else:
            y = normalized_scaled

        # Build output dictionary
        outputs = {self.output_names[0]: y}

        # Add optional outputs if requested
        if len(self.output_names) > 1:
            # Mean output (squeeze if scalar)
            outputs[self.output_names[1]] = mean.squeeze() if mean.numel() == 1 else mean

        if len(self.output_names) > 2:
            outputs[self.output_names[2]] = inv_std_dev.squeeze() if inv_std_dev.numel() == 1 else inv_std_dev

        return outputs

    def convert_attrs_to_forge_attrs(self, attrs):
        """
        Map TIR attr names to forge.op.Layernorm parameter names.

        forge.op.Layernorm signature:
            Layernorm(name, operandA, weights, bias, dim=-1, epsilon=1e-5)
        TIR stores the normalization dimension as 'axis'; Forge calls it 'dim'.
        """
        return {
            "dim": attrs["axis"],
            "epsilon": attrs.get("epsilon", 1e-5),
        }
