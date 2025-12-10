"""
Normalization operations: BatchNormalization
"""
import torch
from typing import Dict, List

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo



class BatchNormalizationNode(TIRNode):
    """
    PyTorch-like BatchNormalization operation.
    ONNX BatchNorm takes 5 inputs: (X, scale, B, mean, var)
    PyTorch BatchNorm also takes these as inputs (not attributes).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               eps: float = 1e-5,
               momentum: float = 0.9) -> 'BatchNormalizationNode':
        """Static factory method to create a BatchNormalizationNode."""
        return BatchNormalizationNode(
            name=name,
            op_type="BatchNormalization",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={'eps': eps, 'momentum': momentum},
            forge_op_function_name="forge.op.Batchnorm"
        )

    def convert_attrs_to_forge_attrs(self, attrs):
        """Convert PyTorch attrs to Forge attrs. Forge Batchnorm uses epsilon, not eps."""
        forge_attrs = {}
        if 'eps' in attrs:
            forge_attrs['epsilon'] = attrs['eps']
        if 'momentum' in attrs:
            forge_attrs['momentum'] = attrs['momentum']
        return forge_attrs

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        scale = input_tensors[self.inputs[1]]  # weight
        bias = input_tensors[self.inputs[2]]    # bias
        mean = input_tensors[self.inputs[3]]    # running_mean
        var = input_tensors[self.inputs[4]]     # running_var
        
        eps = self.attrs.get('eps', 1e-5)
        # Manual batch norm: (x - mean) / sqrt(var + eps) * scale + bias
        normalized = (x - mean) / torch.sqrt(var + eps)
        result = normalized * scale + bias
        return {self.outputs[0]: result}

