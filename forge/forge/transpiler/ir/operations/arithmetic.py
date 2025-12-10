"""
Arithmetic operations: Add, Sub, Mul, Div, MatMul, Gemm
"""
import torch
from typing import Dict, List

from forge.transpiler.ir.nodes import TIRNode
from forge.transpiler.ir.types import TensorInfo


def validate_broadcasting_pytorch_style(shape_a: torch.Size, shape_b: torch.Size, 
                                        dtype_a: torch.dtype, dtype_b: torch.dtype,
                                        op_name: str, tensor_a_name: str, tensor_b_name: str) -> None:
    """
    Validate broadcasting and dtype compatibility following PyTorch style.
    
    This function validates:
    1. Dtype equality: Both tensors must have the same dtype (PyTorch requirement)
    2. Broadcasting compatibility: Shapes must be compatible for broadcasting
    
    Broadcasting rules (PyTorch/NumPy-style):
    - Shapes are compared from right to left
    - Two dimensions are compatible if:
      * They are equal, OR
      * One of them is 1, OR
      * One of them doesn't exist (missing dimension)
    
    Args:
        shape_a: Shape of first tensor
        shape_b: Shape of second tensor
        dtype_a: Dtype of first tensor
        dtype_b: Dtype of second tensor
        op_name: Name of the operation (for error messages)
        tensor_a_name: Name of first tensor (for error messages)
        tensor_b_name: Name of second tensor (for error messages)
        
    Raises:
        ValueError: If dtypes don't match or shapes are not compatible for broadcasting
    """
    # 1. Validate dtype equality (PyTorch requirement)
    if dtype_a != dtype_b:
        raise ValueError(
            f"Type mismatch in {op_name}: "
            f"Input tensors must have the same dtype. "
            f"{tensor_a_name} has dtype {dtype_a}, "
            f"{tensor_b_name} has dtype {dtype_b}. "
            f"PyTorch arithmetic operations require matching dtypes."
        )
    
    # 2. If shapes are equal, no broadcasting needed
    if shape_a == shape_b:
        return
    
    # 3. Validate broadcasting compatibility
    # Convert to tuples for easier manipulation
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)
    
    # Pad shorter shape with 1s on the left (missing dimensions treated as 1)
    max_len = max(len(shape_a_list), len(shape_b_list))
    shape_a_padded = [1] * (max_len - len(shape_a_list)) + shape_a_list
    shape_b_padded = [1] * (max_len - len(shape_b_list)) + shape_b_list
    
    # Check compatibility from right to left
    incompatible_dims = []
    for i in range(max_len - 1, -1, -1):
        dim_a = shape_a_padded[i]
        dim_b = shape_b_padded[i]
        
        # Dimensions are compatible if:
        # 1. They are equal, OR
        # 2. One of them is 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            incompatible_dims.append((i, dim_a, dim_b))
    
    if incompatible_dims:
        dim_info = ", ".join([f"dim {d[0]}: {d[1]} vs {d[2]}" for d in incompatible_dims])
        raise ValueError(
            f"Broadcasting error in {op_name}: "
            f"Shapes {shape_a} ({tensor_a_name}) and {shape_b} ({tensor_b_name}) "
            f"are not compatible for broadcasting. "
            f"Incompatible dimensions: {dim_info}. "
            f"Two dimensions are compatible if they are equal OR one is 1."
        )


class AddNode(TIRNode):
    """
    Addition operation node using PyTorch API.
    
    Performs element-wise addition: output = input1 + input2
    Supports broadcasting automatically via PyTorch.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'AddNode':
        """Static factory method to create an AddNode."""
        return AddNode(
            name=name,
            op_type="Add",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Add"
        )
    
    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate addition operation using PyTorch.
        
        Args:
            input_tensors: Dictionary mapping input names to tensors
            
        Returns:
            Dictionary mapping output name to result tensor
        """
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        
        # Validate dtype equality and broadcasting compatibility (PyTorch style)
        validate_broadcasting_pytorch_style(
            a.shape, b.shape,
            a.dtype, b.dtype,
            self.op_type, self.inputs[0], self.inputs[1]
        )
        
        # Use PyTorch add operation (supports broadcasting)
        return {self.outputs[0]: torch.add(a, b)}



class SubNode(TIRNode):
    """
    Subtraction operation node using PyTorch API.
    
    Performs element-wise subtraction: output = input1 - input2
    Supports broadcasting automatically via PyTorch.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'SubNode':
        """Static factory method to create a SubNode."""
        return SubNode(
            name=name,
            op_type="Sub",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Subtract"
        )
    
    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate subtraction operation using PyTorch.
        
        Args:
            input_tensors: Dictionary mapping input names to tensors
            
        Returns:
            Dictionary mapping output name to result tensor
        """
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        
        # Validate dtype equality and broadcasting compatibility (PyTorch style)
        validate_broadcasting_pytorch_style(
            a.shape, b.shape,
            a.dtype, b.dtype,
            self.op_type, self.inputs[0], self.inputs[1]
        )
        
        # Use PyTorch sub operation (supports broadcasting)
        return {self.outputs[0]: torch.sub(a, b)}



class MulNode(TIRNode):
    """
    Multiplication operation node using PyTorch API.
    
    Performs element-wise multiplication: output = input1 * input2
    Supports broadcasting automatically via PyTorch.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'MulNode':
        """Static factory method to create a MulNode."""
        return MulNode(
            name=name,
            op_type="Mul",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Multiply"
        )
    
    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate multiplication operation using PyTorch.
        
        Args:
            input_tensors: Dictionary mapping input names to tensors
            
        Returns:
            Dictionary mapping output name to result tensor
        """
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        
        # Validate dtype equality and broadcasting compatibility (PyTorch style)
        validate_broadcasting_pytorch_style(
            a.shape, b.shape,
            a.dtype, b.dtype,
            self.op_type, self.inputs[0], self.inputs[1]
        )
        
        # Use PyTorch mul operation (supports broadcasting)
        return {self.outputs[0]: torch.mul(a, b)}



class DivNode(TIRNode):
    """
    Division operation node using PyTorch API.
    
    Performs element-wise division: output = input1 / input2
    Supports broadcasting automatically via PyTorch.
    Note: Division by zero behavior follows PyTorch semantics (inf or NaN).
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'DivNode':
        """Static factory method to create a DivNode."""
        return DivNode(
            name=name,
            op_type="Div",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Divide"
        )
    
    def eval(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluate division operation using PyTorch.
        
        Args:
            input_tensors: Dictionary mapping input names to tensors
            
        Returns:
            Dictionary mapping output name to result tensor
        """
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        
        # Validate dtype equality and broadcasting compatibility (PyTorch style)
        validate_broadcasting_pytorch_style(
            a.shape, b.shape,
            a.dtype, b.dtype,
            self.op_type, self.inputs[0], self.inputs[1]
        )
        
        # Determine rounding mode based on dtype
        # For integer types, use floor division to match ONNX behavior
        # For floating point types, use true division (default)
        is_integer_type = not a.dtype.is_floating_point
        
        if is_integer_type:
            # Use floor division for integer types to match ONNX integer division semantics
            return {self.outputs[0]: torch.div(a, b, rounding_mode="floor")}
        else:
            # Use true division (default) for floating point types
            return {self.outputs[0]: torch.div(a, b)}



class MatMulNode(TIRNode):
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'MatMulNode':
        """Static factory method to create a MatMulNode."""
        return MatMulNode(
            name=name,
            op_type="MatMul",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.Matmul"
        )
    
    def eval(self, input_tensors):
        a = input_tensors[self.inputs[0]]
        b = input_tensors[self.inputs[1]]
        return {self.outputs[0]: torch.matmul(a, b)}


