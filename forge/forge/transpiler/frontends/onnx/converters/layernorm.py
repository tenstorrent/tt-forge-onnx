# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX LayerNormalization operation converter.

This module provides the converter for ONNX LayerNormalization operations (opset v17),
which normalize input tensors over specified dimensions and apply scale and bias.
"""
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from onnx import NodeProto
import torch
import onnx

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.utils.onnx_graph import torch_dtype_to_onnx_dtype
from forge.transpiler.operations.arithmetic import PowNode
from forge.transpiler.operations.normalization import LayerNormNode
from forge.transpiler.operations.other import FullNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts


class LayerNormalizationConverter(OnnxOpConverter):
    """
    Converter for ONNX LayerNormalization operation (opset v17).

    Converts ONNX LayerNormalization to TIR nodes with two strategies:
    1. Single output (Y only): Maps directly to LayerNormNode (uses forge.op.Layernorm)
    2. Multiple outputs (Y, Mean, InvStdDev): Decomposes into multiple TIR nodes

    This is because Forge's Layernorm op currently supports only single output.
    When Mean and InvStdDev are needed (for training), we decompose the operation.
    """

    @classmethod
    def _create_constant_from_fullnode(
        cls,
        name: str,
        value: float,
        torch_dtype: torch.dtype,
        current_outputs: OrderedDict[str, TensorInfo],
    ) -> Tuple[FullNode, torch.Tensor]:
        """
        Create a FullNode for a constant scalar, execute it to get the tensor value,
        and mark it for conversion to a graph constant.

        This method creates a FullNode, executes it immediately to get the tensor value,
        and adds the TensorInfo to current_outputs. The FullNode is marked with a flag
        indicating it should be converted to a constant in the graph's constants dictionary
        rather than being added as a regular node.

        Args:
            name: Name for the constant tensor
            value: Scalar value to fill the tensor with
            torch_dtype: PyTorch data type for the tensor
            current_outputs: Dictionary to add the output tensor info

        Returns:
            Tuple of (FullNode, tensor_value)
        """
        # Convert torch dtype to ONNX dtype
        onnx_dtype = torch_dtype_to_onnx_dtype(torch_dtype)

        # Create TensorInfo for the scalar constant
        tensor_info = TensorInfo(name=name, shape=(), onnx_dtype=onnx_dtype)
        current_outputs[name] = tensor_info

        # Build OrderedDict for FullNode (no inputs, scalar output)
        input_dict = OrderedDict()
        output_dict = OrderedDict()
        output_dict[name] = tensor_info

        # Create FullNode for the constant
        full_node = FullNode.create(
            name=name,
            inputs=input_dict,
            outputs=output_dict,
            shape=(),  # Scalar (0D tensor)
            fill_value=value,
            dtype=torch_dtype,
        )

        # Execute the FullNode immediately to get the tensor value.
        # Storing it in attrs lets the engine skip a second eval() call.
        tensor_value = full_node.eval({})[name]

        return full_node, tensor_value

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
        Convert ONNX LayerNormalization operation to TIR nodes.

        Strategy:
        - If only Y output: Use LayerNormNode (maps to forge.op.Layernorm)
        - If multiple outputs: Decompose into ReduceMean, Sub, Mul, Sqrt, etc.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary mapping input names to TensorInfo
            output_tensors: Dictionary mapping output names to TensorInfo
            attrs: Extracted attributes
            node_index: Index of node in graph
            graph_proto: ONNX graph protocol buffer (unused)
            opset: Opset version (must be >= 17)

        Returns:
            List containing either:
            - Single LayerNormNode (if only Y output)
            - Multiple TIR nodes (if Mean/InvStdDev outputs are present)

        Future Feature Work Notes:
        ==========================

        1. ReduceMean Multi-Dimension Support:
           Currently, forge.op.ReduceAvg only accepts dim as int (single dimension),
           while TIR ReduceMeanNode accepts dim as Union[int, Tuple[int, ...]].
           Due to this constraint, we create multiple ReduceMeanNode instances in a loop,
           one for each dimension in normalized_axes, chaining them together.
           In the future, when forge.op.ReduceAvg is updated to accept dim as List[int]
           (to align with ttir.mean which accepts dim as array of int), we can simplify
           this to a single ReduceMeanNode call instead of the loop-based approach.

        2. FullNode to Constant Conversion:
           FullNode is used for scalar constants (epsilon). The engine automatically
           converts every FullNode to a graph constant during graph construction, so no
           special flag is required on the node.
        """
        # Validate opset version
        if opset < 17:
            raise ValueError(
                f"LayerNormalization requires opset >= 17, got opset {opset}. "
                f"Node: {node_proto.name or f'LayerNormalization_{node_index}'}"
            )

        # Validate inputs
        if len(node_proto.input) < 2:
            raise ValueError(
                f"LayerNormalization requires at least 2 inputs (X, Scale), "
                f"got {len(node_proto.input)}. Node: {node_proto.name or f'LayerNormalization_{node_index}'}"
            )

        # Extract attributes
        axis = attrs.get("axis", -1)
        epsilon = attrs.get("epsilon", 1e-5)
        node_name = node_proto.name if node_proto.name else f"LayerNormalization_{node_index}"

        # Normalize axis to positive value for consistency
        # Get input shape to determine rank
        x_input_name = node_proto.input[0]
        x_info = input_tensors[x_input_name]
        x_shape = x_info.shape if x_info.shape else None
        if x_shape is None:
            raise ValueError(f"LayerNormalization node '{node_name}': Cannot determine input shape")
        rank = len(x_shape)
        normalized_axis = axis if axis >= 0 else rank + axis

        # Check number of outputs
        num_outputs = len(node_proto.output)

        # If only Y output, use LayerNormNode (maps to forge.op.Layernorm)
        if num_outputs == 1:
            input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

            layer_norm_node = LayerNormNode.create(
                name=node_name,
                inputs=input_dict,
                outputs=output_dict,
                axis=normalized_axis,  # Use normalized axis
                epsilon=epsilon,
            )

            return [layer_norm_node]

        # Multiple outputs: decompose into multiple TIR nodes
        # Import required node classes
        from forge.transpiler.operations.reduction import ReduceMeanNode
        from forge.transpiler.operations.arithmetic import SubNode, MulNode, AddNode
        from forge.transpiler.operations.activation import SqrtNode, ReciprocalNode
        from forge.transpiler.operations.other import IdentityNode
        from forge.transpiler.frontends.onnx.converters.reduction import create_multi_dim_reduction_nodes

        nodes = []

        # Get input/output info
        scale_input_name = node_proto.input[1]
        bias_input_name = node_proto.input[2] if len(node_proto.input) > 2 else None

        scale_info = input_tensors[scale_input_name]
        bias_info = input_tensors[bias_input_name] if bias_input_name else None

        # Create normalized_axes: tuple of dimensions from axis to rank-1
        #
        # Explanation: In ONNX and PyTorch, LayerNorm's 'axis' parameter specifies the STARTING axis
        # for normalization. Normalization is performed over all dimensions from 'axis' to the end.
        # For example, if axis=1 and rank=4, we normalize over dimensions [1, 2, 3].
        # This is why we create normalized_axes = tuple(range(normalized_axis, rank)).
        #
        # Example:
        #   - Input shape: (2, 3, 4, 5), axis=1
        #   - normalized_axis = 1 (after normalization from -3 if needed)
        #   - normalized_axes = (1, 2, 3) -> normalize over dims 1, 2, 3
        #   - Result: mean/variance computed over shape (2, 1, 1, 1) with keepdim=True
        normalized_axes = tuple(range(normalized_axis, rank))

        # Compute output shapes for Mean and InvStdDev
        # Shape: same as input but with normalized axes reduced to 1
        mean_shape = list(x_shape)
        for i in range(normalized_axis, rank):
            mean_shape[i] = 1
        mean_shape = tuple(mean_shape)

        onnx_dtype = getattr(x_info, "onnx_dtype", None)
        if onnx_dtype is None:
            onnx_dtype = onnx.TensorProto.FLOAT

        dtype = onnx_dtype_to_torch_dtype(onnx_dtype)

        # Track intermediate outputs (for chaining operations)
        # Start with input tensors
        current_outputs = OrderedDict()
        current_outputs[x_input_name] = x_info
        current_outputs[scale_input_name] = scale_info
        if bias_info:
            current_outputs[bias_input_name] = bias_info

        # Stage 1: Compute Mean
        # Mean = ReduceMean<axes=normalized_axes>(X)
        # NOTE: Since forge.op.ReduceAvg only accepts dim as int (single dimension),
        # we use create_multi_dim_reduction_nodes to create multiple ReduceMeanNode instances
        # in a loop, one for each dimension. In the future, when forge.op.ReduceAvg accepts
        # dim as List[int], we can simplify this to a single ReduceMeanNode call.
        # Get Mean output name from node_proto.output[1] (should match graph output)
        mean_output_name = node_proto.output[1] if num_outputs > 1 else f"{node_name}_mean"
        # Verify Mean output name matches graph output if graph_proto is available
        if graph_proto is not None and num_outputs > 1 and len(graph_proto.output) > 1:
            graph_mean_output_name = graph_proto.output[1].name
            if mean_output_name != graph_mean_output_name:
                # Use graph output name to ensure consistency
                mean_output_name = graph_mean_output_name
        # Register Mean output in output_tensors so it's recognized as a graph output
        mean_tensor_info = TensorInfo(name=mean_output_name, shape=mean_shape, onnx_dtype=onnx_dtype)
        output_tensors[mean_output_name] = mean_tensor_info
        # Ensure output tensor info exists in current_outputs for the helper function
        current_outputs[mean_output_name] = mean_tensor_info
        mean_nodes, _ = create_multi_dim_reduction_nodes(
            node_name=f"{node_name}_mean",
            reduction_node_class=ReduceMeanNode,
            input_name=x_input_name,
            output_name=mean_output_name,
            dims=normalized_axes,
            keepdim=True,
            current_outputs=current_outputs,
            name_prefix="mean",
        )
        nodes.extend(mean_nodes)

        # Stage 2: Center the input
        # D = Sub(X, Mean)
        d_output_name = f"{node_name}_centered"
        d_output_tensors = {d_output_name: TensorInfo(name=d_output_name, shape=x_shape, onnx_dtype=onnx_dtype)}
        d_input_dict, d_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            d_output_tensors,
            input_names=[x_input_name, mean_output_name],
            output_names=[d_output_name],
            check_output_tensors=True,
        )

        d_node = SubNode.create(
            name=f"{node_name}_centered",
            inputs=d_input_dict,
            outputs=d_output_dict,
        )
        nodes.append(d_node)
        current_outputs.update(d_output_dict)

        # Stage 3: Compute variance
        # DD = Pow(D, 2.0)
        # PowNode is purely binary: both X and Y must be tensor inputs.
        # Create the exponent constant 2.0 as a scalar FullNode (same pattern
        # as epsilon), then wire it as the second input to PowNode.
        exponent_const_name = f"{node_name}_exponent_2"
        exponent_full_node, exponent_tensor_value = cls._create_constant_from_fullnode(
            exponent_const_name, 2.0, dtype, current_outputs
        )
        exponent_full_node.attrs["constant_value"] = exponent_tensor_value
        nodes.append(exponent_full_node)

        dd_output_name = f"{node_name}_squared"
        dd_output_tensors = {dd_output_name: TensorInfo(name=dd_output_name, shape=x_shape, onnx_dtype=onnx_dtype)}
        dd_input_dict, dd_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            dd_output_tensors,
            input_names=[d_output_name, exponent_const_name],
            output_names=[dd_output_name],
            check_output_tensors=True,
        )

        dd_node = PowNode.create(
            name=f"{node_name}_squared",
            inputs=dd_input_dict,
            outputs=dd_output_dict,
        )
        nodes.append(dd_node)
        current_outputs.update(dd_output_dict)

        # Var = ReduceMean<axes=normalized_axes>(DD)
        # NOTE: Since forge.op.ReduceAvg only accepts dim as int (single dimension),
        # we use create_multi_dim_reduction_nodes to create multiple ReduceMeanNode instances
        # in a loop, one for each dimension. In the future, when forge.op.ReduceAvg accepts
        # dim as List[int], we can simplify this to a single ReduceMeanNode call.
        var_output_name = f"{node_name}_var"
        var_tensor_info = TensorInfo(name=var_output_name, shape=mean_shape, onnx_dtype=onnx_dtype)
        # Ensure output tensor info exists in current_outputs for the helper function
        current_outputs[var_output_name] = var_tensor_info
        var_nodes, _ = create_multi_dim_reduction_nodes(
            node_name=f"{node_name}_var",
            reduction_node_class=ReduceMeanNode,
            input_name=dd_output_name,
            output_name=var_output_name,
            dims=normalized_axes,
            keepdim=True,
            current_outputs=current_outputs,
            name_prefix="var",
        )
        nodes.extend(var_nodes)

        # Stage 4: Compute standard deviation with epsilon
        # VarEps = Add(Var, epsilon)
        # Create epsilon constant using FullNode
        # NOTE: We create a FullNode and mark it to be converted to a constant in the graph's
        # constants dictionary. In the future, we will apply constant propagation during
        # optimization passes to automatically convert FullNode to constants.
        epsilon_const_name = f"{node_name}_epsilon"
        epsilon_full_node, epsilon_tensor_value = cls._create_constant_from_fullnode(
            epsilon_const_name, epsilon, dtype, current_outputs
        )

        # Store the tensor value in the FullNode's attrs so the engine can add it to constants
        epsilon_full_node.attrs["constant_value"] = epsilon_tensor_value

        # Add the FullNode to the nodes list - the engine will convert it to a constant
        nodes.append(epsilon_full_node)

        # VarEps = Add(Var, epsilon)
        # Ensure both inputs are available before creating Add node
        if var_output_name not in current_outputs:
            raise ValueError(
                f"Variance output '{var_output_name}' not found in current_outputs. "
                f"Available keys: {list(current_outputs.keys())}"
            )
        if epsilon_const_name not in current_outputs:
            raise ValueError(
                f"Constant '{epsilon_const_name}' not found in current_outputs. "
                f"Available keys: {list(current_outputs.keys())}"
            )
        var_eps_output_name = f"{node_name}_var_eps"
        var_eps_output_tensors = {
            var_eps_output_name: TensorInfo(name=var_eps_output_name, shape=mean_shape, onnx_dtype=onnx_dtype)
        }
        var_eps_input_dict, var_eps_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            var_eps_output_tensors,
            input_names=[var_output_name, epsilon_const_name],
            output_names=[var_eps_output_name],
            check_output_tensors=True,
        )

        # Validate that both inputs were found
        if len(var_eps_input_dict) != 2:
            raise ValueError(
                f"AddNode '{node_name}_var_eps' expected 2 inputs but got {len(var_eps_input_dict)}. "
                f"Input names requested: [{var_output_name}, {epsilon_const_name}]. "
                f"Inputs found: {list(var_eps_input_dict.keys())}. "
                f"Available in current_outputs: {list(current_outputs.keys())}"
            )

        var_eps_node = AddNode.create(
            name=f"{node_name}_var_eps",
            inputs=var_eps_input_dict,
            outputs=var_eps_output_dict,
        )
        nodes.append(var_eps_node)
        current_outputs.update(var_eps_output_dict)

        # StdDev = Sqrt(VarEps)
        std_dev_output_name = f"{node_name}_std_dev"
        std_dev_output_tensors = {
            std_dev_output_name: TensorInfo(name=std_dev_output_name, shape=mean_shape, onnx_dtype=onnx_dtype)
        }
        std_dev_input_dict, std_dev_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            std_dev_output_tensors,
            input_names=[var_eps_output_name],
            output_names=[std_dev_output_name],
            check_output_tensors=True,
        )

        std_dev_node = SqrtNode.create(
            name=f"{node_name}_std_dev",
            inputs=std_dev_input_dict,
            outputs=std_dev_output_dict,
        )
        nodes.append(std_dev_node)
        current_outputs.update(std_dev_output_dict)

        # InvStdDev = Reciprocal(StdDev)
        # Use ReciprocalNode instead of Div(1.0, StdDev) for efficiency
        inv_std_dev_output_name = node_proto.output[2] if num_outputs > 2 else f"{node_name}_inv_std_dev"
        # Register InvStdDev output in output_tensors so it's recognized as a graph output
        inv_std_dev_tensor_info = TensorInfo(name=inv_std_dev_output_name, shape=mean_shape, onnx_dtype=onnx_dtype)
        if num_outputs > 2:
            output_tensors[inv_std_dev_output_name] = inv_std_dev_tensor_info
        inv_std_dev_output_tensors = {inv_std_dev_output_name: inv_std_dev_tensor_info}

        inv_std_dev_input_dict, inv_std_dev_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            inv_std_dev_output_tensors,
            input_names=[std_dev_output_name],
            output_names=[inv_std_dev_output_name],
            check_output_tensors=True,
        )

        inv_std_dev_node = ReciprocalNode.create(
            name=f"{node_name}_inv_std_dev",
            inputs=inv_std_dev_input_dict,
            outputs=inv_std_dev_output_dict,
        )
        nodes.append(inv_std_dev_node)
        current_outputs.update(inv_std_dev_output_dict)

        # Normalized = Mul(D, InvStdDev)
        normalized_output_name = f"{node_name}_normalized"
        normalized_output_tensors = {
            normalized_output_name: TensorInfo(name=normalized_output_name, shape=x_shape, onnx_dtype=onnx_dtype)
        }
        normalized_input_dict, normalized_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            normalized_output_tensors,
            input_names=[d_output_name, inv_std_dev_output_name],
            output_names=[normalized_output_name],
            check_output_tensors=True,
        )

        normalized_node = MulNode.create(
            name=f"{node_name}_normalized",
            inputs=normalized_input_dict,
            outputs=normalized_output_dict,
        )
        nodes.append(normalized_node)
        current_outputs.update(normalized_output_dict)

        # Stage 5: Scale and Shift
        # NormalizedScaled = Mul(Normalized, Scale)
        scaled_output_name = f"{node_name}_scaled"
        scaled_output_tensors = {
            scaled_output_name: TensorInfo(name=scaled_output_name, shape=x_shape, onnx_dtype=onnx_dtype)
        }
        scaled_input_dict, scaled_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            scaled_output_tensors,
            input_names=[normalized_output_name, scale_input_name],
            output_names=[scaled_output_name],
            check_output_tensors=True,
        )

        scaled_node = MulNode.create(
            name=f"{node_name}_scaled",
            inputs=scaled_input_dict,
            outputs=scaled_output_dict,
        )
        nodes.append(scaled_node)
        current_outputs.update(scaled_output_dict)

        # Y = Add(NormalizedScaled, Bias) or just NormalizedScaled if no bias
        y_output_name = node_proto.output[0]
        # Register Y output in output_tensors so it's recognized as a graph output
        y_tensor_info = TensorInfo(name=y_output_name, shape=x_shape, onnx_dtype=onnx_dtype)
        output_tensors[y_output_name] = y_tensor_info
        y_output_tensors = {y_output_name: y_tensor_info}

        if bias_info:
            y_input_dict, y_output_dict = build_input_output_dicts(
                node_proto,
                current_outputs,
                y_output_tensors,
                input_names=[scaled_output_name, bias_input_name],
                output_names=[y_output_name],
                check_output_tensors=True,
            )

            y_node = AddNode.create(
                name=f"{node_name}_output",
                inputs=y_input_dict,
                outputs=y_output_dict,
            )
        else:
            y_input_dict, y_output_dict = build_input_output_dicts(
                node_proto,
                current_outputs,
                y_output_tensors,
                input_names=[scaled_output_name],
                output_names=[y_output_name],
                check_output_tensors=True,
            )

            # Use IdentityNode to pass through
            y_node = IdentityNode.create(
                name=f"{node_name}_output",
                inputs=y_input_dict,
                outputs=y_output_dict,
            )

        nodes.append(y_node)

        return nodes
