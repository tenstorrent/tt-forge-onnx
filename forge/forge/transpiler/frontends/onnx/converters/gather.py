# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Gather operation converter.

Converts ONNX Gather operations to TIR nodes. Gather is lowered to:
- EmbeddingNode: When axis=0 (forge.op.Embedding only supports axis 0)
- IndexSelectNode: When axis != 0 (maps to forge.op.AdvIndex, supports any axis via dim parameter)
- IndexNode: When indices are constant and form a contiguous ascending sequence (optimization)
"""
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from onnx import NodeProto
import torch
import onnx
import numpy as np
from loguru import logger

from forge.transpiler.core.types import TensorInfo, onnx_dtype_to_torch_dtype
from forge.transpiler.frontends.onnx.utils.onnx_graph import torch_dtype_to_onnx_dtype
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.validation import validate_constant_input
from forge.transpiler.operations.indexing import EmbeddingNode, IndexSelectNode, IndexNode
from forge.transpiler.operations.other import CastNode, FullNode, WhereNode, ClipNode
from forge.transpiler.operations.shape import ReshapeNode
from forge.transpiler.operations.arithmetic import AddNode
from forge.transpiler.operations.comparison import LessNode


class GatherConverter(OnnxOpConverter):
    """
    Converter for ONNX Gather operation.

    Supports opset versions 1, 11, and 13. The main difference is bfloat16
    support in opset 13+, but the conversion logic is the same.

    Conversion strategy:
    1. If indices are constant and form contiguous range → IndexNode (uses forge.op.Index)
    2. If axis=0 AND data.ndim==2 → EmbeddingNode (uses forge.op.Embedding)
       - Creates preprocessing nodes: Cast (to int64), Where (normalize negatives), Clip (clamp)
       - EmbeddingNode uses ONLY torch.nn.functional.embedding (requires 2D weight tensor)
    3. Otherwise → IndexSelectNode (uses forge.op.AdvIndex)
       - Covers: axis != 0, OR axis=0 with data.ndim > 2
       - Creates preprocessing nodes: Cast (to int64), Reshape (to 1D/2D), Where, Clip
       - IndexSelectNode uses ONLY torch.index_select
       - Validates AdvIndex constraints: indices must be 1D or 2D (from op_adv_index.cpp)

    All lowering logic is handled in the converter - no separate passes needed.
    Each TIR node contains ONLY one torch operation (matching torch API).

    AdvIndex Constraints (from forge/csrc/ops/op_adv_index.cpp):
    - indices tensor must be 1D or 2D
    - If 2D, AdvIndex will reshape to 1D internally before torch.index_select
    - These constraints are validated in the converter and enforced via ReshapeNode
    """

    @classmethod
    def _create_constant_scalar(
        cls,
        name: str,
        value: float,
        torch_dtype: torch.dtype,
        current_outputs: OrderedDict[str, TensorInfo],
    ) -> Tuple[FullNode, torch.Tensor]:
        """
        Create a FullNode for a constant scalar, execute it to get the tensor value,
        and mark it for conversion to a graph constant.

        Similar to LayerNormConverter._create_constant_from_fullnode.

        Args:
            name: Name for the constant tensor
            value: Scalar value to fill the tensor with
            torch_dtype: PyTorch data type for the tensor
            current_outputs: Dictionary to add the output tensor info

        Returns:
            Tuple of (FullNode, tensor_value)
        """
        onnx_dtype = torch_dtype_to_onnx_dtype(torch_dtype)
        tensor_info = TensorInfo(name=name, shape=(), onnx_dtype=onnx_dtype)
        current_outputs[name] = tensor_info

        input_dict = OrderedDict()
        output_dict = OrderedDict([(name, tensor_info)])

        full_node = FullNode.create(
            name=name,
            inputs=input_dict,
            outputs=output_dict,
            shape=(),
            fill_value=value,
            dtype=torch_dtype,
        )

        tensor_value = full_node.eval({})[name]
        full_node.attrs["constant_value"] = tensor_value

        return full_node, tensor_value

    @classmethod
    def _cast_indices_to_int64(
        cls,
        node_proto: NodeProto,
        node_name: str,
        current_indices_name: str,
        indices_tensor: TensorInfo,
        current_outputs: OrderedDict[str, TensorInfo],
        nodes: List,
        force_cast: bool = False,
        suffix: str = "",
    ) -> str:
        """
        Cast indices to int64 if needed.

        When force_cast=True (e.g. for EmbeddingNode), always add Cast to ensure int64.
        This handles cases where reported dtype says int but actual tensor (e.g. from
        subgraph constants during shape resolution) may be float.

        suffix: Optional suffix for unique names when adding multiple casts (e.g. "_embedding").

        Returns:
            Updated indices name (may be same if already int64/int32 and not force_cast)
        """
        if force_cast or indices_tensor.onnx_dtype not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
            cast_output_name = f"{node_name}_indices_cast{suffix}"
            cast_output = TensorInfo(
                name=cast_output_name,
                shape=indices_tensor.shape,
                onnx_dtype=onnx.TensorProto.INT64,
            )
            cast_input_dict, cast_output_dict = build_input_output_dicts(
                node_proto,
                current_outputs,
                {cast_output_name: cast_output},
                input_names=[current_indices_name],
                output_names=[cast_output_name],
            )
            cast_node = CastNode.create(
                name=f"{node_name}_cast{suffix}",
                inputs=cast_input_dict,
                outputs=cast_output_dict,
                dtype=torch.int64,
            )
            nodes.append(cast_node)
            current_outputs[cast_output_name] = cast_output
            return cast_output_name
        return current_indices_name

    @classmethod
    def _normalize_negative_indices(
        cls,
        node_proto: NodeProto,
        node_name: str,
        current_indices_name: str,
        axis_size: int,
        indices_dtype: torch.dtype,
        current_outputs: OrderedDict[str, TensorInfo],
        nodes: List,
    ) -> str:
        """
        Normalize negative indices using Where: where(indices < 0, indices + axis_size, indices).

        Returns:
            Updated indices name after normalization
        """
        if axis_size <= 0:
            return current_indices_name

        # Create constants
        axis_size_const_name = f"{node_name}_axis_size"
        axis_size_full_node, _ = cls._create_constant_scalar(
            axis_size_const_name, float(axis_size), indices_dtype, current_outputs
        )
        nodes.append(axis_size_full_node)

        zero_const_name = f"{node_name}_zero"
        zero_full_node, _ = cls._create_constant_scalar(zero_const_name, 0.0, indices_dtype, current_outputs)
        nodes.append(zero_full_node)

        # Create condition: indices < 0
        condition_output_name = f"{node_name}_condition"
        condition_output = TensorInfo(
            name=condition_output_name,
            shape=current_outputs[current_indices_name].shape,
            onnx_dtype=onnx.TensorProto.BOOL,
        )
        condition_input_dict, condition_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            {condition_output_name: condition_output},
            input_names=[current_indices_name, zero_const_name],
            output_names=[condition_output_name],
        )
        condition_node = LessNode.create(
            name=f"{node_name}_condition",
            inputs=condition_input_dict,
            outputs=condition_output_dict,
        )
        nodes.append(condition_node)
        current_outputs[condition_output_name] = condition_output

        # Create indices + axis_size
        add_output_name = f"{node_name}_indices_add"
        add_output = TensorInfo(
            name=add_output_name,
            shape=current_outputs[current_indices_name].shape,
            onnx_dtype=current_outputs[current_indices_name].onnx_dtype,
        )
        add_input_dict, add_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            {add_output_name: add_output},
            input_names=[current_indices_name, axis_size_const_name],
            output_names=[add_output_name],
        )
        add_node = AddNode.create(
            name=f"{node_name}_add",
            inputs=add_input_dict,
            outputs=add_output_dict,
        )
        nodes.append(add_node)
        current_outputs[add_output_name] = add_output

        # Where: condition ? (indices + axis_size) : indices
        where_output_name = f"{node_name}_indices_normalized"
        where_output = TensorInfo(
            name=where_output_name,
            shape=current_outputs[current_indices_name].shape,
            onnx_dtype=current_outputs[current_indices_name].onnx_dtype,
        )
        where_input_dict, where_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            {where_output_name: where_output},
            input_names=[condition_output_name, add_output_name, current_indices_name],
            output_names=[where_output_name],
        )
        where_node = WhereNode.create(
            name=f"{node_name}_where",
            inputs=where_input_dict,
            outputs=where_output_dict,
        )
        nodes.append(where_node)
        current_outputs[where_output_name] = where_output

        return where_output_name

    @classmethod
    def _clamp_indices(
        cls,
        node_proto: NodeProto,
        node_name: str,
        current_indices_name: str,
        max_val: int,
        indices_dtype: torch.dtype,
        current_outputs: OrderedDict[str, TensorInfo],
        nodes: List,
    ) -> str:
        """
        Clamp indices to valid range [0, max_val].

        ttir.clamp_scalar has F32Attr min/max attributes, so MLIR requires the
        input and output tensor types to both be float32.  When the index tensor
        is an integer type (int32/int64) the verifier raises:
            'ttir.clamp_scalar' op input and output must have same shape.
        because the data-format pass assigns Float32 to the clip output while
        the upstream integer nodes stay Int32/Int64.

        To fix this we explicitly wrap the clip with type casts:
            int → cast_to_float32 → clip(f32 → f32) → cast_back_to_int

        Returns:
            Updated indices name after clamping (same integer dtype as input)
        """
        if max_val < 0:
            return current_indices_name

        in_tensor = current_outputs[current_indices_name]
        orig_onnx_dtype = in_tensor.onnx_dtype  # INT32 or INT64

        # Determine the torch dtype to cast back to after clip
        if orig_onnx_dtype == onnx.TensorProto.INT64:
            cast_back_torch_dtype = torch.int64
        else:
            cast_back_torch_dtype = torch.int32

        # ------------------------------------------------------------------
        # Step 1: cast int → float32 so clip input/output types both match
        # ------------------------------------------------------------------
        f32_input_name = f"{node_name}_indices_clip_f32"
        f32_input_info = TensorInfo(
            name=f32_input_name,
            shape=in_tensor.shape,
            onnx_dtype=onnx.TensorProto.FLOAT,
        )
        cast_to_f32_in, cast_to_f32_out = build_input_output_dicts(
            node_proto,
            current_outputs,
            {f32_input_name: f32_input_info},
            input_names=[current_indices_name],
            output_names=[f32_input_name],
        )
        cast_to_f32_node = CastNode.create(
            name=f"{node_name}_cast_to_f32",
            inputs=cast_to_f32_in,
            outputs=cast_to_f32_out,
            dtype=torch.float32,
        )
        nodes.append(cast_to_f32_node)
        current_outputs[f32_input_name] = f32_input_info

        # ------------------------------------------------------------------
        # Step 2: clip on float32 → float32 (types match, no MLIR error)
        # ------------------------------------------------------------------
        clip_f32_name = f"{node_name}_indices_clamped_f32"
        clip_f32_info = TensorInfo(
            name=clip_f32_name,
            shape=in_tensor.shape,
            onnx_dtype=onnx.TensorProto.FLOAT,
        )
        clip_in, clip_out = build_input_output_dicts(
            node_proto,
            current_outputs,
            {clip_f32_name: clip_f32_info},
            input_names=[f32_input_name],
            output_names=[clip_f32_name],
        )
        clip_node = ClipNode.create(
            name=f"{node_name}_clip",
            inputs=clip_in,
            outputs=clip_out,
            min_val=0.0,
            max_val=float(max_val),
        )
        nodes.append(clip_node)
        current_outputs[clip_f32_name] = clip_f32_info

        # ------------------------------------------------------------------
        # Step 3: cast float32 → original int type (int32 or int64)
        # ------------------------------------------------------------------
        clip_output_name = f"{node_name}_indices_clamped"
        clip_output_info = TensorInfo(
            name=clip_output_name,
            shape=in_tensor.shape,
            onnx_dtype=orig_onnx_dtype,
        )
        cast_back_in, cast_back_out = build_input_output_dicts(
            node_proto,
            current_outputs,
            {clip_output_name: clip_output_info},
            input_names=[clip_f32_name],
            output_names=[clip_output_name],
        )
        cast_back_node = CastNode.create(
            name=f"{node_name}_cast_from_f32",
            inputs=cast_back_in,
            outputs=cast_back_out,
            dtype=cast_back_torch_dtype,
        )
        nodes.append(cast_back_node)
        current_outputs[clip_output_name] = clip_output_info

        return clip_output_name

    @classmethod
    def _reshape_indices_if_needed(
        cls,
        node_proto: NodeProto,
        node_name: str,
        current_indices_name: str,
        current_outputs: OrderedDict[str, TensorInfo],
        nodes: List,
    ) -> str:
        """
        Reshape indices to 1D if rank > 2 (AdvIndex constraint: indices must be 1D or 2D).

        Returns:
            Updated indices name after reshaping (if needed)
        """
        indices_rank = (
            len(current_outputs[current_indices_name].shape) if current_outputs[current_indices_name].shape else 0
        )
        if indices_rank <= 2:
            return current_indices_name

        # Reshape to 1D
        reshape_output_name = f"{node_name}_indices_reshape"
        total_elements = 1
        for dim_size in current_outputs[current_indices_name].shape:
            total_elements *= dim_size
        reshape_shape = (total_elements,)
        reshape_output = TensorInfo(
            name=reshape_output_name,
            shape=reshape_shape,
            onnx_dtype=current_outputs[current_indices_name].onnx_dtype,
        )
        reshape_input_dict, reshape_output_dict = build_input_output_dicts(
            node_proto,
            current_outputs,
            {reshape_output_name: reshape_output},
            input_names=[current_indices_name],
            output_names=[reshape_output_name],
        )
        reshape_node = ReshapeNode.create(
            name=f"{node_name}_reshape",
            inputs=reshape_input_dict,
            outputs=reshape_output_dict,
            shape=reshape_shape,
        )
        nodes.append(reshape_node)
        current_outputs[reshape_output_name] = reshape_output

        return reshape_output_name

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
        Convert ONNX Gather operation to TIR nodes.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (must include 'axis')
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (1, 11, or 13)

        Returns:
            List containing TIRNodes:
            - IndexNode (if contiguous constant indices)
            - OR multiple nodes: Cast, Where, Clip, EmbeddingNode (if axis=0 and data.ndim==2)
            - OR multiple nodes: Cast, Reshape, Where, Clip, IndexSelectNode (if axis != 0 or data.ndim > 2)

        Raises:
            ValueError: If inputs are invalid or attributes are missing
        """
        # Validate inputs
        if len(node_proto.input) != 2:
            raise ValueError(
                f"Gather node {node_proto.name or f'Gather_{node_index}'}: "
                f"Expected 2 inputs, got {len(node_proto.input)}"
            )

        # Extract inputs
        data_name = node_proto.input[0]
        indices_name = node_proto.input[1]

        if data_name not in input_tensors:
            raise ValueError(
                f"Gather node {node_proto.name or f'Gather_{node_index}'}: " f"Input 'data' '{data_name}' not found"
            )
        if indices_name not in input_tensors:
            raise ValueError(
                f"Gather node {node_proto.name or f'Gather_{node_index}'}: "
                f"Input 'indices' '{indices_name}' not found"
            )

        data_tensor = input_tensors[data_name]
        indices_tensor = input_tensors[indices_name]

        # Extract axis attribute (default: 0)
        axis = attrs.get("axis", 0)

        # Normalize axis to positive value
        data_rank = len(data_tensor.shape) if data_tensor.shape else 0
        if axis < 0:
            axis = data_rank + axis

        # Validate axis
        if data_rank > 0 and (axis < 0 or axis >= data_rank):
            raise ValueError(
                f"Gather node {node_proto.name or f'Gather_{node_index}'}: "
                f"axis={attrs.get('axis', 0)} is out of range [-{data_rank}, {data_rank - 1}]"
            )

        # Determine output shape
        # Output rank = indices_rank + data_rank - 1
        indices_rank = len(indices_tensor.shape) if indices_tensor.shape else 0
        output_rank = indices_rank + data_rank - 1

        if output_rank < 0:
            raise ValueError(
                f"Gather node {node_proto.name or f'Gather_{node_index}'}: " f"Invalid output rank: {output_rank}"
            )

        # Compute output shape: [d₀, ..., dₖ₋₁, i₀, ..., iᵩ₋₁, dₖ₊₁, ..., dᵣ₋₁]
        output_shape = None
        if data_tensor.shape and indices_tensor.shape:
            output_shape = data_tensor.shape[:axis] + indices_tensor.shape + data_tensor.shape[axis + 1 :]

        # Update output tensor info
        if len(output_tensors) > 0:
            output_name = list(output_tensors.keys())[0]
            output_tensors[output_name] = TensorInfo(
                name=output_name,
                shape=output_shape,
                onnx_dtype=data_tensor.onnx_dtype,  # Output dtype matches data dtype
            )

        node_name = node_proto.name or f"Gather_{node_index}"

        # Track intermediate outputs for chaining operations
        current_outputs = OrderedDict()
        current_outputs[data_name] = data_tensor
        current_outputs[indices_name] = indices_tensor

        # Get dtype for constants
        indices_dtype = onnx_dtype_to_torch_dtype(indices_tensor.onnx_dtype)
        if indices_dtype not in (torch.int32, torch.int64):
            indices_dtype = torch.int64  # Default to int64 for indices

        # Decision logic: Choose appropriate Forge operation
        # 1. Check if indices are 1D constant and form a contiguous range → use Index
        #    NOTE: IndexNode (strided slice data[start:stop]) is only semantically
        #    equivalent to ONNX Gather when indices are 1D.
        #    ONNX Gather output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
        #    For 1D indices (N,):   output shape = ... + (N,)   + ... → matches data[start:stop]
        #    For 2D indices (B, N): output shape = ... + (B, N) + ... → IndexNode would give (B*N,) ✗
        # 2. Check if axis=0 AND data.ndim==2 → use Embedding
        #    torch.nn.functional.embedding requires a 2D weight tensor; for >2D data fall through to IndexSelect
        # 3. Otherwise → use IndexSelect (maps to AdvIndex, handles any axis and any data rank)

        # Check if indices tensor is available at compile time.
        #
        # Lookup order (mirrors validate_constant_input):
        #   1. tir_graph.params            – ONNX initializers loaded as weights
        #   2. tir_graph.constants         – ONNX initializers loaded as constants
        #   3. tir_graph.computed_constants – outputs of Constant/ConstantOfShape nodes
        #      created during transpilation (not present in graph_proto.initializer)
        #   4. graph_proto.initializer     – raw ONNX fallback
        indices_is_constant = False
        indices_value = None

        is_valid, indices_array, _ = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto, tir_graph=tir_graph, to_python=False
        )
        if is_valid and indices_array is not None:
            # validate_constant_input with to_python=False returns numpy.ndarray directly
            indices_value = indices_array.astype(np.int64)
            indices_is_constant = True

        # Check if indices form a contiguous ascending sequence (for Index/StridedSlice).
        # The IndexNode optimization is only valid when indices are 1D. For multi-dimensional
        # indices (ndim > 1), ONNX Gather preserves the full indices shape in the output
        # (output.shape = data[:axis] + indices.shape + data[axis+1:]), whereas a strided
        # slice always collapses those dimensions into one. Fall through to EmbeddingNode /
        # IndexSelectNode for 2-D+ constant indices so the batch dimensions are retained.
        use_index = False
        if indices_is_constant and indices_value is not None and indices_value.ndim == 1:
            indices_flat = indices_value.flatten()
            if len(indices_flat) > 0:
                start_val = indices_flat[0]
                # Check if indices are contiguous ascending with stride 1
                is_contiguous = all(start_val + idx == val for idx, val in enumerate(indices_flat))
                if is_contiguous:
                    use_index = True

        nodes = []
        current_indices_name = indices_name

        if use_index:
            # Use Index (strided_slice) for contiguous constant indices
            logger.trace(
                f"Gather node '{node_name}': Converting to IndexNode " f"(contiguous constant indices, axis={axis})"
            )
            start = int(indices_value.flatten()[0])
            stop = int(indices_value.flatten()[-1] + 1)
            stride = 1

            # IndexNode only needs the data tensor — start/stop/axis/stride are attrs.
            # Passing both Gather inputs (data + indices) would make the code-emitter
            # treat the indices tensor as an extra positional arg to forge.op.Index,
            # causing "TypeError: Index() got multiple values for argument 'dim'".
            input_dict, output_dict = build_input_output_dicts(
                node_proto, input_tensors, output_tensors, input_names=[data_name]
            )

            return [
                IndexNode.create(
                    name=node_name,
                    inputs=input_dict,
                    outputs=output_dict,
                    axis=axis,
                    start=start,
                    stop=stop,
                    stride=stride,
                )
            ]

        elif axis == 0 and data_rank == 2:
            # Use Embedding for axis=0 with 2D data (torch.nn.functional.embedding requires 2D weight)
            logger.trace(
                f"Gather node '{node_name}': Converting to EmbeddingNode " f"(axis=0, 2D data) with preprocessing nodes"
            )
            # Preprocess indices: cast, normalize negatives, clamp
            # Force cast to int64 for EmbeddingNode (subgraph evaluation may pass float from constants)
            current_indices_name = cls._cast_indices_to_int64(
                node_proto, node_name, current_indices_name, indices_tensor, current_outputs, nodes, force_cast=True
            )

            vocab_size = data_tensor.shape[0] if data_tensor.shape else 0
            if vocab_size > 0:
                current_indices_name = cls._normalize_negative_indices(
                    node_proto, node_name, current_indices_name, vocab_size, indices_dtype, current_outputs, nodes
                )
                # _clamp_indices wraps with float32 casts internally:
                #   int → cast_to_f32 → clip → cast_back_to_int
                # The output type always matches the input type, so no extra
                # cast is needed after this call.
                current_indices_name = cls._clamp_indices(
                    node_proto, node_name, current_indices_name, vocab_size - 1, indices_dtype, current_outputs, nodes
                )

            # Create EmbeddingNode
            embedding_input_dict, embedding_output_dict = build_input_output_dicts(
                node_proto,
                current_outputs,
                output_tensors,
                input_names=[data_name, current_indices_name],
            )
            nodes.append(
                EmbeddingNode.create(
                    name=node_name,
                    inputs=embedding_input_dict,
                    outputs=embedding_output_dict,
                )
            )

            return nodes

        else:
            # Use IndexSelect (maps to AdvIndex) for:
            # - axis != 0, OR
            # - axis=0 with data.ndim > 2 (EmbeddingNode requires 2D weight tensor)
            logger.trace(
                f"Gather node '{node_name}': Converting to IndexSelectNode "
                f"(axis={axis}, data_rank={data_rank}) with preprocessing nodes"
            )

            # Step 1: Cast indices to int64
            current_indices_name = cls._cast_indices_to_int64(
                node_proto, node_name, current_indices_name, indices_tensor, current_outputs, nodes
            )

            # Step 2: Save original indices shape before flattening.
            # torch.index_select requires a 1D index; multi-dim indices must be
            # flattened here and the output reshaped afterwards.
            original_indices_shape = current_outputs[current_indices_name].shape
            indices_rank = len(original_indices_shape) if original_indices_shape else 0

            # Step 3: Flatten multi-dim indices to 1D
            if indices_rank > 1:
                total_elements = 1
                for d in original_indices_shape:
                    total_elements *= d
                flat_shape = (total_elements,)
                flat_name = f"{node_name}_indices_flat"
                flat_info = TensorInfo(
                    name=flat_name,
                    shape=flat_shape,
                    onnx_dtype=current_outputs[current_indices_name].onnx_dtype,
                )
                flat_in, flat_out = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    {flat_name: flat_info},
                    input_names=[current_indices_name],
                    output_names=[flat_name],
                )
                nodes.append(
                    ReshapeNode.create(
                        name=f"{node_name}_flatten_indices",
                        inputs=flat_in,
                        outputs=flat_out,
                        shape=flat_shape,
                    )
                )
                current_outputs[flat_name] = flat_info
                current_indices_name = flat_name

            # Step 4: Normalize negative indices and clamp to valid range
            axis_size = data_tensor.shape[axis] if data_tensor.shape else 0
            if axis_size > 0:
                current_indices_name = cls._normalize_negative_indices(
                    node_proto, node_name, current_indices_name, axis_size, indices_dtype, current_outputs, nodes
                )
                current_indices_name = cls._clamp_indices(
                    node_proto, node_name, current_indices_name, axis_size - 1, indices_dtype, current_outputs, nodes
                )

            # Step 5: Create IndexSelectNode.
            # For multi-dim original indices, IndexSelectNode uses 1D flattened indices and
            # outputs an intermediate shape; a ReshapeNode then restores the ONNX output shape.
            needs_output_reshape = indices_rank > 1

            if needs_output_reshape:
                # IndexSelectNode output shape with N = product(original_indices_shape):
                #   data[:axis] + (N,) + data[axis+1:]
                total_elements = 1
                for d in original_indices_shape:
                    total_elements *= d
                intermediate_shape = data_tensor.shape[:axis] + (total_elements,) + data_tensor.shape[axis + 1 :]
                intermediate_name = f"{node_name}_idx_sel_out"
                intermediate_info = TensorInfo(
                    name=intermediate_name,
                    shape=intermediate_shape,
                    onnx_dtype=data_tensor.onnx_dtype,
                )
                current_outputs[intermediate_name] = intermediate_info

                idx_sel_in, idx_sel_out = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    {intermediate_name: intermediate_info},
                    input_names=[data_name, current_indices_name],
                    output_names=[intermediate_name],
                )
                nodes.append(
                    IndexSelectNode.create(
                        name=f"{node_name}_idx_sel",
                        inputs=idx_sel_in,
                        outputs=idx_sel_out,
                        dim=axis,
                    )
                )

                # Reshape to ONNX Gather output shape:
                #   data[:axis] + original_indices_shape + data[axis+1:]
                final_shape = data_tensor.shape[:axis] + original_indices_shape + data_tensor.shape[axis + 1 :]
                reshape_in, reshape_out = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    output_tensors,
                    input_names=[intermediate_name],
                )
                nodes.append(
                    ReshapeNode.create(
                        name=node_name,
                        inputs=reshape_in,
                        outputs=reshape_out,
                        shape=final_shape,
                    )
                )
            else:
                # 1D indices: IndexSelectNode output shape is already correct
                index_select_input_dict, index_select_output_dict = build_input_output_dicts(
                    node_proto,
                    current_outputs,
                    output_tensors,
                    input_names=[data_name, current_indices_name],
                )
                nodes.append(
                    IndexSelectNode.create(
                        name=node_name,
                        inputs=index_select_input_dict,
                        outputs=index_select_output_dict,
                        dim=axis,
                    )
                )

            return nodes
