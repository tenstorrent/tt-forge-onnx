# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Debug and visualization utilities for ONNX frontend.

These utilities are useful for debugging and inspecting ONNX models.
ONNX-specific utilities are here, while framework-agnostic TIR graph utilities
are in forge.transpiler.utils.graph_printer.
"""
import onnx
from loguru import logger
from typing import Union


def _format_shape(shape_proto) -> str:
    """Format ONNX shape proto to readable string."""
    if not shape_proto:
        return "unknown"
    dims = []
    for dim in shape_proto.dim:
        which = dim.WhichOneof("value")
        if which == "dim_value":
            dims.append(str(dim.dim_value))
        elif which == "dim_param":
            dims.append(f"'{dim.dim_param}'")
        else:
            dims.append("?")
    return f"({', '.join(dims)})" if dims else "()"


def _get_dtype_name(dtype: int) -> str:
    """Get human-readable dtype name from ONNX TensorProto.DataType."""
    dtype_map = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.BOOL: "bool",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT16: "uint16",
        onnx.TensorProto.INT16: "int16",
        onnx.TensorProto.UINT32: "uint32",
        onnx.TensorProto.UINT64: "uint64",
        onnx.TensorProto.STRING: "string",
    }
    return dtype_map.get(dtype, f"unknown({dtype})")


def print_onnx_model_with_shapes(
    onnx_model: Union[onnx.ModelProto, onnx.GraphProto], title: str = "ONNX Model (with Shape Inference)"
):
    """
    Print ONNX model structure with shape and dtype information for all tensors.

    This is useful for debugging shape inference issues and understanding
    the model structure before transpilation.

    Args:
        onnx_model: ONNX ModelProto to print (should have shape inference run)
        title: Optional title for the output
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

    if isinstance(onnx_model, onnx.ModelProto):
        graph = onnx_model.graph
    else:
        graph = onnx_model

    # Model metadata
    print(f"Model Name: {graph.name}")

    if isinstance(onnx_model, onnx.ModelProto):
        print(f"IR Version: {onnx_model.ir_version}")
        if onnx_model.opset_import:
            print(f"Opset Version: {onnx_model.opset_import[0].version}")
    print()

    # Build value_info map for shape/dtype lookup
    value_info_map = {}
    for vi in graph.value_info:
        value_info_map[vi.name] = vi
    for inp in graph.input:
        value_info_map[inp.name] = inp
    for out in graph.output:
        value_info_map[out.name] = out

    # Inputs
    print("Inputs:")
    for inp in graph.input:
        if inp.type.tensor_type:
            shape_str = _format_shape(inp.type.tensor_type.shape)
            dtype_str = _get_dtype_name(inp.type.tensor_type.elem_type)
            print(f"  - {inp.name}: shape={shape_str}, dtype={dtype_str}")
        else:
            print(f"  - {inp.name}: (non-tensor type)")
    print()

    # Outputs
    print("Outputs:")
    for out in graph.output:
        if out.type.tensor_type:
            shape_str = _format_shape(out.type.tensor_type.shape)
            dtype_str = _get_dtype_name(out.type.tensor_type.elem_type)
            print(f"  - {out.name}: shape={shape_str}, dtype={dtype_str}")
        else:
            print(f"  - {out.name}: (non-tensor type)")
    print()

    # Initializers (constants)
    if graph.initializer:
        print("Initializers (Constants):")
        for init in graph.initializer:
            shape_str = f"({', '.join(str(d) for d in init.dims)})" if init.dims else "()"
            dtype_str = _get_dtype_name(init.data_type)
            print(f"  - {init.name}: shape={shape_str}, dtype={dtype_str}")
        print()

    # Nodes with input/output shapes
    print("Nodes (with Shape/Dtype Information):")
    print("-" * 80)
    for i, node in enumerate(graph.node, 1):
        print(f"\n[{i}] {node.name or f'{node.op_type}_{i}'} ({node.op_type})")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")

        # Print input tensor shapes/dtypes
        if node.input:
            print("    Input Tensor Info:")
            for inp_name in node.input:
                if inp_name in value_info_map:
                    vi = value_info_map[inp_name]
                    if vi.type.tensor_type:
                        shape_str = _format_shape(vi.type.tensor_type.shape)
                        dtype_str = _get_dtype_name(vi.type.tensor_type.elem_type)
                        print(f"      {inp_name}: shape={shape_str}, dtype={dtype_str}")
                    else:
                        print(f"      {inp_name}: (non-tensor type)")
                elif inp_name in [init.name for init in graph.initializer]:
                    # Find initializer
                    for init in graph.initializer:
                        if init.name == inp_name:
                            shape_str = f"({', '.join(str(d) for d in init.dims)})" if init.dims else "()"
                            dtype_str = _get_dtype_name(init.data_type)
                            print(f"      {inp_name}: shape={shape_str}, dtype={dtype_str} (initializer)")
                            break
                else:
                    print(f"      {inp_name}: (shape/dtype unknown)")

        # Print output tensor shapes/dtypes
        if node.output:
            print("    Output Tensor Info:")
            for out_name in node.output:
                if out_name in value_info_map:
                    vi = value_info_map[out_name]
                    if vi.type.tensor_type:
                        shape_str = _format_shape(vi.type.tensor_type.shape)
                        dtype_str = _get_dtype_name(vi.type.tensor_type.elem_type)
                        print(f"      {out_name}: shape={shape_str}, dtype={dtype_str}")
                    else:
                        print(f"      {out_name}: (non-tensor type)")
                else:
                    print(f"      {out_name}: (shape/dtype unknown)")

        # Print attributes
        if node.attribute:
            print("    Attributes:")
            for attr in node.attribute:
                attr_str = str(attr)
                if len(attr_str) > 100:
                    attr_str = attr_str[:100] + "..."
                print(f"      {attr.name} = {attr_str}")

    print(f"\n{'=' * 80}\n")


def print_onnx_model(onnx_model: onnx.ModelProto, title: str = "ONNX Model"):
    """
    Print ONNX model using ONNX's built-in printer.

    Args:
        onnx_model: ONNX ModelProto to print
        title: Optional title for the output
    """
    try:
        import onnx.printer

        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}")
        print(onnx.printer.to_text(onnx_model))
        print(f"{'=' * 80}\n")
    except ImportError:
        logger.warning("onnx.printer not available, falling back to string representation")
        print(f"\n{title}:")
        print(str(onnx_model))
    except Exception as e:
        logger.warning(f"Failed to print ONNX model: {e}")
        print(f"\n{title}:")
        print(f"Model: {onnx_model.graph.name}")
        print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")
        print(f"Nodes: {[node.name for node in onnx_model.graph.node]}")
