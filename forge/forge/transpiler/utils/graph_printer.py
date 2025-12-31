# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Debug and visualization utilities for TIR graphs.

These utilities are framework-agnostic and work with TIRGraph, which is the
core intermediate representation used across all frontends.
"""


def print_tir_graph(tir_graph, title: str = "TIR Graph", detailed: bool = True):
    """
    Print TIR graph structure in a readable format.

    This is a framework-agnostic utility that works with TIRGraph from any frontend.

    Args:
        tir_graph: TIRGraph instance to print
        title: Optional title for the output
        detailed: If True, print detailed node information including attributes
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Graph metadata
    print(f"Graph Name: {tir_graph.name}")
    print(f"Debug Mode: {tir_graph.debug_mode}")
    print(f"Number of Nodes: {len(tir_graph.nodes)}")
    print()

    # Inputs
    print("Inputs:")
    for inp in tir_graph.inputs:
        print(f"  - {inp}")
    print()

    # Outputs
    print("Outputs:")
    for out in tir_graph.outputs:
        print(f"  - {out}")
    print()

    # Parameters and Constants
    if tir_graph.params:
        print("Parameters (Trainable):")
        for name, tensor in tir_graph.params.items():
            print(f"  - {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print()

    if tir_graph.constants:
        print("Constants (Non-trainable):")
        for name, tensor in tir_graph.constants.items():
            print(f"  - {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print()

    # Nodes in topological order
    print("Nodes (Topological Order):")
    print("-" * 80)
    sorted_nodes = tir_graph.get_topological_sort()

    for i, node in enumerate(sorted_nodes, 1):
        print(f"\n[{i}] {node.name} ({node.op_type})")
        print(f"    Inputs:  {node.input_names}")
        print(f"    Outputs: {node.output_names}")

        if detailed:
            # Print input tensor info
            if hasattr(node, "input_tensors") and node.input_tensors:
                print("    Input Tensor Info:")
                for inp_name, tensor_info in zip(node.input_names, node.input_tensors):
                    shape_str = tuple(tensor_info.shape) if tensor_info.shape else "unknown"
                    print(f"      {inp_name}: shape={shape_str}, dtype={tensor_info.torch_dtype}")

            # Print output tensor info
            if hasattr(node, "output_tensors") and node.output_tensors:
                print("    Output Tensor Info:")
                for out_name, tensor_info in zip(node.output_names, node.output_tensors):
                    shape_str = tuple(tensor_info.shape) if tensor_info.shape else "unknown"
                    print(f"      {out_name}: shape={shape_str}, dtype={tensor_info.torch_dtype}")

            # Print attributes
            if hasattr(node, "attrs") and node.attrs:
                print("    Attributes:")
                for key, value in node.attrs.items():
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"      {key} = {value_str}")

            # Print Forge operation name
            if hasattr(node, "forge_op_name") and node.forge_op_name:
                print(f"    Forge Op: {node.forge_op_function_name}")

    print(f"\n{'='*80}\n")
