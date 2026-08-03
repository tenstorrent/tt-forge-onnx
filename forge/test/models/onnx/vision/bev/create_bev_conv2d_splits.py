# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract each Conv2d operation from block_A_single_cam0.onnx and
block_C_cylinder_backbone.onnx as standalone single-op ONNX models.

Weights and biases are embedded as initializers; the activation tensor
is the sole graph input.  Models are written to:
    BEV_model/split_models/conv2d_splits/

Usage
-----
    python forge/test/models/onnx/vision/bev/create_bev_conv2d_splits.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto


def _collect_shapes(graph: onnx.GraphProto) -> dict:
    shapes: dict[str, list] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            shapes[vi.name] = [d.dim_value for d in t.shape.dim]
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _collect_initializers(graph: onnx.GraphProto) -> dict:
    return {init.name: init for init in graph.initializer}


def _short_sig(ic: int, ih: int, iw: int, oc: int, kh: int, s: int, g: int) -> str:
    return f"ic{ic}_ih{ih}_iw{iw}_oc{oc}_k{kh}_s{s}_g{g}"


def extract_conv2d_models(
    src_path: Path,
    block_tag: str,
    out_dir: Path,
) -> list[dict]:
    """Extract every Conv node in src_path as a standalone ONNX model.

    Returns a list of metadata dicts (one per extracted model).
    """
    model = onnx.load(str(src_path))
    graph = model.graph
    shapes = _collect_shapes(graph)
    inits = _collect_initializers(graph)

    out_dir.mkdir(parents=True, exist_ok=True)

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    records = []

    for idx, node in enumerate(conv_nodes):
        attrs = {a.name: a for a in node.attribute}
        kernel = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [1, 1]
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
        dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
        groups = attrs["group"].i if "group" in attrs else 1
        auto_pad = attrs["auto_pad"].s.decode() if "auto_pad" in attrs else "NOTSET"

        act_name = node.input[0]
        w_name = node.input[1]
        has_bias = len(node.input) > 2 and node.input[2]

        act_shape = shapes.get(act_name, [])
        w_init = inits.get(w_name)
        if w_init is None or not act_shape:
            print(f"  [{idx:2d}] SKIP — weight or activation shape missing for {node.name}")
            continue

        w_shape = list(w_init.dims)
        out_name_orig = node.output[0]
        out_shape = shapes.get(out_name_orig, [])

        ic = act_shape[1] if len(act_shape) > 1 else 0
        ih = act_shape[2] if len(act_shape) > 2 else 0
        iw = act_shape[3] if len(act_shape) > 3 else 0
        oc = w_shape[0] if w_shape else 0
        sig = _short_sig(ic, ih, iw, oc, kernel[0], strides[0], groups)
        model_name = f"{block_tag}_conv{idx:03d}_{sig}"
        out_path = out_dir / f"{model_name}.onnx"

        # Build new graph ─────────────────────────────────────────────────────
        NEW_ACT = "activation"
        NEW_W   = "weight"
        NEW_B   = "bias"
        NEW_OUT = "output"

        # Graph input (activation — no initializer, runtime-provided)
        act_vi = onnx.helper.make_tensor_value_info(
            NEW_ACT, onnx.TensorProto.FLOAT, act_shape
        )

        # Graph output
        if out_shape:
            out_vi = onnx.helper.make_tensor_value_info(
                NEW_OUT, onnx.TensorProto.FLOAT, out_shape
            )
        else:
            out_vi = onnx.helper.make_tensor_value_info(
                NEW_OUT, onnx.TensorProto.FLOAT, None
            )

        # Initializers: weight (and optional bias)
        w_arr = onnx.numpy_helper.to_array(w_init)
        new_w = onnx.numpy_helper.from_array(w_arr, name=NEW_W)
        new_initializers = [new_w]

        node_inputs = [NEW_ACT, NEW_W]
        if has_bias:
            b_init = inits.get(node.input[2])
            if b_init is not None:
                b_arr = onnx.numpy_helper.to_array(b_init)
                new_b = onnx.numpy_helper.from_array(b_arr, name=NEW_B)
                new_initializers.append(new_b)
                node_inputs.append(NEW_B)

        # Conv node with original attributes
        conv_kwargs: dict = {}
        if kernel:
            conv_kwargs["kernel_shape"] = kernel
        if strides != [1, 1]:
            conv_kwargs["strides"] = strides
        if dilations != [1, 1]:
            conv_kwargs["dilations"] = dilations
        if auto_pad != "NOTSET":
            conv_kwargs["auto_pad"] = auto_pad
        else:
            conv_kwargs["pads"] = pads
        if groups != 1:
            conv_kwargs["group"] = groups

        new_node = onnx.helper.make_node(
            "Conv",
            inputs=node_inputs,
            outputs=[NEW_OUT],
            name=f"conv_{idx:03d}",
            **conv_kwargs,
        )

        new_graph = onnx.helper.make_graph(
            [new_node],
            model_name,
            [act_vi],
            [out_vi],
            initializer=new_initializers,
        )

        new_model = onnx.helper.make_model(
            new_graph,
            opset_imports=[onnx.helper.make_opsetid("", 11)],
        )
        new_model.ir_version = 7

        onnx.checker.check_model(new_model)
        onnx.save(new_model, str(out_path))

        record = {
            "model_name": model_name,
            "path": str(out_path),
            "block": block_tag,
            "index": idx,
            "act_shape": act_shape,
            "w_shape": w_shape,
            "out_shape": out_shape,
            "kernel": kernel,
            "strides": strides,
            "groups": groups,
            "has_bias": bool(has_bias),
        }
        records.append(record)
        print(f"  [{idx:2d}] {model_name}.onnx  "
              f"in={act_shape} w={w_shape} out={out_shape}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Extract per-conv2d ONNX models from BEV blocks")
    parser.add_argument(
        "--assets-dir",
        default="BEV_model",
        help="Root BEV assets directory (default: BEV_model)",
    )
    args = parser.parse_args()

    assets = Path(args.assets_dir)
    split_dir = assets / "split_models"
    out_dir = split_dir / "conv2d_splits"

    sources = [
        (split_dir / "block_A_single_cam0.onnx", "block_A_cam0"),
        (split_dir / "block_C_cylinder_backbone.onnx", "block_C"),
    ]

    all_records = []
    for src_path, tag in sources:
        if not src_path.exists():
            print(f"WARNING: {src_path} not found — skipping")
            continue
        print(f"\nExtracting Conv2d ops from {src_path.name} ...")
        recs = extract_conv2d_models(src_path, tag, out_dir)
        all_records.extend(recs)

    print(f"\n{'='*60}")
    print(f"Total extracted: {len(all_records)} conv2d models")
    print(f"Output dir     : {out_dir}")
    print(f"{'='*60}")

    # Write a summary manifest
    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for r in all_records:
            f.write(f"{r['model_name']}\t{r['act_shape']}\t{r['w_shape']}\t{r['out_shape']}\n")
    print(f"Manifest       : {manifest}")


if __name__ == "__main__":
    main()
