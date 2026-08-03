#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract the single-camera "conv to conv after concat" subgraph from
block_A_deformed_backbone.onnx for camera 0 (input_0).

The extracted subgraph mirrors block_C_conv_to_conv.onnx in structure:

  input_0  (1, 3, 1280, 2304)
    → Conv[0]                   YUV 420 input adapter
    → Slice[4,5]                Y / UV split
    → Reshape[12], AveragePool[13], Transpose[20]   spatial permute
    → Reshape[21], Transpose[29]
    → Reshape[28], Reshape[36]
    → Concat[40]                pixel-unshuffle concat
    → Conv[44]                  first backbone conv after concat
    → output  (1, 64, 320, 576)

Op indices 0, 4, 5, 12, 13, 20, 21, 28, 29, 36, 40, 44 are camera 0's branch.
The ONNX extractor reaches only those nodes via input_0; cameras 1–3 are dropped.

Usage:
    python forge/test/models/onnx/vision/bev/create_block_A_single_cam_split.py

Output:
    BEV_model/split_models/block_A_single_cam_conv_to_conv.onnx
"""
import sys
from pathlib import Path

import onnx
import onnx.utils

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SRC = _REPO_ROOT / "BEV_model/split_models/block_A_deformed_backbone.onnx"
_OUT_DIR = _REPO_ROOT / "BEV_model/split_models"

# Indices within block_A_deformed_backbone.onnx for camera 0's chain.
# [40] is the first Concat (pixel-unshuffle result for cam 0).
# [44] is the first Conv after that Concat.
_CONCAT_IDX = 40
_CONV_AFTER_CONCAT_IDX = 44


def main() -> None:
    if not _SRC.exists():
        print(f"ERROR: {_SRC} not found.")
        print("Run: python forge/test/models/onnx/vision/bev/create_bev_splits.py first.")
        sys.exit(1)

    print(f"Loading {_SRC} ...")
    model = onnx.load(str(_SRC))
    onnx.checker.check_model(model)
    print(f"  {len(model.graph.node)} nodes total")

    # Confirm node types at the expected indices
    for idx in [0, _CONCAT_IDX, _CONV_AFTER_CONCAT_IDX]:
        node = model.graph.node[idx]
        print(f"  [{idx:3d}] {node.op_type:18s}  out={node.output[0][:80]}")

    assert model.graph.node[_CONCAT_IDX].op_type == "Concat", \
        f"Expected Concat at [{_CONCAT_IDX}], got {model.graph.node[_CONCAT_IDX].op_type}"
    assert model.graph.node[_CONV_AFTER_CONCAT_IDX].op_type == "Conv", \
        f"Expected Conv at [{_CONV_AFTER_CONCAT_IDX}], got {model.graph.node[_CONV_AFTER_CONCAT_IDX].op_type}"

    # Verify the Conv at [44] feeds from the Concat at [40]
    concat_out = model.graph.node[_CONCAT_IDX].output[0]
    conv_inputs = list(model.graph.node[_CONV_AFTER_CONCAT_IDX].input)
    assert concat_out in conv_inputs, \
        f"Conv[{_CONV_AFTER_CONCAT_IDX}] inputs {conv_inputs} do not include Concat output {concat_out[:40]}"

    conv_output = model.graph.node[_CONV_AFTER_CONCAT_IDX].output[0]

    print(f"\nExtracting camera-0 subgraph:")
    print(f"  graph input : ['input_0']")
    print(f"  graph output: {conv_output[:80]}")

    extractor = onnx.utils.Extractor(model)
    sub = extractor.extract_model(["input_0"], [conv_output])
    onnx.checker.check_model(sub)

    out_path = _OUT_DIR / "block_A_single_cam_conv_to_conv.onnx"
    onnx.save(sub, str(out_path))

    in_shapes  = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                  for i in sub.graph.input]
    out_shapes = [(o.name[:60], [d.dim_value for d in o.type.tensor_type.shape.dim])
                  for o in sub.graph.output]

    print(f"\nSaved: {out_path}")
    print(f"  Nodes  : {len(sub.graph.node)}")
    print(f"  Inputs : {in_shapes}")
    print(f"  Outputs: {out_shapes}")


if __name__ == "__main__":
    main()
