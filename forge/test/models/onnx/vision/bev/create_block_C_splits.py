#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Split block_C_cylinder_backbone.onnx into two halves and save them to disk.

Output files (written to BEV_model/split_models/):
    block_C_first_half.onnx   ops[0..63]   (64 ops, input: input_4)
    block_C_second_half.onnx  ops[64..128] (65 ops, input: intermediate tensor)

Usage:
    python forge/test/models/onnx/vision/bev/create_block_C_splits.py
"""

import sys
from pathlib import Path

import onnx
import onnx.utils

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SRC = _REPO_ROOT / "BEV_model/split_models/block_C_q1a.onnx"
_OUT_DIR = _REPO_ROOT / "BEV_model/split_models"

SPLIT_IDX = 10  # split q1a (14 ops) after pixel_unshuffle chain (ops 0..10), before Conv op[11]


def main():
    if not _SRC.exists():
        print(f"ERROR: {_SRC} not found.")
        print("Run: python forge/test/models/onnx/vision/bev/create_bev_splits.py")
        sys.exit(1)

    print(f"Loading {_SRC} ...")
    model = onnx.load(str(_SRC))
    onnx.checker.check_model(model)

    n = len(model.graph.node)
    print(f"Block C: {n} ops total, splitting at op[{SPLIT_IDX}]")

    # Print op list around the split point
    for i in range(max(0, SPLIT_IDX - 2), min(n, SPLIT_IDX + 3)):
        marker = " <-- split" if i == SPLIT_IDX else ""
        print(f"  [{i:3d}] {model.graph.node[i].op_type:20s}  "
              f"{model.graph.node[i].output[0][:60]}{marker}")

    # --- First half: ops[0..SPLIT_IDX] ---
    cut_node = model.graph.node[SPLIT_IDX]
    cut_output = cut_node.output[0]
    model_inputs = [inp.name for inp in model.graph.input]

    print(f"\nExtracting first half (ops 0..{SPLIT_IDX}) ...")
    print(f"  input : {model_inputs}")
    print(f"  output: {cut_output[:80]}")
    extractor = onnx.utils.Extractor(model)
    first_half = extractor.extract_model(model_inputs, [cut_output])
    onnx.checker.check_model(first_half)
    out1 = _OUT_DIR / "block_C_q1a_p1.onnx"
    onnx.save(first_half, str(out1))
    print(f"  Saved: {out1}")
    print(f"  Ops: {len(first_half.graph.node)}, "
          f"Input: {[i.name for i in first_half.graph.input]}, "
          f"Output: {[o.name for o in first_half.graph.output]}")

    # --- Second half: ops[SPLIT_IDX+1..end] ---
    # Some ops in the second half still reference input_4 directly (multi-branch
    # model), so both input_4 and the intermediate cut tensor are inputs.
    last_node = model.graph.node[-1]
    last_output = last_node.output[0]
    second_inputs = model_inputs + [cut_output]

    print(f"\nExtracting second half (ops {SPLIT_IDX+1}..{n-1}) ...")
    print(f"  inputs: {second_inputs}")
    print(f"  output: {last_output[:80]}")
    extractor2 = onnx.utils.Extractor(model)
    second_half = extractor2.extract_model(second_inputs, [last_output])
    onnx.checker.check_model(second_half)
    out2 = _OUT_DIR / "block_C_q1a_p2.onnx"
    onnx.save(second_half, str(out2))
    print(f"  Saved: {out2}")
    print(f"  Ops: {len(second_half.graph.node)}, "
          f"Input: {[i.name for i in second_half.graph.input]}, "
          f"Output: {[o.name for o in second_half.graph.output]}")

    print(f"\nDone. Split files written to {_OUT_DIR}")


if __name__ == "__main__":
    main()
