#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
One-shot setup script: splits simple_bev_prep.onnx into 6 block subgraphs and
extracts intermediate boundary tensors for every available input sequence.

Run once before executing the block benchmark tests:
    python forge/test/models/onnx/vision/bev/create_bev_splits.py

Outputs written to BEV_model/:
    split_models/block_A_deformed_backbone.onnx
    split_models/block_B_deformed_bev_transform.onnx
    split_models/block_C_cylinder_backbone.onnx
    split_models/block_D_cylinder_bev_transform.onnx
    split_models/block_E_bev_aggregator.onnx
    split_models/block_F_output_heads.onnx
    intermediate_samples/<seq_id>/feat_deformed_cam{0-3}.npy
    intermediate_samples/<seq_id>/feat_cylinder_cam4.npy
    intermediate_samples/<seq_id>/bev_deformed_cam{0-3}.npy
    intermediate_samples/<seq_id>/bev_cylinder_cam4.npy
    intermediate_samples/<seq_id>/vision_bev_encoder_output.npy
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import onnx

# forge/ must be at sys.path[0] so our `test.*` packages win over the stdlib `test` module.
# forge/test/models/onnx/vision/bev/create_bev_splits.py  →  parents[5] == forge/
_FORGE_DIR = str(Path(__file__).resolve().parents[5])
if _FORGE_DIR in sys.path:
    sys.path.remove(_FORGE_DIR)
sys.path.insert(0, _FORGE_DIR)

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
    load_inputs,
)
from test.models.onnx.vision.bev.model_utils.bev_utils import INPUT_NAMES
from test.models.onnx.vision.bev.model_utils.bev_split_utils import (
    ALL_BOUNDARY_TENSORS,
    BLOCK_DEFS,
    INTERMEDIATE_FILE_NAMES,
    MERGED_BLOCK_DEFS,
    create_merged_models,
    intermediate_samples_dir,
    merged_models_dir,
    split_model,
    split_models_dir,
)


# ---------------------------------------------------------------------------
# Intermediate tensor extraction via ONNX Runtime
# ---------------------------------------------------------------------------

def _add_intermediate_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of *model* with all boundary tensors appended as graph outputs."""
    model = copy.deepcopy(model)
    g = model.graph
    vi_map = {vi.name: vi for vi in g.value_info}
    existing = {o.name for o in g.output}
    missing = []
    for name in ALL_BOUNDARY_TENSORS:
        if name in existing:
            continue
        if name in vi_map:
            g.output.append(vi_map[name])
        else:
            missing.append(name)
    if missing:
        print(f"  [warn] value_info not found for {len(missing)} boundary tensor(s) — they will be skipped.")
        for m in missing:
            print(f"    {m}")
    return model


def extract_intermediate_tensors(model_path: Path, seq_ids: list) -> bool:
    """
    Run ONNX Runtime on an augmented copy of the model that exposes boundary tensors
    as extra graph outputs, then save results as .npy files.

    Returns True on success, False if ORT is unavailable or inference fails.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed — skipping intermediate tensor extraction.")
        print("       Block B/D/E/F benchmarks will use synthetic zero tensors as inputs.")
        return False

    print("\n[create_bev_splits] Adding boundary tensors as graph outputs ...")
    full_model = onnx.load(str(model_path))
    augmented = _add_intermediate_outputs(full_model)

    tmp_path = model_path.parent / "_augmented_tmp.onnx"
    onnx.save(augmented, str(tmp_path))

    success = False
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(
            str(tmp_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        output_names = [o.name for o in sess.get_outputs()]
        input_names_ort = [i.name for i in sess.get_inputs()]

        out_root = intermediate_samples_dir()
        out_root.mkdir(parents=True, exist_ok=True)

        for seq_id in seq_ids:
            print(f"  Sequence {seq_id} ...")
            inputs = load_inputs(seq_id)
            feed = {n: t.numpy() for n, t in zip(input_names_ort, inputs)}

            results = sess.run(None, feed)
            result_map = dict(zip(output_names, results))

            seq_dir = out_root / seq_id
            seq_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for tensor_name, stem in INTERMEDIATE_FILE_NAMES.items():
                if tensor_name in result_map:
                    np.save(str(seq_dir / f"{stem}.npy"), result_map[tensor_name])
                    saved += 1
            print(f"    -> saved {saved}/{len(INTERMEDIATE_FILE_NAMES)} tensors to {seq_dir}")

        success = True

    except Exception as exc:
        print(f"[warn] ONNX Runtime inference failed: {exc}")
        print("       Block B/D/E/F benchmarks will use synthetic zero tensors as inputs.")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not assets_available():
        paths = bev_paths()
        print(f"[error] BEV assets not found under {paths['root']}")
        print("  Set BEV_ASSETS_DIR or populate model/input_samples/output_samples.")
        sys.exit(1)

    paths = bev_paths()
    sequences = list_sequences()
    print(f"[create_bev_splits] Model  : {paths['model']}")
    print(f"[create_bev_splits] Sequences: {sequences}")

    # ── Step 1: split the model ──────────────────────────────────────────────
    print("\n[create_bev_splits] Splitting model into 6 blocks ...")
    split_paths = split_model(paths["model"], split_models_dir())

    print("\n  Summary of extracted blocks:")
    print(f"  {'Block label':<47} {'Size':>8}  {'Nodes':>6}")
    print("  " + "-" * 65)
    for name, cfg in BLOCK_DEFS.items():
        p = split_paths[name]
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {cfg['label']:<47} {size_mb:>6.1f} MB  {cfg['node_count']:>6}")

    # ── Step 2: extract merged block combinations ────────────────────────────
    print("\n[create_bev_splits] Extracting merged block combinations ...")
    merged_paths = create_merged_models(paths["model"], merged_models_dir())

    print("\n  Summary of merged blocks:")
    print(f"  {'Merged label':<52} {'Size':>8}  {'Nodes':>6}")
    print("  " + "-" * 70)
    for name, cfg in MERGED_BLOCK_DEFS.items():
        p = merged_paths[name]
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {cfg['label']:<52} {size_mb:>6.1f} MB  {cfg['node_count']:>6}")

    # ── Step 3: extract intermediate tensors via ONNX Runtime ───────────────
    print("\n[create_bev_splits] Extracting intermediate tensors via ONNX Runtime ...")
    ok = extract_intermediate_tensors(paths["model"], sequences)

    if ok:
        print("\n[create_bev_splits] All done. You can now run:")
        print("  pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py -v")
    else:
        print("\n[create_bev_splits] Model split complete (intermediate tensors not extracted).")
        print("  Block benchmarks that depend on intermediate inputs will use synthetic tensors.")
        print("  You can still run:")
        print("  pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py -v")


if __name__ == "__main__":
    main()
