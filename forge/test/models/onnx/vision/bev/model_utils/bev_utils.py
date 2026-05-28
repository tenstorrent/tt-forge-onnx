# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ONNX graph input order — must match onnx_model.graph.input sequence
INPUT_NAMES: Tuple[str, ...] = (
    "input_0", "input_1", "input_2", "input_3",
    "input_lut_0", "input_lut_1", "input_lut_2", "input_lut_3",
    "input_4", "input_lut_4",
)

OUTPUT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "occupancy_mid_range_height_map": (1, 1, 256, 128),
    "aux_semantic_logits_mid":        (1, 12, 256, 128),
    "aux_visibility_logits_mid":      (1, 1, 256, 128),
}


def _repo_root() -> Path:
    # forge/test/models/onnx/vision/bev/model_utils/bev_utils.py -> repo root
    return Path(__file__).resolve().parents[7]


def bev_assets_dir() -> Path:
    override = os.environ.get("BEV_ASSETS_DIR")
    return Path(override) if override else _repo_root() / "BEV_model"


def bev_paths() -> Dict[str, Path]:
    root = bev_assets_dir()
    return {
        "root": root,
        "model": root / "model" / "simple_bev_prep.onnx",
        "input_samples": root / "input_samples" / "bev_input_samples",
        "output_samples": root / "output_samples",
    }


def assets_available() -> bool:
    paths = bev_paths()
    return (
        paths["model"].is_file()
        and paths["input_samples"].is_dir()
        and paths["output_samples"].is_dir()
    )


def list_sequences() -> List[str]:
    """Return sorted sequence IDs that have both inputs and outputs."""
    paths = bev_paths()
    in_seqs = {p.name for p in paths["input_samples"].iterdir() if p.is_dir()}
    out_seqs = {p.name for p in paths["output_samples"].iterdir() if p.is_dir()}
    return sorted(in_seqs & out_seqs)


def load_inputs(seq_id: Optional[str] = None) -> List[torch.Tensor]:
    """Load all 10 inputs for *seq_id* (defaults to the first available sequence)
    and return them as a list of float32 torch.Tensors in ONNX graph input order."""
    paths = bev_paths()
    if seq_id is None:
        seqs = list_sequences()
        if not seqs:
            raise FileNotFoundError(f"No input sequences found under {paths['input_samples']}")
        seq_id = seqs[0]

    seq_dir = paths["input_samples"] / seq_id
    tensors = []
    for name in INPUT_NAMES:
        arr = np.load(seq_dir / f"{name}.npy")
        tensors.append(torch.from_numpy(arr))
    return tensors


def load_ground_truth_outputs(seq_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load GT output `.bin` files for *seq_id* (defaults to first sequence)
    and return a dict mapping output name → float32 numpy array with the
    declared static shape."""
    paths = bev_paths()
    if seq_id is None:
        seqs = list_sequences()
        if not seqs:
            raise FileNotFoundError(f"No output sequences found under {paths['output_samples']}")
        seq_id = seqs[0]

    out_dir = paths["output_samples"] / seq_id
    gt: Dict[str, np.ndarray] = {}
    for name, shape in OUTPUT_SHAPES.items():
        fpath = out_dir / f"{name}.bin"
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing ground-truth output for '{name}': {fpath}")
        data = np.fromfile(fpath, dtype=np.float32)
        expected = int(np.prod(shape))
        if data.size != expected:
            raise ValueError(
                f"Size mismatch for '{name}': {fpath} has {data.size} floats, "
                f"expected {expected} for shape {shape}"
            )
        gt[name] = data.reshape(shape)
    return gt
