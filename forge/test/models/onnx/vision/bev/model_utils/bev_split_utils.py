# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    INPUT_NAMES,
    bev_assets_dir,
    list_sequences,
    load_inputs,
)

# ---------------------------------------------------------------------------
# Boundary tensor names
# ---------------------------------------------------------------------------

_D = "/model/_backbone/CameraDeformedCylinderEncoder/_camera_encoder"
_C = "/model/_backbone/CameraCylinderEncoder/_camera_encoder"
_D_NECK = (
    f"{_D}/_camera_backbone/_encoder/_trifocal_backbone"
    "/_neck/_top_down.3/_conv_block/_convblock_3/_net"
)
_C_NECK = (
    f"{_C}/_camera_backbone/_encoder/_trifocal_backbone"
    "/_neck/_bottom_up.0/_conv_block/_convblock_3/_net"
)
_D_BEV = f"{_D}/_transformation/_bev_transformation/_bev_transformation"
_C_BEV = f"{_C}/_transformation/_bev_transformation/_bev_transformation"

# Block A → Block B: backbone features for cameras 0-3, each (1, 192, 96, 96)
BLOCK_A_OUTPUTS: Tuple[str, ...] = (
    f"{_D_NECK}/_net.2/Clip_output_0",
    f"{_D_NECK}/_net.2_1/Clip_output_0",
    f"{_D_NECK}/_net.2_2/Clip_output_0",
    f"{_D_NECK}/_net.2_3/Clip_output_0",
)

# Block C → Block D: backbone feature for camera 4, (1, 192, 80, 144)
BLOCK_C_OUTPUT: str = f"{_C_NECK}/_net.2/Clip_output_0"

# Block B → Block E: BEV-space features for cameras 0-3, each (1, 64, 128, 64)
BLOCK_B_OUTPUTS: Tuple[str, ...] = (
    f"{_D_BEV}/_reduce_conv/Conv_output_0",
    f"{_D_BEV}/_reduce_conv_1/Conv_output_0",
    f"{_D_BEV}/_reduce_conv_2/Conv_output_0",
    f"{_D_BEV}/_reduce_conv_3/Conv_output_0",
)

# Block D → Block E: BEV-space feature for camera 4, (1, 64, 128, 64)
BLOCK_D_OUTPUT: str = f"{_C_BEV}/_reduce_conv/Conv_output_0"

# ---------------------------------------------------------------------------
# Block D sub-model tensor boundaries (for MLA hang isolation)
# ---------------------------------------------------------------------------
# Conv+Clip projection output: (1, 64, 80, 144)
_C_BEV_PROJ = f"{_C}/_transformation/_bev_transformation/_to_final_encoding_conv/_net/_net.2"
BLOCK_D_CONV_CLIP_OUTPUT: str = f"{_C_BEV_PROJ}/Clip_output_0"

# 8 Gather outputs from input_lut_4, each (1, 128, 64, 2)
BLOCK_D_GATHER_OUTPUTS: Tuple[str, ...] = tuple(
    f"{_C_BEV}/Gather_{'output_0' if i == 0 else f'{i}_output_0'}"
    for i in range(8)
)

# Concat of all 8 GridSample outputs: (1, 512, 128, 64)
BLOCK_D_CONCAT_OUTPUT: str = f"{_C_BEV}/Concat_output_0"

# Individual GridSample output names (i=0 → "GridSample_output_0", i>0 → "GridSample_{i}_output_0")
BLOCK_D_GRIDSAMPLE_OUTPUTS: Tuple[str, ...] = tuple(
    f"{_C_BEV}/GridSample_{'output_0' if i == 0 else f'{i}_output_0'}"
    for i in range(8)
)

# Sub-model definitions for block D hang debugging
BLOCK_D_SUB_DEFS: Dict[str, Dict] = {
    "block_D_sub1_conv_clip": {
        "label": "Block D Sub1: Conv + Clip (2 nodes — feature projection 192→64)",
        "node_count": 2,
        "inputs": [BLOCK_C_OUTPUT],
        "outputs": [BLOCK_D_CONV_CLIP_OUTPUT],
        "input_shapes": {BLOCK_C_OUTPUT: (1, 192, 80, 144)},
        "output_shapes": {BLOCK_D_CONV_CLIP_OUTPUT: (1, 64, 80, 144)},
    },
    "block_D_sub2_gather": {
        "label": "Block D Sub2: 8× Gather (8 nodes — LUT coordinate slicing)",
        "node_count": 8,
        "inputs": ["input_lut_4"],
        "outputs": list(BLOCK_D_GATHER_OUTPUTS),
        "input_shapes": {"input_lut_4": (1, 128, 64, 8, 2)},
        "output_shapes": {g: (1, 128, 64, 2) for g in BLOCK_D_GATHER_OUTPUTS},
    },
    "block_D_sub3_gridsample_concat": {
        "label": "Block D Sub3: 8× GridSample + Concat (9 nodes — BEV sampling, primary suspect)",
        "node_count": 9,
        "inputs": [BLOCK_D_CONV_CLIP_OUTPUT] + list(BLOCK_D_GATHER_OUTPUTS),
        "outputs": [BLOCK_D_CONCAT_OUTPUT],
        "input_shapes": {
            BLOCK_D_CONV_CLIP_OUTPUT: (1, 64, 80, 144),
            **{g: (1, 128, 64, 2) for g in BLOCK_D_GATHER_OUTPUTS},
        },
        "output_shapes": {BLOCK_D_CONCAT_OUTPUT: (1, 512, 128, 64)},
    },
    "block_D_sub4_reduce_conv": {
        "label": "Block D Sub4: Final Conv (1 node — channel reduction 512→64)",
        "node_count": 1,
        "inputs": [BLOCK_D_CONCAT_OUTPUT],
        "outputs": [BLOCK_D_OUTPUT],
        "input_shapes": {BLOCK_D_CONCAT_OUTPUT: (1, 512, 128, 64)},
        "output_shapes": {BLOCK_D_OUTPUT: (1, 64, 128, 64)},
    },
}

# One sub-model per GridSample for per-op MLA hang isolation.
# All 8 share the same data input (clip_out) but use a different grid tensor.
# Attributes on every GS: mode=nearest, padding=zeros, align_corners=1.
BLOCK_D_GRIDSAMPLE_DEFS: Dict[str, Dict] = {
    f"block_D_gs{i}": {
        "label": (
            f"Block D GS{i}: GridSample (nearest, align_corners=1) "
            f"data(1,64,80,144) grid(1,128,64,2) → (1,64,128,64)"
        ),
        "node_count": 1,
        "gs_index": i,
        "inputs": [BLOCK_D_CONV_CLIP_OUTPUT, BLOCK_D_GATHER_OUTPUTS[i]],
        "outputs": [BLOCK_D_GRIDSAMPLE_OUTPUTS[i]],
        "input_shapes": {
            BLOCK_D_CONV_CLIP_OUTPUT: (1, 64, 80, 144),
            BLOCK_D_GATHER_OUTPUTS[i]: (1, 128, 64, 2),
        },
        "output_shapes": {BLOCK_D_GRIDSAMPLE_OUTPUTS[i]: (1, 64, 128, 64)},
    }
    for i in range(8)
}

# Block E → Block F: fused BEV feature, (1, 64, 128, 64)
BLOCK_E_OUTPUT: str = "vision_bev_encoder_output"

# Final model outputs
BLOCK_F_OUTPUTS: Tuple[str, ...] = (
    "occupancy_mid_range_height_map",
    "aux_semantic_logits_mid",
    "aux_visibility_logits_mid",
)

# ---------------------------------------------------------------------------
# Block definitions (inputs + outputs for onnx.utils.extract_model)
# ---------------------------------------------------------------------------

BLOCK_DEFS: Dict[str, Dict] = {
    "block_A_deformed_backbone": {
        "label": "Block A: CameraDeformedCylinder Backbone",
        "node_count": 460,
        "inputs": ["input_0", "input_1", "input_2", "input_3"],
        "outputs": list(BLOCK_A_OUTPUTS),
    },
    "block_B_deformed_bev_transform": {
        "label": "Block B: CameraDeformedCylinder BEV Transform",
        "node_count": 80,
        "inputs": list(BLOCK_A_OUTPUTS) + ["input_lut_0", "input_lut_1", "input_lut_2", "input_lut_3"],
        "outputs": list(BLOCK_B_OUTPUTS),
    },
    "block_C_cylinder_backbone": {
        "label": "Block C: CameraCylinder Backbone",
        "node_count": 129,
        "inputs": ["input_4"],
        "outputs": [BLOCK_C_OUTPUT],
    },
    "block_D_cylinder_bev_transform": {
        "label": "Block D: CameraCylinder BEV Transform",
        "node_count": 20,
        "inputs": [BLOCK_C_OUTPUT, "input_lut_4"],
        "outputs": [BLOCK_D_OUTPUT],
    },
    "block_E_bev_aggregator": {
        "label": "Block E: BEV Aggregator Backbone",
        "node_count": 57,
        "inputs": list(BLOCK_B_OUTPUTS) + [BLOCK_D_OUTPUT],
        "outputs": [BLOCK_E_OUTPUT],
    },
    "block_F_output_heads": {
        "label": "Block F: Output Heads",
        "node_count": 24,
        "inputs": [BLOCK_E_OUTPUT],
        "outputs": list(BLOCK_F_OUTPUTS),
    },
}

# All boundary tensors that need to be captured as intermediate outputs
ALL_BOUNDARY_TENSORS: Tuple[str, ...] = (
    *BLOCK_A_OUTPUTS,
    BLOCK_C_OUTPUT,
    *BLOCK_B_OUTPUTS,
    BLOCK_D_OUTPUT,
    BLOCK_E_OUTPUT,
)

# Maps boundary tensor name → short file stem for .npy storage
INTERMEDIATE_FILE_NAMES: Dict[str, str] = {
    BLOCK_A_OUTPUTS[0]: "feat_deformed_cam0",
    BLOCK_A_OUTPUTS[1]: "feat_deformed_cam1",
    BLOCK_A_OUTPUTS[2]: "feat_deformed_cam2",
    BLOCK_A_OUTPUTS[3]: "feat_deformed_cam3",
    BLOCK_C_OUTPUT:     "feat_cylinder_cam4",
    BLOCK_B_OUTPUTS[0]: "bev_deformed_cam0",
    BLOCK_B_OUTPUTS[1]: "bev_deformed_cam1",
    BLOCK_B_OUTPUTS[2]: "bev_deformed_cam2",
    BLOCK_B_OUTPUTS[3]: "bev_deformed_cam3",
    BLOCK_D_OUTPUT:     "bev_cylinder_cam4",
    BLOCK_E_OUTPUT:     "vision_bev_encoder_output",
}

# Shapes used for synthetic fallback tensors when real intermediates are not available
INTERMEDIATE_SHAPES: Dict[str, Tuple[int, ...]] = {
    BLOCK_A_OUTPUTS[0]: (1, 192, 96, 96),
    BLOCK_A_OUTPUTS[1]: (1, 192, 96, 96),
    BLOCK_A_OUTPUTS[2]: (1, 192, 96, 96),
    BLOCK_A_OUTPUTS[3]: (1, 192, 96, 96),
    BLOCK_C_OUTPUT:     (1, 192, 80, 144),
    BLOCK_B_OUTPUTS[0]: (1, 64, 128, 64),
    BLOCK_B_OUTPUTS[1]: (1, 64, 128, 64),
    BLOCK_B_OUTPUTS[2]: (1, 64, 128, 64),
    BLOCK_B_OUTPUTS[3]: (1, 64, 128, 64),
    BLOCK_D_OUTPUT:     (1, 64, 128, 64),
    BLOCK_E_OUTPUT:     (1, 64, 128, 64),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def split_models_dir() -> Path:
    return bev_assets_dir() / "split_models"


def intermediate_samples_dir() -> Path:
    return bev_assets_dir() / "intermediate_samples"


def split_models_available() -> bool:
    sdir = split_models_dir()
    if not sdir.is_dir():
        return False
    return all((sdir / f"{name}.onnx").is_file() for name in BLOCK_DEFS)


def intermediate_samples_available(seq_id: Optional[str] = None) -> bool:
    idir = intermediate_samples_dir()
    if not idir.is_dir():
        return False
    if seq_id is None:
        seqs = sorted(p.name for p in idir.iterdir() if p.is_dir())
        if not seqs:
            return False
        seq_id = seqs[0]
    seq_dir = idir / seq_id
    return all((seq_dir / f"{stem}.npy").is_file() for stem in INTERMEDIATE_FILE_NAMES.values())


# ---------------------------------------------------------------------------
# Model splitting
# ---------------------------------------------------------------------------

def split_model(model_path: Path, output_dir: Path) -> Dict[str, Path]:
    """Extract each block as a separate ONNX file using onnx.utils.extract_model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, Path] = {}

    for name, cfg in BLOCK_DEFS.items():
        out_path = output_dir / f"{name}.onnx"
        print(f"  Extracting {name} ...")
        onnx.utils.extract_model(
            str(model_path),
            str(out_path),
            cfg["inputs"],
            cfg["outputs"],
        )
        split_paths[name] = out_path
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    -> {out_path.name}  ({size_mb:.1f} MB)")

    return split_paths


# ---------------------------------------------------------------------------
# Intermediate tensor helpers
# ---------------------------------------------------------------------------

def load_intermediate_tensors(seq_id: str) -> Dict[str, torch.Tensor]:
    """Load saved intermediate boundary tensors for *seq_id*."""
    seq_dir = intermediate_samples_dir() / seq_id
    return {
        tensor_name: torch.from_numpy(np.load(seq_dir / f"{stem}.npy"))
        for tensor_name, stem in INTERMEDIATE_FILE_NAMES.items()
    }


def synthetic_intermediate_tensors() -> Dict[str, torch.Tensor]:
    """Zero tensors of the correct shapes — used when real intermediates are unavailable."""
    return {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in INTERMEDIATE_SHAPES.items()
    }


# ---------------------------------------------------------------------------
# Block input loaders
# ---------------------------------------------------------------------------

def load_block_inputs(block_name: str, seq_id: str) -> List[torch.Tensor]:
    """
    Return the input tensors for *block_name* in the order declared in BLOCK_DEFS,
    mixing full model inputs with intermediate tensors as needed.
    Uses real intermediates if available, synthetic zeros otherwise.
    """
    cfg = BLOCK_DEFS[block_name]
    full_inputs: Dict[str, torch.Tensor] = dict(zip(INPUT_NAMES, load_inputs(seq_id)))

    if intermediate_samples_available(seq_id):
        intermediates = load_intermediate_tensors(seq_id)
    else:
        intermediates = synthetic_intermediate_tensors()

    merged = {**full_inputs, **intermediates}
    return [merged[inp_name] for inp_name in cfg["inputs"]]


def load_block_inputs_pool(
    block_name: str, seq_ids: List[str], pool_size: int
) -> List[List[torch.Tensor]]:
    """Return a pool of *pool_size* input lists cycling through *seq_ids*."""
    return [load_block_inputs(block_name, seq_ids[i % len(seq_ids)]) for i in range(pool_size)]


# ---------------------------------------------------------------------------
# Merged block definitions
# A merged block spans multiple consecutive blocks extracted as one ONNX model.
# ---------------------------------------------------------------------------

MERGED_BLOCK_DEFS: Dict[str, Dict] = {
    "merged_AB": {
        "label":      "Merged A+B: Deformed Camera Backbone → BEV",
        "node_count": 540,   # 460 + 80
        "blocks":     ["block_A_deformed_backbone", "block_B_deformed_bev_transform"],
        "inputs":     ["input_0", "input_1", "input_2", "input_3",
                       "input_lut_0", "input_lut_1", "input_lut_2", "input_lut_3"],
        "outputs":    list(BLOCK_B_OUTPUTS),
    },
    "merged_CD": {
        "label":      "Merged C+D: Cylinder Camera Backbone → BEV",
        "node_count": 149,   # 129 + 20
        "blocks":     ["block_C_cylinder_backbone", "block_D_cylinder_bev_transform"],
        "inputs":     ["input_4", "input_lut_4"],
        "outputs":    [BLOCK_D_OUTPUT],
    },
    "merged_ABCD": {
        "label":      "Merged A+B+C+D: All Cameras → BEV",
        "node_count": 689,   # 460 + 80 + 129 + 20
        "blocks":     [
            "block_A_deformed_backbone", "block_B_deformed_bev_transform",
            "block_C_cylinder_backbone", "block_D_cylinder_bev_transform",
        ],
        "inputs":     [
            "input_0", "input_1", "input_2", "input_3",
            "input_lut_0", "input_lut_1", "input_lut_2", "input_lut_3",
            "input_4", "input_lut_4",
        ],
        "outputs":    list(BLOCK_B_OUTPUTS) + [BLOCK_D_OUTPUT],
    },
    "merged_EF": {
        "label":      "Merged E+F: BEV Aggregator → Output Heads",
        "node_count": 81,    # 57 + 24
        "blocks":     ["block_E_bev_aggregator", "block_F_output_heads"],
        "inputs":     [BLOCK_E_OUTPUT],
        "outputs":    list(BLOCK_F_OUTPUTS),
    },
    "merged_ABCDEF": {
        "label":      "Merged A+B+C+D+E+F: Full Pipeline",
        "node_count": 770,   # sum of all blocks
        "blocks":     list(BLOCK_DEFS.keys()),
        "inputs":     list(INPUT_NAMES),
        "outputs":    list(BLOCK_F_OUTPUTS),
    },
}


def merged_models_dir() -> Path:
    return bev_assets_dir() / "merged_models"


def merged_models_available() -> bool:
    mdir = merged_models_dir()
    if not mdir.is_dir():
        return False
    return all((mdir / f"{name}.onnx").is_file() for name in MERGED_BLOCK_DEFS)


def create_merged_models(model_path: Path, output_dir: Path) -> Dict[str, Path]:
    """Extract each merged block combination as a separate ONNX file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_paths: Dict[str, Path] = {}
    for name, cfg in MERGED_BLOCK_DEFS.items():
        out_path = output_dir / f"{name}.onnx"
        print(f"  Extracting {name} ...")
        onnx.utils.extract_model(str(model_path), str(out_path), cfg["inputs"], cfg["outputs"])
        merged_paths[name] = out_path
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    -> {out_path.name}  ({size_mb:.1f} MB)")
    return merged_paths


def load_merged_inputs(merged_name: str, seq_id: str) -> List[torch.Tensor]:
    """Return input tensors for a merged block combination, same fallback logic as load_block_inputs."""
    cfg = MERGED_BLOCK_DEFS[merged_name]
    full_inputs: Dict[str, torch.Tensor] = dict(zip(INPUT_NAMES, load_inputs(seq_id)))
    if intermediate_samples_available(seq_id):
        intermediates = load_intermediate_tensors(seq_id)
    else:
        intermediates = synthetic_intermediate_tensors()
    tensor_map = {**full_inputs, **intermediates}
    return [tensor_map[inp_name] for inp_name in cfg["inputs"]]


def load_merged_inputs_pool(
    merged_name: str, seq_ids: List[str], pool_size: int
) -> List[List[torch.Tensor]]:
    """Return a pool of *pool_size* input lists for a merged block combination."""
    return [load_merged_inputs(merged_name, seq_ids[i % len(seq_ids)]) for i in range(pool_size)]


# ---------------------------------------------------------------------------
# Block D sub-model helpers (for MLA hang isolation)
# ---------------------------------------------------------------------------

def block_d_debug_dir() -> Path:
    return bev_assets_dir() / "block_d_debug_models"


def block_d_subs_available() -> bool:
    d = block_d_debug_dir()
    if not d.is_dir():
        return False
    return all((d / f"{name}.onnx").is_file() for name in BLOCK_D_SUB_DEFS)


def split_block_d_subs(block_d_model_path: Path, output_dir: Path) -> Dict[str, Path]:
    """Extract the 4 Block D sub-models for MLA hang isolation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, cfg in BLOCK_D_SUB_DEFS.items():
        out_path = output_dir / f"{name}.onnx"
        print(f"  Extracting {name} ...")
        onnx.utils.extract_model(
            str(block_d_model_path), str(out_path), cfg["inputs"], cfg["outputs"]
        )
        paths[name] = out_path
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    -> {out_path.name}  ({size_mb:.2f} MB, {cfg['node_count']} nodes)")
    return paths


# ---------------------------------------------------------------------------
# Per-GridSample sub-model helpers (extracted from D3 sub-model)
# ---------------------------------------------------------------------------

def block_d_gridsample_dir() -> Path:
    return bev_assets_dir() / "block_d_gridsample_models"


def block_d_gridsample_available() -> bool:
    d = block_d_gridsample_dir()
    if not d.is_dir():
        return False
    return all((d / f"{name}.onnx").is_file() for name in BLOCK_D_GRIDSAMPLE_DEFS)


def split_block_d_gridsample_subs(output_dir: Path) -> Dict[str, Path]:
    """Extract 8 single-GridSample sub-models from the D3 sub-model."""
    d3_path = block_d_debug_dir() / "block_D_sub3_gridsample_concat.onnx"
    if not d3_path.is_file():
        raise FileNotFoundError(
            f"D3 sub-model not found at {d3_path}. "
            "Run split_block_d_subs() first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, cfg in BLOCK_D_GRIDSAMPLE_DEFS.items():
        out_path = output_dir / f"{name}.onnx"
        print(f"  Extracting {name} (GS{cfg['gs_index']}) ...")
        onnx.utils.extract_model(
            str(d3_path), str(out_path), cfg["inputs"], cfg["outputs"]
        )
        paths[name] = out_path
        size_kb = out_path.stat().st_size / 1024
        print(f"    -> {out_path.name}  ({size_kb:.1f} KB)")
    return paths
