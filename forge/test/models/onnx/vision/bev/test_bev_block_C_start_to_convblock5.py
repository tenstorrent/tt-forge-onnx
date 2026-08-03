# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Block C isolation and full-model PCC tests.

Models (BEV_model/split_models/):
    block_C_start_to_convblock5.onnx   — model input → _convblock_5 Conv  (34 nodes)
    block_C_cylinder_backbone.onnx     — full Block C                      (129 nodes)

Usage:
    pytest forge/test/models/onnx/vision/bev/test_bev_block_C_start_to_convblock5.py -s -v
    pytest ... ::test_start_to_convblock5
    pytest ... ::test_full_block_C
"""

import onnx
import os
import pytest
import torch
from pathlib import Path
from typing import List

import forge
import forge._C
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.onnx.vision.bev.model_utils.bev_split_utils import (
    load_block_inputs,
    split_models_dir,
)
from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
)

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SPLIT_DIR = _REPO_ROOT / "BEV_model/split_models"

_MODEL_CONV_TO_CONCAT    = _SPLIT_DIR / "block_C_conv_to_concat.onnx"
_MODEL_START_TO_CONV_B1  = _SPLIT_DIR / "block_C_start_to_conv_blocks1.onnx"
_MODEL_START_TO_CONV_B3  = _SPLIT_DIR / "block_C_start_to_conv_blocks3.onnx"
_MODEL_START_TO_CONV5    = _SPLIT_DIR / "block_C_start_to_convblock5.onnx"
_MODEL_FULL_BLOCK_C      = _SPLIT_DIR / "block_C_cylinder_backbone.onnx"

_PCC = 0.99


def _make_cfg() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(False)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg


def _compile_onnx(model_path: Path, sample_inputs: List[torch.Tensor], module_name: str):
    """Compile an ONNX model through forge, matching the benchmark compile pattern."""
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    _dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    _sync_prof = os.environ.pop("TT_METAL_PROFILER_SYNC", None)
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        compiled = forge.compile(
            onnx_model,
            sample_inputs=sample_inputs,
            compiler_cfg=_make_cfg(),
            module_name=module_name,
        )
    finally:
        if _dispatch_prof is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
        if _sync_prof is not None:
            os.environ["TT_METAL_PROFILER_SYNC"] = _sync_prof
    return compiled


def _validate(model_path: Path, module_name: str, compiled, inputs: List[torch.Tensor]):
    """Compare compiled output against ONNX Runtime reference (pcc=0.98)."""
    onnx_model = onnx.load(str(model_path))
    framework_model = forge.OnnxModule(module_name, onnx_model)
    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=_PCC))
    verify(inputs, framework_model, compiled, verify_cfg=verify_cfg)


@pytest.fixture(scope="session")
def bev_assets():
    if not assets_available():
        paths = bev_paths()
        pytest.skip(
            f"BEV assets not found under {paths['root']}. "
            "Set BEV_ASSETS_DIR or populate model/input_samples/output_samples."
        )
    return bev_paths()


@pytest.fixture(scope="module")
def sample_inputs(bev_assets):
    """Real intermediate samples for block_C_cylinder_backbone (input_4)."""
    sequences = list_sequences()
    seq_id = sequences[0]
    return load_block_inputs("block_C_cylinder_backbone", seq_id)


def test_conv_to_concat(sample_inputs):
    """Block C: initial Conv → pixel_unshuffle chain → Concat (11 nodes). Validates PCC >= 0.98."""
    module_name = "block_C_conv_to_concat"
    compiled = _compile_onnx(_MODEL_CONV_TO_CONCAT, sample_inputs, module_name)

    output = compiled(*sample_inputs)
    print(f"\n[conv_to_concat] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_CONV_TO_CONCAT, module_name, compiled, sample_inputs)
    print(f"[conv_to_concat] PASSED (pcc >= {_PCC})")


def test_start_to_conv_blocks1(sample_inputs):
    """Block C: model input → _conv_blocks_1.0 Conv (12 nodes). Validates PCC >= 0.99."""
    module_name = "block_C_start_to_conv_blocks1"
    compiled = _compile_onnx(_MODEL_START_TO_CONV_B1, sample_inputs, module_name)

    output = compiled(*sample_inputs)
    print(f"\n[start_to_conv_blocks1] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_START_TO_CONV_B1, module_name, compiled, sample_inputs)
    print(f"[start_to_conv_blocks1] PASSED (pcc >= {_PCC})")


def test_start_to_conv_blocks3(sample_inputs):
    """Block C: model input → _conv_blocks_3.0 Conv (21 nodes). Validates PCC >= 0.99."""
    module_name = "block_C_start_to_conv_blocks3"
    compiled = _compile_onnx(_MODEL_START_TO_CONV_B3, sample_inputs, module_name)

    output = compiled(*sample_inputs)
    print(f"\n[start_to_conv_blocks3] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_START_TO_CONV_B3, module_name, compiled, sample_inputs)
    print(f"[start_to_conv_blocks3] PASSED (pcc >= {_PCC})")


def test_start_to_convblock5(sample_inputs):
    """Block C start → _convblock_5 Conv (34 nodes). Validates PCC >= 0.98."""
    module_name = "block_C_start_to_convblock5"
    compiled = _compile_onnx(_MODEL_START_TO_CONV5, sample_inputs, module_name)

    output = compiled(*sample_inputs)
    print(f"\n[start_to_convblock5] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_START_TO_CONV5, module_name, compiled, sample_inputs)
    print(f"[start_to_convblock5] PASSED (pcc >= {_PCC})")


def test_full_block_C(sample_inputs):
    """Full Block C cylinder backbone (129 nodes). Validates PCC >= 0.98."""
    module_name = "block_C_cylinder_backbone"
    compiled = _compile_onnx(_MODEL_FULL_BLOCK_C, sample_inputs, module_name)

    output = compiled(*sample_inputs)
    print(f"\n[full_block_C] Output shape: {list(output[0].shape)}")

    _validate(_MODEL_FULL_BLOCK_C, module_name, compiled, sample_inputs)
    print(f"[full_block_C] PASSED (pcc >= {_PCC})")
