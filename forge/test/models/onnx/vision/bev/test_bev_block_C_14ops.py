# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test subgraphs of Block C to reproduce and isolate the CB clash:

    TT_THROW: Statically allocated circular buffers clash with L1 buffers.
    L1 buffer at 696320, CB region ends at 849184.

Model files (BEV_model/split_models/):
    block_C_14ops_cb_clash.onnx    ops[0..13]  — full 14-op repro
    block_C_conv_to_concat.onnx    ops[0..10]  — pixel_unshuffle chain only
    block_C_conv_to_conv.onnx      ops[0..11]  — pixel_unshuffle + clashing Conv

Op list:
    [0]  Conv             initial 3x3 conv on raw input
    [1]  Slice
    [2]  Slice
    [3]  Reshape
    [4]  AveragePool
    [5]  Transpose
    [6]  Reshape
    [7]  Reshape
    [8]  Transpose
    [9]  Reshape
    [10] Concat           end of pixel_unshuffle chain
    [11] Conv             <-- height-sharded conv2d (suspected culprit)
    [12] Clip
    [13] Slice

Usage:
    pytest forge/test/models/onnx/vision/bev/test_bev_block_C_14ops.py -s -v
"""

import onnx
import pytest
import torch
from pathlib import Path

import forge
import forge._C
from forge._C import MLIRConfig
from forge.config import CompilerConfig
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

_REPO_ROOT = Path(__file__).resolve().parents[6]
_SPLIT_DIR = _REPO_ROOT / "BEV_model/split_models"

_MODEL_CONV_CONV = _SPLIT_DIR / "block_C_conv_to_conv.onnx"


_PCC = 0.99


def _validate(model_path: Path, compiled, inputs) -> None:
    """Compare compiled output against ONNX Runtime reference (pcc=0.98)."""
    onnx_model = onnx.load(str(model_path))
    module_name = model_path.stem
    framework_model = forge.OnnxModule(module_name, onnx_model)
    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=_PCC))
    verify(inputs, framework_model, compiled, verify_cfg=verify_cfg)


def _make_cfg() -> CompilerConfig:
    mlir_config = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_trace(True)
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg


@pytest.fixture(scope="module")
def sample_input():
    return torch.zeros(1, 3, 1280, 2304, dtype=torch.float32)


def _configure_device() -> None:
    from forge._C import runtime as forge_runtime
    ds = forge_runtime.experimental.DeviceSettings()
    ds.enable_program_cache = True
    forge_runtime.experimental.configure_devices(ds)


def test_conv_to_conv(sample_input):
    """ops[0..11] — pixel_unshuffle + clashing Conv."""
    model = onnx.load(str(_MODEL_CONV_CONV))
    onnx.checker.check_model(model)
    print(f"\n[conv_to_conv] {len(model.graph.node)} ops (Conv to Conv)")
    compiled = forge.compile(model, sample_inputs=[sample_input],
                             compiler_cfg=_make_cfg(), module_name="block_C_conv_to_conv")
    _configure_device()
    output = compiled(sample_input)
    print(f"[conv_to_conv] Output shape: {list(output[0].shape)}")
    _validate(_MODEL_CONV_CONV, compiled, [sample_input])
    print(f"[conv_to_conv] PASSED (pcc >= {_PCC})")
