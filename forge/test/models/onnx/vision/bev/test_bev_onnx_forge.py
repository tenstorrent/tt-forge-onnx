# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import pytest
import torch
from loguru import logger

import forge

from forge.verify.compare import calculate_atol, calculate_pcc, compare_with_golden
from forge.verify.verify import verify

from test.models.onnx.vision.bev.model_utils.bev_utils import (
    assets_available,
    bev_paths,
    list_sequences,
    load_ground_truth_outputs,
    load_inputs,
)


PCC = 0.99

_SEQUENCES = list_sequences() if assets_available() else []


@pytest.fixture(scope="module")
def bev_compiled():
    if not assets_available():
        paths = bev_paths()
        pytest.skip(
            f"BEV assets not found under {paths['root']}. "
            "Set BEV_ASSETS_DIR or populate model/input_samples/output_samples."
        )


    paths = bev_paths()
    onnx_model = onnx.load(str(paths["model"]))
    onnx.checker.check_model(onnx_model)

    sample_inputs = load_inputs(_SEQUENCES[0])
    framework_model = forge.OnnxModule("onnx_bev", onnx_model)
    compiled_model = forge.compile(onnx_model, sample_inputs=sample_inputs, module_name="onnx_bev")
    output_names = [o.name for o in onnx_model.graph.output]
    return framework_model, compiled_model, output_names


@pytest.mark.nightly
@pytest.mark.parametrize(
    "seq_id",
    _SEQUENCES if _SEQUENCES else [pytest.param("", marks=pytest.mark.skip(reason="BEV assets not found"))],
)
def test_bev_onnx(bev_compiled, seq_id):
    framework_model, compiled_model, output_names = bev_compiled

    input_tensors = load_inputs(seq_id)

    verify(input_tensors, framework_model, compiled_model)

    forge_outputs = compiled_model(*input_tensors)
    if not isinstance(forge_outputs, (list, tuple)):
        forge_outputs = [forge_outputs]

    gt = load_ground_truth_outputs(seq_id)

    failures = []
    for name, out in zip(output_names, forge_outputs):
        golden = torch.from_numpy(gt[name])
        calculated = out.detach().cpu() if hasattr(out, "detach") else torch.tensor(out)

        pcc_value = calculate_pcc(golden, calculated) if golden.numel() > 1 else None
        atol_value = calculate_atol(golden, calculated)
        passed = compare_with_golden(golden=golden, calculated=calculated, pcc=PCC)

        if pcc_value is not None:
            logger.info(
                "Output '{}': PCC={:.6f} (required={:.2f}), max|Δ|={:.3e} — {}",
                name, pcc_value, PCC, atol_value, "PASS" if passed else "FAIL",
            )
        else:
            logger.info(
                "Output '{}' (scalar): max|Δ|={:.3e} — {}",
                name, atol_value, "PASS" if passed else "FAIL",
            )

        if not passed:
            failures.append((name, pcc_value, atol_value))

    assert not failures, (
        f"[seq={seq_id}] {len(failures)}/{len(output_names)} output(s) failed "
        f"ground-truth comparison (pcc={PCC}):\n"
        + "\n".join(
            f"  {name}: PCC={pcc:.6f}, max|Δ|={atol:.3e}"
            if pcc is not None
            else f"  {name}: max|Δ|={atol:.3e}"
            for name, pcc, atol in failures
        )
    )
