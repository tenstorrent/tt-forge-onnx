# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import VerifyConfig, verify
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.onnx.vision.monodepth2.model_utils.utils import (
    download_model,
    load_input,
    load_model,
    postprocess_and_save_disparity_map,
)

variants = [
    pytest.param("mono_640x192"),
    pytest.param("stereo_640x192"),
    pytest.param("mono+stereo_640x192", marks=pytest.mark.pr_models_regression),
    pytest.param("mono_no_pt_640x192"),
    pytest.param("stereo_no_pt_640x192"),
    pytest.param("mono+stereo_no_pt_640x192"),
    pytest.param("mono_1024x320"),
    pytest.param("stereo_1024x320"),
    pytest.param("mono+stereo_1024x320"),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_monodepth2(variant, forge_tmp_path):

    pcc = 0.99
    if variant in [
        "mono_640x192",
        "stereo_640x192",
        "mono+stereo_640x192",
        "mono_no_pt_640x192",
        "stereo_no_pt_640x192",
    ]:
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MONODEPTH2,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.CV_DEPTH_ESTIMATION,
    )

    # prepare model and input
    download_model(variant)
    framework_model, height, width = load_model(variant)
    input_tensor, original_width, original_height = load_input(height, width)

    inputs = [input_tensor]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}_monodepth2.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post-process and save result
    postprocess_and_save_disparity_map(co_out, original_height, original_width, variant)
