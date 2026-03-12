# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from test.models.onnx.vision.sam.model_utils.model import get_model_inputs


@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/sam-vit-huge",
        ),
        pytest.param(
            "facebook/sam-vit-large",
        ),
        pytest.param("facebook/sam-vit-base", marks=pytest.mark.xfail()),
    ],
)
@pytest.mark.nightly
def test_sam_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SAM,
        variant=variant,
        task=Task.CV_IMAGE_SEGMENTATION,
        source=Source.GITHUB,
    )
    if variant != "facebook/sam-vit-base":
        pytest.xfail(reason="Requires multi-chip support")

    # Load  model and input
    framework_model, sample_inputs = get_model_inputs(variant)
    input_tensor = sample_inputs[0]
    sample_inputs = [input_tensor]

    onnx_path = f"{forge_tmp_path}/sam_" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(
        framework_model,
        input_tensor,
        onnx_path,
        input_names=["image"],
        output_names=["segmentation"],
        dynamic_axes={"image": {0: "batch_size"}},
    )

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(
        sample_inputs,
        framework_model,
        compiled_model,
    )
