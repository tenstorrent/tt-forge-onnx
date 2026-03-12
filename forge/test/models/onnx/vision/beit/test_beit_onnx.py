# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.beit.model_utils.utils import load_input, load_model
import torch
import onnx

variants = [
    pytest.param("microsoft/beit-base-patch16-224", marks=[pytest.mark.xfail]),
    pytest.param(
        "microsoft/beit-large-patch16-224",
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2969")],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_beit_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BEIT,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    torch_model = load_model(variant)
    inputs = load_input(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
