# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from third_party.tt_forge_models.ssd300_resnet50.object_detection.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_pytorch_ssd300_resnet50(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SSD300RESNET50,
        variant=ModelVariant.BASE.value,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    loader = ModelLoader(variant=ModelVariant.BASE)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model
    verify(inputs, framework_model, compiled_model)
