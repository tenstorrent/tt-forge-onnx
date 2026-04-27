# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from third_party.tt_forge_models.regnet.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


variants = [
    pytest.param(ModelVariant.Y_040, marks=pytest.mark.pr_models_regression),
    ModelVariant.Y_064,
    ModelVariant.Y_080,
    ModelVariant.Y_120,
    ModelVariant.Y_160,
    ModelVariant.Y_320,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_regnet_img_classification_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.REGNET,
        variant=variant.value,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inputs = loader.load_inputs(dtype_override=torch.float32)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Verify
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-processing
    loader.torch_loader.output_postprocess(co_out)
