# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import os
from third_party.tt_forge_models.dla.pytorch import ModelLoader, ModelVariant
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify


variants = [
    ModelVariant.DLA34,
    ModelVariant.DLA46_C,
    ModelVariant.DLA46X_C,
    ModelVariant.DLA60,
    pytest.param(ModelVariant.DLA60X, marks=pytest.mark.pr_models_regression),
    ModelVariant.DLA60X_C,
    ModelVariant.DLA102,
    ModelVariant.DLA102X,
    ModelVariant.DLA102X2,
    ModelVariant.DLA169,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_onnx(variant, tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DLA,
        variant=variant,
        task=Task.CV_IMAGE_ENCODING,
        source=Source.TIMM,
    )

    # Load model and input using tt_forge_models
    loader = ModelLoader(variant=variant)
    torch_model = loader.load_model()
    input_tensor = loader.load_inputs()
    inputs = [input_tensor]

    # Export ONNX
    onnx_path = tmp_path / f"{variant}_exported.onnx"
    if not os.path.exists(onnx_path):
        torch.onnx.export(
            torch_model,
            input_tensor,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    # Load exported ONNX model
    model_name = f"onnx_dla_{variant}"
    onnx_model = onnx.load(str(onnx_path))
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Compile
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Verify
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-processing
    loader.print_cls_results(co_out)
