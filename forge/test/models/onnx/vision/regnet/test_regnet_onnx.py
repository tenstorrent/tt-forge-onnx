# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from pathlib import Path

from third_party.tt_forge_models.regnet.pytorch import ModelLoader, ModelVariant

import forge
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
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input using tt_forge_models
    loader = ModelLoader(variant=variant)
    torch_model = loader.load_model(dtype_override=torch.float32)
    input_tensor = loader.load_inputs(dtype_override=torch.float32)
    inputs = [input_tensor]

    # Export ONNX model from PyTorch
    onnx_path = Path(forge_tmp_path) / f"{variant.value}_exported.onnx"
    if not onnx_path.exists():
        torch.onnx.export(
            torch_model,
            input_tensor,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    # Load ONNX model properly
    model_name = f"onnx_regnet_{variant.value}"
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-processing
    loader.output_postprocess(co_out)
