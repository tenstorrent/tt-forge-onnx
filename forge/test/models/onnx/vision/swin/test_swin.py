# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from third_party.tt_forge_models.swin.image_classification.onnx import (
    ModelLoader as SwinImgClsOnnxLoader,
    ModelVariant as SwinImgClsOnnxVariant,
)
from third_party.tt_forge_models.swin.masked_image_modeling.onnx import (
    ModelLoader as SwinMimOnnxLoader,
    ModelVariant as SwinMimOnnxVariant,
)
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [SwinImgClsOnnxVariant.SWINV2_TINY_HF])
def test_swin_v2_tiny_image_classification_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SWIN,
        variant=variant.value,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Segmentation Fault")

    # Load model and input
    loader = SwinImgClsOnnxLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", [SwinMimOnnxVariant.SWINV2_TINY])
def test_swin_v2_tiny_masked_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SWIN,
        variant=variant.value,
        task=Task.CV_MASKED_IMAGE_MODELING,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = SwinMimOnnxLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [SwinImgClsOnnxVariant.SWIN_V2_T])
def test_swin_torchvision(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SWIN,
        variant=variant.value,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    loader = SwinImgClsOnnxLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
