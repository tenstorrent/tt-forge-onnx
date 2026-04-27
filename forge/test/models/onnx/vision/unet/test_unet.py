# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from third_party.tt_forge_models.unet.image_segmentation.onnx import ModelLoader, ModelVariant
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.TORCHHUB_BRAIN_UNET])
def test_unet_onnx(variant, forge_tmp_path):
    # Build Module Name
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.UNET,
        variant=variant.value,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_SEGMENTATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
