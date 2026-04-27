# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from third_party.tt_forge_models.retinanet.object_detection.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [
    ModelVariant.RETINANET_RN18FPN,
    ModelVariant.RETINANET_RN34FPN,
    ModelVariant.RETINANET_RN50FPN,
    ModelVariant.RETINANET_RN101FPN,
    ModelVariant.RETINANET_RN152FPN,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RETINANET,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.CV_OBJECT_DETECTION,
    )

    loader = ModelLoader(variant=variant)
    try:
        onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
        inp = loader.load_inputs()
        inputs = [inp] if isinstance(inp, torch.Tensor) else inp
        framework_model = forge.OnnxModule(module_name, onnx_model)

        compiled_model = forge.compile(
            onnx_model,
            sample_inputs=inputs,
            module_name=module_name,
        )

        verify(inputs, framework_model, compiled_model)
    finally:
        if getattr(loader, "torch_loader", None) is not None:
            loader.torch_loader.cleanup()
