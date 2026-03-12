# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
from third_party.tt_forge_models.alexnet.image_classification.onnx import ModelLoader, ModelVariant


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_alexnet_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.ALEXNET,
        variant=ModelVariant.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader(variant=ModelVariant.ALEXNET)
    inputs = loader.load_inputs().contiguous()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

    # Compile model
    compiled_model = forge.compile(framework_model, [inputs], module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        [inputs],
        framework_model,
        compiled_model,
    )

    # Print classification results
    loader.print_cls_results(co_out)
