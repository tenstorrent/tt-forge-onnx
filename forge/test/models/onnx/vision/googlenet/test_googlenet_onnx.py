# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from third_party.tt_forge_models.googlenet.image_classification.onnx import ModelLoader, ModelVariant


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(ModelVariant.GOOGLENET, marks=pytest.mark.pr_models_regression),
    ],
)
@pytest.mark.nightly
def test_googlenet_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GOOGLENET,
        variant=variant.value,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load inputs
    loader = ModelLoader(variant=variant)
    inputs = loader.load_inputs().contiguous()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        [inputs],
        module_name=module_name,
    )

    # Model Verification and Inference
    _, co_out = verify([inputs], framework_model, compiled_model)

    # Print classification results
    loader.print_cls_results(co_out)
