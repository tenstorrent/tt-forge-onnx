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

from test.models.onnx.vision.vision_utils.utils import (
    load_imagenet_inputs,
    load_onnx_with_fallback,
    load_torch_hub_model,
    post_processing,
)


variants = [
    pytest.param("wide_resnet50_2", marks=pytest.mark.pr_models_regression, id="wide_resnet50_2"),
    "wide_resnet101_2",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_wideresnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WIDERESNET,
        source=Source.TIMM,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load input data
    inputs = load_imagenet_inputs()

    # Load ONNX model
    safe_name = variant.replace(".", "_")
    onnx_model = load_onnx_with_fallback(
        torch_model_loader=lambda: load_torch_hub_model(variant),
        s3_onnx_path=f"test_files/onnx/wideresnet/{safe_name}.onnx",
        onnx_filename=f"{safe_name}.onnx",
        forge_tmp_path=forge_tmp_path,
        inputs=inputs,
    )
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    post_processing(co_out)
