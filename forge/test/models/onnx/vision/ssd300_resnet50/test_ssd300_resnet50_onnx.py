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

from test.models.onnx.vision.ssd300_resnet50.model_utils.utils import (
    load_ssd300_torch_model,
    load_ssd300_inputs,
)
from test.models.onnx.vision.vision_utils.utils import (
    load_onnx_with_fallback,
)


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_pytorch_ssd300_resnet50(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SSD300RESNET50,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Prepare input
    inputs = load_ssd300_inputs()

    # Load ONNX model
    onnx_model = load_onnx_with_fallback(
        torch_model_loader=load_ssd300_torch_model,
        s3_onnx_path="test_files/onnx/ssd300_resnet50/ssd300_resnet50.onnx",
        onnx_filename="ssd300_resnet50.onnx",
        forge_tmp_path=forge_tmp_path,
        inputs=inputs,
    )
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model
    verify(inputs, framework_model, compiled_model)
