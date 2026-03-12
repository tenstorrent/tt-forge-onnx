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

from test.models.onnx.vision.mobilenet.model_utils.utils import (
    load_mobilenet_model,
    post_processing,
)
import onnx
import torch

variants = [pytest.param("mobilenet_v3_small", marks=pytest.mark.pr_models_regression)]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mobilenetv3_basic(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV3,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    torch_model, inputs = load_mobilenet_model(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Set data format override
    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name, compiler_cfg=compiler_cfg)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)
