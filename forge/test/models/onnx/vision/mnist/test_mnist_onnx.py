# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from third_party.tt_forge_models.mnist.image_classification.onnx import ModelLoader, ModelVariant
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.models_utils import print_cls_results


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_mnist(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MNIST,
        variant=ModelVariant.CNN_DROPOUT.value,
        source=Source.GITHUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input (float32: bfloat16 tensors cannot be converted to NumPy in forge.compile's ONNX path)
    loader = ModelLoader(variant=ModelVariant.CNN_DROPOUT)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path, dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Set data format override
    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model verification and inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )

    print_cls_results(fw_out[0], co_out[0])
