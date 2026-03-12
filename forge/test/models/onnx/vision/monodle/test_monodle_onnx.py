# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.monodle.model_utils.model import CenterNet3D
import onnx
from test.models.models_utils import preprocess_inputs


@pytest.mark.nightly
@pytest.mark.xfail
def test_monodle_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX, model=ModelArch.MONODLE, source=Source.TORCHVISION, task=Task.CV_OBJECT_DETECTION
    )

    pytest.xfail(reason="Fatal Python error: Floating point exception")

    # Load input
    inputs = preprocess_inputs()

    # Load Model
    torch_model = CenterNet3D(backbone="dla34")
    torch_model.eval()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/monodle.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
