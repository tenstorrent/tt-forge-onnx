# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.onnx.vision.glpn_kitti.model_utils.utils import (
    load_input,
    load_model,
)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "vinvino02/glpn-kitti",
        ),
    ],
)
def test_glpn_kitti(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GLPNKITTI,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_DEPTH_ESTIMATION,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    framework_model = load_model(variant)
    inputs = load_input(variant)
    inputs = [inputs[0]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/linear_ae.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
