# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
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
from test.models.onnx.vision.yolo.model_utils.yolov4_utils import (
    Wrapper,
)
from third_party.tt_forge_models.yolov4.pytorch import ModelLoader  # isort:skip


@pytest.mark.out_of_memory
@pytest.mark.nightly
def test_yolo_v4(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV4,
        variant="default",
        task=Task.CV_OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = ModelLoader()
    framework_model = loader.load_model()
    framework_model = Wrapper(framework_model)
    input_sample = loader.load_inputs()

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/yolov3.onnx"
    torch.onnx.export(
        framework_model,
        input_sample,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=[input_sample],
        module_name=module_name,
    )

    # Model Verification
    verify(
        [input_sample],
        onnx_module,
        compiled_model,
    )
