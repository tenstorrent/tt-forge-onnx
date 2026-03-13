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

from test.models.onnx.vision.yolo.model_utils.yolov5_utils import load_model_and_inputs

size = [
    pytest.param("n", id="yolov5n", marks=pytest.mark.pr_models_regression),
    pytest.param("s", id="yolov5s"),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param(
        "x",
        id="yolov5x",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_320x320(size, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    # Load model
    onnx_model, inputs = load_model_and_inputs(size, forge_tmp_path)

    # Create ONNX module and compile
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Set data format override
    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Compile ONNX
    compiled_model = forge.compile(
        onnx_module,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Verify
    verify(inputs, onnx_module, compiled_model)
