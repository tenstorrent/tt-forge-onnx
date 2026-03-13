# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import onnx
import torch
from datasets import load_dataset
from transformers import ResNetForImageClassification, AutoImageProcessor

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = [
    "microsoft/resnet-50",
]


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet_onnx(variant, forge_tmp_path):
    random.seed(0)

    # Record model details
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNET,
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load processor and Model
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    torch_model = ResNetForImageClassification.from_pretrained(variant)

    # Prepare input
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    inputs = processor(image, return_tensors="pt")
    input_sample = inputs["pixel_values"]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/resnet50.onnx"
    torch.onnx.export(torch_model, input_sample, onnx_path, opset_version=17)

    # Load framework model
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Compile model
    input_sample = [input_sample]
    compiled_model = forge.compile(onnx_model, input_sample, module_name=module_name, compiler_cfg=compiler_cfg)

    # Model Verification and Inference
    _, co_out = verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Post processing
    predicted_label = co_out[0].argmax(-1).item()
    print("Predicted class: ", torch_model.config.id2label[predicted_label])
