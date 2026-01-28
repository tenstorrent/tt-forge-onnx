# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import onnx
import torch
from datasets import load_dataset
from transformers import ResNetForImageClassification, AutoImageProcessor

import forge
from forge.verify.config import DeprecatedVerifyConfig
from forge.config import CompilerConfig
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
@pytest.mark.parametrize("use_transpiler", [False, True], ids=["tvm", "transpiler"])
def test_resnet_onnx(variant, forge_tmp_path, use_transpiler):
    random.seed(0)

    # Record model details
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNET,
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
        suffix="_transpiler" if use_transpiler else "_tvm",
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

    # Configure compiler and verification based on compilation path
    if use_transpiler:
        # Transpiler path configuration
        compiler_cfg = CompilerConfig(
            compile_transpiler_to_python=True,  # Enable transpiler path
            compile_tvm_to_python=False,  # Disable TVM path
            transpiler_enable_debug=True,  # Enable debug mode for transpiler (ONNX Runtime comparison)
        )

        # Create verify config with all verification flags enabled for transpiler
        verify_cfg = DeprecatedVerifyConfig(
            # Transpiler-specific verification
            verify_transpiler_graph=True,  # Compare Framework output vs TIR graph output after transpiler conversion
            verify_forge_codegen_vs_framework=True,  # Compare Framework output vs Forge codegen outputs
        )
    else:
        # TVM path configuration (default)
        compiler_cfg = CompilerConfig()
        verify_cfg = DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True)

    # Compile model
    input_sample = [input_sample]
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

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
