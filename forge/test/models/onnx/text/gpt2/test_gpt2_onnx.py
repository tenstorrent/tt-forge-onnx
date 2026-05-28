# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from forge.config import CompilerConfig
from forge.verify.config import DeprecatedVerifyConfig

from test.utils import download_model
import onnx


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("mnoukhov/gpt2-imdb-sentiment-classifier", marks=pytest.mark.pr_models_regression),
    ],
)
@pytest.mark.parametrize("use_transpiler", [True, False], ids=["transpiler", "tvm"])
def test_gpt2_sequence_classification_onnx(variant, forge_tmp_path, use_transpiler):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.GPT,
        variant=variant,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
        suffix="_transpiler" if use_transpiler else "_tvm",
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, padding_side="left")
    torch_model = download_model(
        AutoModelForSequenceClassification.from_pretrained, variant, return_dict=False, use_cache=False
    )
    torch_model.eval()

    # Prepare input
    test_input = "This is a sample text from "
    input_tokens = tokenizer(test_input, return_tensors="pt")
    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
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
        # Set data format override
        data_format_override = forge._C.DataFormat.Float16_b
        compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)
        verify_cfg = DeprecatedVerifyConfig()

    # Compile model
    compiled_model = forge.compile(
        onnx_model, inputs, module_name=module_name, compiler_cfg=compiler_cfg, verify_cfg=verify_cfg
    )

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {torch_model.config.id2label[predicted_value]}")
