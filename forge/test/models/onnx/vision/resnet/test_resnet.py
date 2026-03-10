# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import random
import onnx
import torch
from datasets import load_dataset
from transformers import ResNetForImageClassification, AutoImageProcessor

import forge
import os
import copy

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.onnx.onnx_utils import (
    patch_onnx_module_bfloat16,
    convert_inputs_to_bfloat16,
    compile_onnx_initial_graph,
    compile_and_run_onnx_bfloat16,
)
from forge.verify.compare import compare_with_golden

# Cast ONNX model parameters to bfloat16 when Forge reads them.
patch_onnx_module_bfloat16()


def compile_and_run_model_pytorch(framework_model, inputs, module_name):
    framework_model = framework_model.to(torch.bfloat16)
    compiler_cfg = forge.config.CompilerConfig(default_df_override=forge.DataFormat.Float16_b)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )
    return compiled_model(inputs[0])


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
    inputs = [input_sample]

    # Convert inputs to bfloat16
    bfloat16_inputs = convert_inputs_to_bfloat16(inputs)

    # Compile and run the PyTorch model with bfloat16 (deep-copy keeps the
    # original torch_model at float32 for the ONNX export below).
    # Compiled pytorch model output will be used as the golden reference for the compiled onnx model in bfloat16.
    pytorch_compiled_out = compile_and_run_model_pytorch(
        copy.deepcopy(torch_model),
        bfloat16_inputs,
        module_name + "_pytorch",
    )
    print(pytorch_compiled_out)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/resnet50.onnx"
    torch.onnx.export(torch_model, input_sample, onnx_path, opset_version=17)

    # Load framework model
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    onnx_module_name = module_name + "_onnx"

    # Phase 1: Generate the Forge ONNX module using float32 inputs.
    # bfloat16 is not supported during TVM relay graph construction (TVM uses
    # NumPy as its backend), so we trace the graph in float32 and stop at
    # GENERATE_INITIAL_GRAPH.  The generated module is written to disk for
    # Phase 2 to reuse.
    os.environ["FORGE_RELOAD_GENERATED_MODULES"] = "1"
    compile_onnx_initial_graph(onnx_model, inputs, onnx_module_name)

    # Phase 2: Compile and run the ONNX model with bfloat16 inputs, reusing the
    # module generated in Phase 1
    onnx_compiled_out = compile_and_run_onnx_bfloat16(onnx_model, bfloat16_inputs, onnx_module_name)
    print(onnx_compiled_out)

    # Post processing
    predicted_label = pytorch_compiled_out[0].argmax(-1).item()
    print("PyTorch Compiled Model Predicted class: ", torch_model.config.id2label[predicted_label])
    predicted_label = onnx_compiled_out[0].argmax(-1).item()
    print("ONNX Compiled Model Predicted class: ", torch_model.config.id2label[predicted_label])
