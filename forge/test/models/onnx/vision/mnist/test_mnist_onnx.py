# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
import onnx
import os

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)

from test.models.pytorch.vision.mnist.model_utils.utils import load_input, load_model
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


@pytest.mark.pr_models_regression
@pytest.mark.nightly
def test_mnist(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MNIST,
        source=Source.GITHUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    framework_model = load_model()
    inputs = load_input()
    inputs = [inputs[0]]

    # Convert inputs to bfloat16
    bfloat16_inputs = convert_inputs_to_bfloat16(inputs)

    # Compile and run the PyTorch model with bfloat16 (deep-copy keeps the
    # original framework_model at float32 for the ONNX export below).
    # Compiled pytorch model output will be used as the golden reference for the compiled onnx model in bfloat16.
    pytorch_compiled_out = compile_and_run_model_pytorch(
        copy.deepcopy(framework_model),
        bfloat16_inputs,
        module_name + "_pytorch",
    )
    print(pytorch_compiled_out)

    # Export the pytorch model (float32) model to ONNX.
    onnx_path = f"{forge_tmp_path}/mnist.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

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
