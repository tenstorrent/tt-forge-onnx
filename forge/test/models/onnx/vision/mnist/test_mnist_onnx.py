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
from forge.verify.config import DeprecatedVerifyConfig
from forge.config import CompilerConfig

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.mnist.model_utils.utils import load_input, load_model


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("use_transpiler", [False, True], ids=["tvm", "transpiler"])
def test_mnist(forge_tmp_path, use_transpiler):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MNIST,
        source=Source.GITHUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Add suffix based on compilation path
    path_suffix = "_transpiler" if use_transpiler else "_tvm"
    module_name = module_name + path_suffix

    # Load model and input
    framework_model = load_model()
    inputs = load_input()
    inputs = [inputs[0]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/mnist.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and check ONNX model
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

    # Forge compile ONNX model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Model verification and inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post Processing
    print_cls_results(fw_out[0], co_out[0])
