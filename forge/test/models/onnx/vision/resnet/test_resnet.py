# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.config import DeprecatedVerifyConfig
from forge.config import CompilerConfig
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from third_party.tt_forge_models.resnet.image_classification.onnx import ModelLoader, ModelVariant

variants = [
    ModelVariant.RESNET_50,
    ModelVariant.RESNET_101,
    ModelVariant.RESNET_152,
]


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.parametrize("use_transpiler", [False, True], ids=["tvm", "transpiler"])
def test_resnet_onnx(variant, forge_tmp_path, use_transpiler):

    # Record model details
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNET,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
        suffix="_transpiler" if use_transpiler else "_tvm",
    )

    # Load inputs
    loader = ModelLoader(variant=ModelVariant(variant))
    inputs = loader.load_inputs().contiguous()

    # Load framework model
    framework_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    framework_model = forge.OnnxModule(module_name, framework_model)

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
        framework_model,
        sample_inputs=[inputs],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(
        [inputs],
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99 if use_transpiler else 0.95)),
    )

    # Print classification results
    loader.print_cls_results(co_out)
