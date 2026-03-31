# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import timm
import forge
import onnx
import torch
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from test.models.onnx.vision.mobilenet.model_utils.utils import load_inputs
from test.models.models_utils import print_cls_results
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from third_party.tt_forge_models.mobilenetv2.pytorch import ModelLoader, ModelVariant

params = [
    pytest.param("mobilenetv2_050"),
    pytest.param("mobilenetv2_100", marks=[pytest.mark.pr_models_regression]),
    pytest.param("mobilenetv2_110d"),
    pytest.param("mobilenetv2_140"),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_mobilenetv2_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV2,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load mobilenetv2 model
    model = timm.create_model(variant, pretrained=True)

    # Load the inputs
    inputs = load_inputs(model)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(model, inputs[0], onnx_path, opset_version=17)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99
    if variant == "mobilenetv2_050":
        pcc = 0.95

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.MOBILENET_V2_TORCHVISION])
def test_mobilenetv2_onnx_torchvision(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV2,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    torch_model = loader.load_model()
    input_tensor = loader.load_inputs().contiguous()
    inputs = [input_tensor]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/mobilenetv2_torchvision.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)
