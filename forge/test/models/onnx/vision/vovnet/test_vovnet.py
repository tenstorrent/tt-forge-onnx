# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.verify.verify import verify

from test.models.onnx.vision.vovnet.model_utils.model_utils import (
    get_image,
    preprocess_steps,
    preprocess_timm_model,
)
from test.models.onnx.vision.vovnet.model_utils.src_vovnet_stigma import vovnet39, vovnet57
from forge.forge_property_utils import (
    Framework,
    Source,
    Task,
    ModelArch,
    record_model_properties,
    ModelGroup,
    ModelPriority,
)
from test.utils import download_model


def generate_model_vovnet_imgcls_osmr_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    image_tensor = get_image()

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["vovnet27s"])
def test_vovnet_osmr_pytorch(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.OSMR,
        task=Task.CV_OBJECT_DETECTION,
    )
    pytest.xfail(reason="Segmentation Fault")

    # Load model and inputs
    framework_model, inputs, _ = generate_model_vovnet_imgcls_osmr_pytorch(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/vovnet_osmr.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)


def generate_model_vovnet39_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["vovnet39"])
def test_vovnet_v1_39_stigma_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VOVNETV1,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
    )
    pytest.xfail(reason="Segmentation Fault")

    framework_model, inputs, _ = generate_model_vovnet39_imgcls_stigma_pytorch()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/vovnet_v1_39.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)


def generate_model_vovnet57_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet57)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["vovnet_v1_57"])
def test_vovnet_v1_57_stigma_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
    )
    pytest.xfail(reason="Segmentation Fault")

    framework_model, inputs, _ = generate_model_vovnet57_imgcls_stigma_pytorch()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/vovnet_v1_57.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)


def generate_model_vovnet_imgcls_timm_pytorch(variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)

    return model, [image_tensor], {}


@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2974")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["ese_vovnet19b_dw.ra_in1k"])
def test_vovnet_timm_onnx(variant, forge_tmp_path):

    if variant == "ese_vovnet19b_dw.ra_in1k":
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
        group=group,
        priority=priority,
    )

    pytest.xfail(reason="Build failed with ttnn-dylib option")

    framework_model, inputs, _ = generate_model_vovnet_imgcls_timm_pytorch(
        variant,
    )

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/vovnet_timm.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
