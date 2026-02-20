# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
import forge
import onnx
import torch
from datasets import load_dataset
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from test.models.onnx.vision.vision_utils import load_inputs
from test.models.models_utils import print_cls_results
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from efficientnet_pytorch import EfficientNet
from third_party.tt_forge_models.tools.utils import get_file
from PIL import Image
from torchvision import transforms

params = [
    pytest.param("efficientnet_b0", marks=[pytest.mark.pr_models_regression]),
    pytest.param("efficientnet_b1"),
    pytest.param("efficientnet_b2"),
    pytest.param("efficientnet_b2a"),
    pytest.param("efficientnet_b3"),
    pytest.param("efficientnet_b3a"),
    pytest.param("efficientnet_b4"),
    pytest.param("efficientnet_b5"),
    pytest.param("efficientnet_lite0"),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_efficientnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.EFFICIENTNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )
    if variant == "efficientnet_b5":
        pytest.xfail(reason="Requires multi-chip support")

    # Load efficientnet model
    model = timm.create_model(variant, pretrained=True)

    # Load the inputs
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    img = next(iter(dataset.skip(10)))["image"]
    inputs = load_inputs(img, model)
    onnx_path = f"{forge_tmp_path}/efficientnet.onnx"
    torch.onnx.export(model, inputs[0], onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99

    if variant == "efficientnet_b1":
        pcc = 0.95

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(
            value_checker=AutomaticValueChecker(pcc=pcc),
        ),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


variants = [
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    pytest.param("efficientnet-b6"),
    pytest.param("efficientnet-b7"),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_efficientnet_onnx_export_from_package(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.EFFICIENTNET,
        variant=variant,
        source=Source.GITHUB,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model
    torch_model = EfficientNet.from_pretrained(variant)
    torch_model.set_swish(memory_efficient=False)
    torch_model.eval()

    # Prepare Input
    image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    img = Image.open(str(image_file))
    tfms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = tfms(img).unsqueeze(0)
    inputs = [input_tensor]

    # Export to ONNX
    onnx_variant = variant.replace("-", "_")
    onnx_path = f"{forge_tmp_path}/{onnx_variant}.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model verification and inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
