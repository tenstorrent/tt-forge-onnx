# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
import torch
import onnx
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.models_utils import print_cls_results
from test.utils import download_model

varaints = [
    pytest.param(
        "mixer_b16_224",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "mixer_b16_224_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_b16_224_miil"),
    pytest.param(
        "mixer_b16_224_miil_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_b32_224"),
    pytest.param(
        "mixer_l16_224",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "mixer_l16_224_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_l32_224"),
    pytest.param(
        "mixer_s16_224",
        marks=pytest.mark.pr_models_regression,
    ),
    pytest.param("mixer_s32_224"),
    pytest.param(
        "mixer_b16_224.goog_in21k",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MLPMIXER,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    load_pretrained_weights = True
    if variant in ["mixer_s32_224", "mixer_s16_224", "mixer_b32_224", "mixer_l32_224"]:
        load_pretrained_weights = False

    framework_model = download_model(timm.create_model, variant, pretrained=load_pretrained_weights)
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    try:
        if variant in [
            "mixer_b16_224_in21k",
            "mixer_b16_224_miil_in21k",
            "mixer_l16_224_in21k",
            "mixer_b16_224.goog_in21k",
        ]:
            input_image = get_file(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
            use_1k_labels = False
        else:
            input_image = get_file(
                "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
            )
            use_1k_labels = True
        image = Image.open(str(input_image)).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)
    pixel_values = transform(image).unsqueeze(0)

    inputs = [pixel_values]
    pcc = 0.99
    if variant == "mixer_b16_224_miil":
        pcc = 0.95

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}_mlpmixer.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=True,
    )

    # Load and verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify model
    fw_out, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Print classification results
    print_cls_results(fw_out[0], co_out[0], use_1k_labels=use_1k_labels)
