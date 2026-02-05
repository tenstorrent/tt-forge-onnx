# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch
from datasets import load_dataset

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant  # isort:skip


@pytest.mark.nightly
def test_resnet_hf():
    random.seed(0)

    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load tiny dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")
    images = random.sample(dataset["valid"]["image"], 10)

    # Load model and inputs
    loader = ModelLoader()
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model.config.return_dict = False
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    input_sample = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    compiled_model = forge.compile(
        framework_model,
        input_sample,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Run model on sample data and print results
    loader.post_process(
        framework_model=framework_model, compiled_model=compiled_model, inputs=images, dtype_override=torch.bfloat16
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.RESNET_50_TIMM])
def test_resnet_timm(variant):
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        source=Source.TIMM,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    input_sample = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Run model on sample data and print results
    loader.post_process(co_out)


variants = [
    # HuggingFace variant
    ModelVariant.RESNET_50_HF,
    # TIMM variant
    ModelVariant.RESNET_50_TIMM,
    ModelVariant.RESNET_18,
    ModelVariant.RESNET_34,
    ModelVariant.RESNET_50,
    ModelVariant.RESNET_101,
    ModelVariant.RESNET_152,
]


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_resnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Run inference on Tenstorrent device
    _ = compiled_model(input_tensor)

    # pcc = 0.99
    # if variant == ModelVariant.RESNET_34:
    #     pcc = 0.98
    # elif variant in [ModelVariant.RESNET_50, ModelVariant.RESNET_152]:
    #     pcc = 0.95

    # # Model Verification and Inference
    # _, co_out = verify(
    #     inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    # )

    # # Run model on sample data and print results
    # loader.post_process(co_out)
