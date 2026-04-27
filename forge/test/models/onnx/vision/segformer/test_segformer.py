# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from third_party.tt_forge_models.segformer.image_classification.onnx import (
    ModelLoader as SegformerImgClsOnnxLoader,
    ModelVariant as SegformerImgClsVariant,
)
from third_party.tt_forge_models.segformer.semantic_segmentation.onnx import (
    ModelLoader as SegformerSemSegOnnxLoader,
    ModelVariant as SegformerSemSegVariant,
)
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants_img_classification = [
    pytest.param(SegformerImgClsVariant.MIT_B0, marks=pytest.mark.pr_models_regression),
    pytest.param(SegformerImgClsVariant.MIT_B2),
    pytest.param(SegformerImgClsVariant.MIT_B3, marks=pytest.mark.xfail),
    pytest.param(SegformerImgClsVariant.MIT_B4, marks=pytest.mark.xfail),
    pytest.param(SegformerImgClsVariant.MIT_B5, marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("variant", variants_img_classification)
@pytest.mark.nightly
def test_segformer_image_classification_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SEGFORMER,
        variant=variant.value,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    if variant in (
        SegformerImgClsVariant.MIT_B3,
        SegformerImgClsVariant.MIT_B4,
        SegformerImgClsVariant.MIT_B5,
    ):
        pytest.xfail(reason="Fatal Python error: Aborted")

    # Load model and input
    loader = SegformerImgClsOnnxLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Set data format override
    data_format_override = forge._C.DataFormat.Float16_b
    compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name, compiler_cfg=compiler_cfg)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    logits = co_out[0]
    predicted_label = logits.argmax(-1).item()
    print("Predicted class: ", loader.torch_loader.model.config.id2label[predicted_label])


variants_semseg = [
    pytest.param(SegformerSemSegVariant.B0_FINETUNED, marks=pytest.mark.pr_models_regression),
    SegformerSemSegVariant.B1_FINETUNED,
    pytest.param(SegformerSemSegVariant.B2_FINETUNED),
    pytest.param(SegformerSemSegVariant.B3_FINETUNED),
    pytest.param(SegformerSemSegVariant.B4_FINETUNED),
]


@pytest.mark.parametrize("variant", variants_semseg)
@pytest.mark.nightly
def test_segformer_semantic_segmentation_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SEGFORMER,
        variant=variant.value,
        task=Task.CV_IMAGE_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = SegformerSemSegOnnxLoader(variant=variant)
    onnx_model = loader.load_model(onnx_tmp_path=forge_tmp_path)
    inp = loader.load_inputs()
    inputs = [inp] if isinstance(inp, torch.Tensor) else inp
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
