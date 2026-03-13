# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from transformers import AutoImageProcessor
import pytest
import onnx
import torch
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification
from test.models.models_utils import get_sample_data
from test.utils import download_model

variants_img_classification = [
    pytest.param("nvidia/mit-b0", marks=pytest.mark.pr_models_regression),
    pytest.param("nvidia/mit-b2"),
    pytest.param("nvidia/mit-b3", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b4", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b5", marks=pytest.mark.xfail),
]


@pytest.mark.parametrize("variant", variants_img_classification)
@pytest.mark.nightly
def test_segformer_image_classification_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SEGFORMER,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    if variant in ["nvidia/mit-b3", "nvidia/mit-b4", "nvidia/mit-b5"]:
        pytest.xfail(reason="Fatal Python error: Aborted")

    # Load the model from HuggingFace
    torch_model = download_model(SegformerForImageClassification.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # prepare input
    inputs = get_sample_data(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/segformer_" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
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
    print("Predicted class: ", torch_model.config.id2label[predicted_label])


variants_semseg = [
    pytest.param("nvidia/segformer-b0-finetuned-ade-512-512", marks=pytest.mark.pr_models_regression),
    "nvidia/segformer-b1-finetuned-ade-512-512",
    pytest.param("nvidia/segformer-b2-finetuned-ade-512-512"),
    pytest.param("nvidia/segformer-b3-finetuned-ade-512-512"),
    pytest.param("nvidia/segformer-b4-finetuned-ade-512-512"),
]


@pytest.mark.parametrize("variant", variants_semseg)
@pytest.mark.nightly
def test_segformer_semantic_segmentation_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.SEGFORMER,
        variant=variant,
        task=Task.CV_IMAGE_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load the model from HuggingFace
    torch_model = download_model(SegformerForSemanticSegmentation.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # prepare input
    inputs = get_sample_data(variant)

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
