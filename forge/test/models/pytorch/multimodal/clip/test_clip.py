# # SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# # SPDX-License-Identifier: Apache-2.0
# import pytest
# from third_party.tt_forge_models.clip.pytorch.loader import ModelLoader, ModelVariant

# import forge
# from forge.forge_property_utils import (
#     Framework,
#     ModelArch,
#     Source,
#     Task,
#     record_model_properties,
# )
# from forge.verify.verify import verify

# from test.models.pytorch.multimodal.clip.model_utils.clip_model import CLIPTextWrapper

# variants = [
#     ModelVariant.CLIP_VIT_BASE_PATCH32,
# ]


# @pytest.mark.nightly
# @pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2998")
# @pytest.mark.parametrize("variant", variants)
# def test_clip_pytorch(variant):
#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.CLIP,
#         variant=variant.value,
#         suffix="text",
#         source=Source.HUGGINGFACE,
#         task=Task.NLP_CAUSAL_LM,
#     )

#     # Load model and inputs using ModelLoader
#     loader = ModelLoader(variant=variant)
#     model = loader.load_model()
#     framework_model = CLIPTextWrapper(model)
#     inputs_dict = loader.load_inputs()

#     # Extract inputs in the format expected by CLIPTextWrapper
#     input_ids = inputs_dict["input_ids"]
#     attention_mask = inputs_dict["attention_mask"]
#     inputs = [input_ids, attention_mask]

#     # Forge compile framework model
#     compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

#     # Model Verification
#     fw_out, co_out = verify(inputs, framework_model, compiled_model)

#     # Model Postprocessing
#     loader.print_cls_results(co_out)
