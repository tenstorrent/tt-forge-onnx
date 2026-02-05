# # SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
# #
# # SPDX-License-Identifier: Apache-2.0
# import pytest
# from third_party.tt_forge_models.qwen_2_5.casual_lm.pytorch import (
#     ModelLoader as CausalLMLoader,
# )
# from third_party.tt_forge_models.qwen_2_5.casual_lm.pytorch import (
#     ModelVariant as CausalLMVariant,
# )

# import forge
# from forge.forge_property_utils import (
#     Framework,
#     ModelArch,
#     ModelGroup,
#     ModelPriority,
#     Source,
#     Task,
#     record_model_properties,
# )
# from forge.verify.verify import verify

# from test.models.models_utils import TextModelWrapper

# # Variants for testing
# variants = [
#     # CausalLMVariant.QWEN_2_5_0_5B,
#     CausalLMVariant.QWEN_2_5_0_5B_INSTRUCT,
#     CausalLMVariant.QWEN_2_5_1_5B,
#     # CausalLMVariant.QWEN_2_5_1_5B_INSTRUCT,
#     # pytest.param(CausalLMVariant.QWEN_2_5_3B, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_3B_INSTRUCT, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_7B, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_7B_INSTRUCT, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_7B_INSTRUCT_1M, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_14B_INSTRUCT, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_14B_INSTRUCT_1M, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_32B_INSTRUCT, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_72B_INSTRUCT, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_MATH_7B, marks=[pytest.mark.out_of_memory]),
#     # pytest.param(CausalLMVariant.QWEN_2_5_14B, marks=[pytest.mark.out_of_memory]),
# ]


# @pytest.mark.parametrize("variant", variants)
# @pytest.mark.xfail
# @pytest.mark.nightly
# def test_qwen_clm(variant):
#     if variant in [
#         CausalLMVariant.QWEN_2_5_0_5B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_1_5B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_3B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_7B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_7B_INSTRUCT_1M,
#         CausalLMVariant.QWEN_2_5_14B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_14B_INSTRUCT_1M,
#         CausalLMVariant.QWEN_2_5_32B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_72B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_MATH_7B,
#         CausalLMVariant.QWEN_2_5_14B,
#     ]:
#         group = ModelGroup.RED
#         priority = ModelPriority.P1
#     else:
#         group = ModelGroup.GENERALITY
#         priority = ModelPriority.P2

#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.QWENV2,
#         variant=variant,
#         task=Task.NLP_CAUSAL_LM,
#         source=Source.HUGGINGFACE,
#         group=group,
#         priority=priority,
#     )

#     if variant in [
#         CausalLMVariant.QWEN_2_5_3B,
#         CausalLMVariant.QWEN_2_5_3B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_7B,
#         CausalLMVariant.QWEN_2_5_7B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_14B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_32B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_72B_INSTRUCT,
#         CausalLMVariant.QWEN_2_5_7B_INSTRUCT_1M,
#         CausalLMVariant.QWEN_2_5_14B_INSTRUCT_1M,
#         CausalLMVariant.QWEN_2_5_MATH_7B,
#         CausalLMVariant.QWEN_2_5_14B,
#     ]:
#         pytest.xfail(reason="Requires multi-chip support")

#     # Load Model and inputs
#     loader = CausalLMLoader(variant=variant)
#     model = loader.load_model()
#     model.config.use_cache = False
#     framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
#     framework_model.eval()
#     input_dict = loader.load_inputs()
#     inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

#     # Forge compile framework model
#     compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

#     # Model Verification and Inference
#     verify(inputs, framework_model, compiled_model)


# variants = [
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     "Qwen/Qwen2.5-VL-72B-Instruct",
#     "Qwen/QVQ-72B-Preview",
# ]


# @pytest.mark.parametrize("variant", variants)
# @pytest.mark.nightly
# @pytest.mark.xfail
# def test_qwen2_conditional_generation(variant):

#     # Record Forge Property
#     record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.QWENV2,
#         variant=variant,
#         task=Task.CONDITIONAL_GENERATION,
#         source=Source.HUGGINGFACE,
#         group=ModelGroup.RED,
#         priority=ModelPriority.P1,
#     )

#     pytest.xfail(reason="Requires multi-chip support")
