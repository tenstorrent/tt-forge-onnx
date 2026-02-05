# # SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# #
# # SPDX-License-Identifier: Apache-2.0
# import pytest
# import torch
# from third_party.tt_forge_models.qwen_3.causal_lm.pytorch import (
#     ModelLoader as CausalLMLoader,
# )
# from third_party.tt_forge_models.qwen_3.causal_lm.pytorch import (
#     ModelVariant as CausalLMVariant,
# )
# from third_party.tt_forge_models.qwen_3.embedding.pytorch import (
#     ModelLoader as EmbeddingLoader,
# )
# from third_party.tt_forge_models.qwen_3.embedding.pytorch import (
#     ModelVariant as EmbeddingVariant,
# )

# import forge
# from forge._C import DataFormat
# from forge.config import CompilerConfig
# from forge.forge_property_utils import (
#     Framework,
#     ModelArch,
#     ModelGroup,
#     ModelPriority,
#     Source,
#     Task,
#     record_model_properties,
# )
# from forge.verify.config import VerifyConfig
# from forge.verify.value_checkers import AutomaticValueChecker
# from forge.verify.verify import verify

# from test.models.models_utils import TextModelWrapper

# causal_lm_variants = [
#     pytest.param(CausalLMVariant.QWEN_3_32B, marks=[pytest.mark.skip_model_analysis]),
#     pytest.param(CausalLMVariant.QWEN_3_30B_A3B, marks=[pytest.mark.skip_model_analysis]),
#     pytest.param(CausalLMVariant.QWQ_32B, marks=[pytest.mark.skip_model_analysis]),
#     pytest.param(CausalLMVariant.QWEN_3_14B, marks=[pytest.mark.skip_model_analysis]),
#     CausalLMVariant.QWEN_3_0_6B,
#     CausalLMVariant.QWEN_3_1_7B,
#     CausalLMVariant.QWEN_3_4B,
#     pytest.param(CausalLMVariant.QWEN_3_8B, marks=[pytest.mark.out_of_memory]),
# ]


# @pytest.mark.parametrize("variant", causal_lm_variants)
# @pytest.mark.nightly
# @pytest.mark.xfail
# def test_qwen3_clm_pytorch(variant):

#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.QWENV3,
#         variant=variant,
#         task=Task.NLP_CAUSAL_LM,
#         source=Source.HUGGINGFACE,
#         group=ModelGroup.RED,
#         priority=ModelPriority.P1,
#     )

#     if variant in [
#         CausalLMVariant.QWEN_3_4B,
#         CausalLMVariant.QWEN_3_8B,
#         CausalLMVariant.QWEN_3_14B,
#         CausalLMVariant.QWEN_3_32B,
#         CausalLMVariant.QWEN_3_30B_A3B,
#         CausalLMVariant.QWQ_32B,
#     ]:
#         pytest.xfail(reason="Requires multi-chip support")

#     # Load Model and inputs using loader
#     loader = CausalLMLoader(variant=variant)
#     model = loader.load_model()
#     model.config.use_cache = False
#     framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
#     framework_model.eval()
#     input_dict = loader.load_inputs()
#     sample_inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

#     # Forge compile framework model
#     compiled_model = forge.compile(framework_model, sample_inputs, module_name=module_name)

#     # Model Verification and Inference
#     _, co_out = verify(sample_inputs, framework_model, compiled_model)


# embedding_variants = [
#     EmbeddingVariant.QWEN_3_EMBEDDING_0_6B,
#     EmbeddingVariant.QWEN_3_EMBEDDING_4B,
#     pytest.param(EmbeddingVariant.QWEN_3_EMBEDDING_8B, marks=[pytest.mark.out_of_memory]),
# ]


# @pytest.mark.nightly
# @pytest.mark.parametrize("variant", embedding_variants)
# def test_qwen3_embedding(variant):

#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.QWENV3,
#         variant=variant,
#         task=Task.NLP_TEXT_ENCODING,
#         source=Source.HUGGINGFACE,
#         group=ModelGroup.GENERALITY if variant == EmbeddingVariant.QWEN_3_EMBEDDING_0_6B else ModelGroup.RED,
#         priority=ModelPriority.P2 if variant == EmbeddingVariant.QWEN_3_EMBEDDING_0_6B else ModelPriority.P1,
#     )

#     if variant in [EmbeddingVariant.QWEN_3_EMBEDDING_4B, EmbeddingVariant.QWEN_3_EMBEDDING_8B]:
#         pytest.xfail(reason="Requires multi-chip support")

#     # Load Model and inputs using loader
#     loader = EmbeddingLoader(variant=variant)
#     framework_model = loader.load_model(torch.bfloat16)
#     framework_model.config.use_cache = False
#     framework_model.config.return_dict = False
#     input_dict = loader.load_inputs()
#     inputs = [input_dict["input_ids"]]

#     data_format_override = DataFormat.Float16_b
#     compiler_cfg = CompilerConfig(default_df_override=data_format_override)

#     # Forge compile framework model
#     compiled_model = forge.compile(
#         framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
#     )

#     # Model Verification and Inference
#     _, co_out = verify(
#         inputs,
#         framework_model,
#         compiled_model,
#         verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98)),
#     )

#     # Post processing
#     outputs = co_out[0]
#     print("Similarity scores:", loader.decode_output(outputs, input_dict))
