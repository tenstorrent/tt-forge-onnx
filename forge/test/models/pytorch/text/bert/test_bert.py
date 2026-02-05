# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.bert.masked_lm.pytorch.loader import (
    ModelLoader as MaskedLMLoader,
)
from third_party.tt_forge_models.bert.masked_lm.pytorch.loader import (
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.bert.question_answering.pytorch.loader import (
    ModelLoader as QuestionAnsweringLoader,
)
from third_party.tt_forge_models.bert.question_answering.pytorch.loader import (
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import (
    ModelLoader as SentenceEmbeddingGenerationLoader,
)
from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import (
    ModelVariant as SentenceEmbeddingGenerationVariant,
)
from third_party.tt_forge_models.bert.sequence_classification.pytorch.loader import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.bert.sequence_classification.pytorch.loader import (
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.bert.token_classification.pytorch.loader import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.bert.token_classification.pytorch.loader import (
    ModelVariant as TokenClassificationVariant,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize("variant", [MaskedLMVariant.BERT_BASE_UNCASED])
def test_bert_masked_lm_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant.value,
        task=Task.NLP_MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = MaskedLMLoader()
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Run inference on Tenstorrent device
    _ = compiled_model(*inputs)

    # # Model Verification and Inference
    # _, co_out = verify(
    #     inputs,
    #     framework_model,
    #     compiled_model,
    # )

    # # Post processing
    # loader.decode_output(co_out)


variants = [
    QuestionAnsweringVariant.PHIYODR_BERT_LARGE_FINETUNED_SQUAD2,
    QuestionAnsweringVariant.BERT_LARGE_CASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD,
]


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_bert_question_answering_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant.value,
        task=Task.NLP_QA,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = QuestionAnsweringLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Run inference on Tenstorrent device
    _ = compiled_model(*inputs)

    # verify_cfg = VerifyConfig()
    # if variant == QuestionAnsweringVariant.PHIYODR_BERT_LARGE_FINETUNED_SQUAD2:
    #     verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))

    # # Model Verification and Inference
    # _, co_out = verify(
    #     inputs,
    #     framework_model,
    #     compiled_model,
    #     verify_cfg=verify_cfg,
    # )

    # # Post processing
    # loader.decode_output(co_out)


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize("variant", [SequenceClassificationVariant.TEXTATTACK_BERT_BASE_UNCASED_SST_2])
def test_bert_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant.value,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = SequenceClassificationLoader()
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Run inference on Tenstorrent device
    _ = compiled_model(*inputs)

    # # Model Verification and Inference
    # _, co_out = verify(inputs, framework_model, compiled_model)

    # # Post processing
    # loader.decode_output(co_out)


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize("variant", [TokenClassificationVariant.DBMDZ_BERT_LARGE_CASED_FINETUNED_CONLL03_ENGLISH])
def test_bert_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant.value,
        task=Task.NLP_TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model using the new loader
    loader = TokenClassificationLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Run inference on Tenstorrent device
    _ = compiled_model(*inputs)

    # # Model Verification
    # _, co_out = verify(
    #     inputs,
    #     framework_model,
    #     compiled_model,
    # )

    # # Post processing
    # loader.decode_output(co_out)


@pytest.mark.demos_tests
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant", [SentenceEmbeddingGenerationVariant.EMRECAN_BERT_BASE_TURKISH_CASED_MEAN_NLI_STSB_TR]
)
def test_bert_sentence_embedding_generation_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant.value,
        task=Task.NLP_TEXT_ENCODING,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model using the new loader
    loader = SentenceEmbeddingGenerationLoader(variant=variant)
    framework_model = loader.load_model()

    # Get sample inputs from the loader
    input_tokens = loader.load_inputs()
    inputs = [
        input_tokens["input_ids"],
        input_tokens["attention_mask"],
    ]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Run inference on Tenstorrent device
    _ = compiled_model(*inputs)

    # # Model Verification and Inference
    # _, co_out = verify(inputs, framework_model, compiled_model)

    # # Post processing
    # loader.decode_output(co_out)
