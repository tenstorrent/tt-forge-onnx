# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import paddlenlp_from_pretrained

from paddlenlp.transformers import (
    BertForSequenceClassification,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertTokenizer,
)

inputs_map = {
    "bert-base-uncased": {
        "sequence": ["Hello, my dog is cute"],
        "mask": ["One, [MASK], three, four"],
        "question": ["What is the capital of China?"],
    },
    "cl-tohoku/bert-base-japanese": {
        "sequence": ["こんにちは、私の犬はかわいいです"],
        "mask": ["一つ、[MASK]、三、四"],
        "question": ["中国の首都はどこですか？"],
    },
    "uer/chinese-roberta-base": {"sequence": ["你好，我的狗很可爱"], "mask": ["一，[MASK]，三，四"], "question": ["中国的首都是哪里？"]},
}


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant, input",
    [
        ("bert-base-uncased", inputs_map["bert-base-uncased"]["sequence"]),
        ("cl-tohoku/bert-base-japanese", inputs_map["cl-tohoku/bert-base-japanese"]["sequence"]),
        pytest.param(
            "uer/chinese-roberta-base",
            inputs_map["uer/chinese-roberta-base"]["sequence"],
            marks=pytest.mark.skip(reason="Skipped test"),
        ),
    ],
)
def test_bert_sequence_classification(variant, input):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BERT,
        variant=variant.split("/")[-1] if "/" in variant else variant,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer (with retry for transient network failures)
    model = paddlenlp_from_pretrained(
        BertForSequenceClassification.from_pretrained,
        variant,
        num_classes=2,
    )
    tokenizer = paddlenlp_from_pretrained(BertTokenizer.from_pretrained, variant)

    # Load sample
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant, input", [(key, value["mask"]) for key, value in inputs_map.items()])
def test_bert_maskedlm(variant, input):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BERT,
        variant=variant.split("/")[-1] if "/" in variant else variant,
        task=Task.NLP_MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer (with retry for transient network failures)
    model = paddlenlp_from_pretrained(BertForMaskedLM.from_pretrained, variant)
    tokenizer = paddlenlp_from_pretrained(BertTokenizer.from_pretrained, variant)

    # Load sample
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)

    # Inference
    outputs = compiled_model(*inputs)
    logits = outputs[0]
    mask_token_index = (inputs[0] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


@pytest.mark.nightly
@pytest.mark.parametrize("variant, input", [(key, value["question"]) for key, value in inputs_map.items()])
def test_bert_question_answering(variant, input):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.BERT,
        variant=variant.split("/")[-1] if "/" in variant else variant,
        task=Task.NLP_QA,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer (with retry for transient network failures)
    model = paddlenlp_from_pretrained(BertForQuestionAnswering.from_pretrained, variant)
    tokenizer = paddlenlp_from_pretrained(BertTokenizer.from_pretrained, variant)

    # Load sample
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)

    # Inference
    outputs = compiled_model(*inputs)
    start_logits, end_logits = outputs
    start_index = start_logits.argmax(dim=-1).item()
    end_index = end_logits.argmax(dim=-1).item()
    answer = tokenizer.decode(encoded_input["input_ids"][0][start_index : end_index + 1])
    print("The answer is: ", answer)
