# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertModel,
    BertTokenizer,
)

import forge
from forge.verify.verify import verify
from forge.verify.config import DeprecatedVerifyConfig
from forge.config import CompilerConfig

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.utils import download_model


# Opset 9 is the minimum version to support BERT in Torch.
# Opset 17 is the maximum version in Torchscript.
opset_versions = [17]


@pytest.mark.pr_models_regression
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["bert-base-uncased"])
@pytest.mark.parametrize("use_transpiler", [False, True], ids=["tvm", "transpiler"])
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_bert_masked_lm_onnx(variant, forge_tmp_path, opset_version, use_transpiler):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.NLP_MASKED_LM,
        source=Source.HUGGINGFACE,
        suffix="_transpiler" if use_transpiler else "_tvm",
    )

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertForMaskedLM.from_pretrained, variant, return_dict=False)

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_path = f"{forge_tmp_path}/bert_masked_lm.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=opset_version)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Configure compiler and verification based on compilation path
    if use_transpiler:
        # Transpiler path configuration
        compiler_cfg = CompilerConfig(
            compile_transpiler_to_python=True,  # Enable transpiler path
            compile_tvm_to_python=False,  # Disable TVM path
            transpiler_enable_debug=True,  # Enable debug mode for transpiler (ONNX Runtime comparison)
        )

        # Create verify config with all verification flags enabled for transpiler
        verify_cfg = DeprecatedVerifyConfig(
            # Transpiler-specific verification
            verify_transpiler_graph=True,  # Compare Framework output vs TIR graph output after transpiler conversion
            verify_forge_codegen_vs_framework=True,  # Compare Framework output vs Forge codegen outputs
        )
    else:
        # Set data format override
        data_format_override = forge._C.DataFormat.Float16_b
        compiler_cfg = forge.config.CompilerConfig(default_df_override=data_format_override)
        verify_cfg = DeprecatedVerifyConfig()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
    )

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    logits = co_out[0]
    mask_token_index = (input_tokens.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("opset_version", opset_versions, ids=opset_versions)
def test_bert_question_answering_onnx(variant, forge_tmp_path, opset_version):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.NLP_QA,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertForQuestionAnswering.from_pretrained, variant, return_dict=False)

    # Load data sample from SQuADv1.1
    context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
    (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
    National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
    The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
    As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
    initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
    (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
    feature the Arabic numerals 50."""

    question = "Which NFL team represented the AFC at Super Bowl 50?"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Export model to ONNX
    # TODO: Replace with pre-generated ONNX model to avoid exporting from scratch.
    onnx_path = f"{forge_tmp_path}/bert_qa.onnx"
    torch.onnx.export(framework_model, inputs[0], onnx_path, opset_version=opset_version)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"])
def test_bert_sentence_embedding_generation_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.NLP_TEXT_ENCODING,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()

    # prepare input
    sentences = "Bu örnek bir cümle"
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = [encoded_input["input_ids"], encoded_input["attention_mask"], encoded_input["token_type_ids"]]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/bert_sentence_emb_gen.onnx"
    torch.onnx.export(framework_model, (inputs[0], inputs[1], inputs[2]), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    sentence_embeddings = mean_pooling(co_out, encoded_input["attention_mask"])

    print("Sentence embeddings:", sentence_embeddings)
