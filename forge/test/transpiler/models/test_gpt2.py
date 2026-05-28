# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler tests for the GPT-2 sequence-classification model
(mnoukhov/gpt2-imdb-sentiment-classifier).

Two focused test cases:
1. ``test_gpt2_tir_graph``        – PyTorch → ONNX → TIRGraph conversion +
                                    ONNX Runtime output comparison.
2. ``test_gpt2_forge_module_gen`` – ONNX → Forge module code generation.

GPT-2 model notes
-----------------
- Architecture  : Decoder-only transformer (GPT-2) with a classification head.
- HuggingFace   : AutoModelForSequenceClassification + AutoTokenizer.
- Export flags  : use_cache=False, return_dict=False (required for torch.onnx.export).
- Opset         : 17
- ONNX ops used : Add, ArgMax, Cast, Concat, Constant, ConstantOfShape, Div,
                  Equal, Expand, Flatten, Gather, Gemm, LayerNormalization,
                  MatMul, Mul, Not, Pow, Reshape, Shape, Slice, Softmax,
                  Split, Sqrt, Tanh, Transpose, Trilu, Unsqueeze, Where
                  (28 distinct op types).
"""
import pytest
import torch

from test.transpiler.models.model_test_utils import (
    SEPARATOR,
    export_to_onnx,
    print_section,
    run_forge_module_gen,
    run_onnx_comparison,
    run_tir_transpilation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANT = "mnoukhov/gpt2-imdb-sentiment-classifier"
_OPSET = 17
_GRAPH_NAME = "gpt2_imdb_sentiment_classifier"

# Short sample text used for tokenisation; produces a fixed-length input so
# ONNX export (which traces the model) always sees the same shape.
_SAMPLE_TEXT = "This is a sample text from "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gpt2_model(variant: str = _VARIANT):
    """Load GPT-2 sequence-classification model and tokenizer from HuggingFace; skip if unavailable."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers library not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(variant, padding_side="left")
        model = AutoModelForSequenceClassification.from_pretrained(
            variant,
            return_dict=False,
            use_cache=False,
        )
        model.eval()
        return model, tokenizer
    except Exception as exc:
        pytest.skip(f"Failed to load GPT-2 model '{variant}': {exc}")


def _create_test_input(tokenizer, text: str = _SAMPLE_TEXT) -> torch.Tensor:
    """Tokenise ``text`` and return the resulting ``input_ids`` tensor."""
    tokens = tokenizer(text, return_tensors="pt")
    return tokens["input_ids"]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestGPT2Transpilation:
    """Transpiler tests for GPT-2 (mnoukhov/gpt2-imdb-sentiment-classifier)."""

    def test_gpt2_tir_graph(self):
        """
        Verify PyTorch → ONNX → TIRGraph conversion and ONNX Runtime parity.

        Steps:
        1. Load GPT-2 from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Transpile to TIRGraph and print the graph summary.
        4. Run ONNX Runtime and compare outputs with the TIRGraph execution.
        """
        print_section(f"GPT-2 ({_VARIANT}) — TIRGraph Conversion Test")

        model, tokenizer = _load_gpt2_model()
        test_input = _create_test_input(tokenizer)

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=_OPSET,
        ) as onnx_model:
            tir_graph = run_tir_transpilation(
                onnx_model,
                title=f"TIRGraph — GPT-2 {_VARIANT}",
            )

            input_data = {"input_ids": test_input.detach().cpu().numpy()}
            result = run_onnx_comparison(tir_graph, onnx_model, input_data)
            assert not result["errors"], f"Comparison errors: {result['errors']}"
            assert all(result["matches"].values()), f"Output mismatch: {result['matches']}"

        print(f"\n{SEPARATOR}\n")

    def test_gpt2_forge_module_gen(self):
        """
        Verify Forge module code generation from the GPT-2 ONNX model.

        Steps:
        1. Load GPT-2 from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Run the Forge codegen pipeline and verify the generated module.
        """
        print_section(f"GPT-2 ({_VARIANT}) — Forge Module Generation Test")

        model, tokenizer = _load_gpt2_model()
        test_input = _create_test_input(tokenizer)

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=_OPSET,
        ) as onnx_model:
            run_forge_module_gen(
                onnx_model=onnx_model,
                test_input=test_input,
                graph_name=_GRAPH_NAME,
                resolve_dynamic_shapes=False,
            )

        print(f"\n{SEPARATOR}\n")
