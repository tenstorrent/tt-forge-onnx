# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler tests for the BERT (bert-base-uncased) model.

Two focused test cases:
1. ``test_bert_tir_graph``        – PyTorch → ONNX → TIRGraph conversion +
                                    ONNX Runtime output comparison.
2. ``test_bert_forge_module_gen`` – ONNX → Forge module code generation.
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

_VARIANT = "bert-base-uncased"
_OPSET = 17
_MAX_LENGTH = 128
_GRAPH_NAME = "bert_base_uncased_model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bert_model(variant: str = _VARIANT):
    """Load BERT model and tokenizer from HuggingFace; skip if unavailable."""
    try:
        from transformers import BertForMaskedLM, BertTokenizer
    except ImportError:
        pytest.skip("transformers library not available")

    try:
        tokenizer = BertTokenizer.from_pretrained(variant)
        model = BertForMaskedLM.from_pretrained(variant, return_dict=False)
        model.eval()
        return model, tokenizer
    except Exception as exc:
        pytest.skip(f"Failed to load BERT model '{variant}': {exc}")


def _create_test_input(tokenizer, batch_size: int = 1) -> torch.Tensor:
    """Tokenise a short sample sentence and return the input_ids tensor."""
    tokens = tokenizer(
        "The capital of France is [MASK].",
        max_length=_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens["input_ids"]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestBertTranspilation:
    """Transpiler tests for BERT (bert-base-uncased)."""

    def test_bert_tir_graph(self):
        """
        Verify PyTorch → ONNX → TIRGraph conversion and ONNX Runtime parity.

        Steps:
        1. Load BERT from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Transpile to TIRGraph and print the graph summary.
        4. Run ONNX Runtime and compare outputs with the TIRGraph execution.
        """
        print_section(f"BERT {_VARIANT} — TIRGraph Conversion Test")

        model, tokenizer = _load_bert_model()
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
                title=f"TIRGraph — BERT {_VARIANT}",
            )

            input_data = {"input_ids": test_input.detach().cpu().numpy()}
            result = run_onnx_comparison(tir_graph, onnx_model, input_data)
            assert not result["errors"], f"Comparison errors: {result['errors']}"
            assert all(result["matches"].values()), f"Output mismatch: {result['matches']}"

        print(f"\n{SEPARATOR}\n")

    def test_bert_forge_module_gen(self):
        """
        Verify Forge module code generation from the BERT ONNX model.

        Steps:
        1. Load BERT from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Run the Forge codegen pipeline and verify the generated module.
        """
        print_section(f"BERT {_VARIANT} — Forge Module Generation Test")

        model, tokenizer = _load_bert_model()
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
