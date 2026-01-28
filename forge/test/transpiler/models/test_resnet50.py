# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler tests for the ResNet-50 model (microsoft/resnet-50).

Two focused test cases:
1. ``test_resnet50_tir_graph``        – PyTorch → ONNX → TIRGraph conversion +
                                        ONNX Runtime output comparison.
2. ``test_resnet50_forge_module_gen`` – ONNX → Forge module code generation.
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

_HF_REPO = "microsoft/resnet-50"
_OPSET = 17
_GRAPH_NAME = "resnet50_model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_resnet50_model():
    """Load ResNet-50 model and image processor; skip if unavailable."""
    try:
        from transformers import AutoImageProcessor, ResNetForImageClassification
    except ImportError:
        pytest.skip("transformers library not available")

    try:
        model = ResNetForImageClassification.from_pretrained(_HF_REPO)
        processor = AutoImageProcessor.from_pretrained(_HF_REPO)
        model.eval()
        return model, processor
    except Exception as exc:
        pytest.skip(f"Failed to load ResNet-50 model: {exc}")


def _create_test_input(processor=None, batch_size: int = 1) -> torch.Tensor:
    """
    Create a test input tensor for ResNet-50.

    Attempts to load a real image via the HuggingFace datasets library.
    Falls back to a random ``(B, 3, 224, 224)`` tensor if that fails.
    """
    if processor is not None:
        try:
            from datasets import load_dataset

            dataset = load_dataset("huggingface/cats-image", split="test")
            inputs = processor(dataset[0]["image"], return_tensors="pt")
            return inputs["pixel_values"]
        except Exception:
            pass

    return torch.randn(batch_size, 3, 224, 224)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestResNet50Transpilation:
    """Transpiler tests for ResNet-50."""

    def test_resnet50_tir_graph(self):
        """
        Verify PyTorch → ONNX → TIRGraph conversion and ONNX Runtime parity.

        Steps:
        1. Load ResNet-50 from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Transpile to TIRGraph and print the graph summary.
        4. Run ONNX Runtime and compare outputs with the TIRGraph execution.
        """
        print_section("ResNet-50 — TIRGraph Conversion Test")

        model, processor = _load_resnet50_model()
        test_input = _create_test_input(processor=processor)

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=_OPSET,
        ) as onnx_model:
            tir_graph = run_tir_transpilation(
                onnx_model,
                title="TIRGraph — ResNet-50",
            )

            input_data = {"pixel_values": test_input.detach().cpu().numpy()}
            run_onnx_comparison(tir_graph, onnx_model, input_data)

        print(f"\n{SEPARATOR}\n")

    def test_resnet50_forge_module_gen(self):
        """
        Verify Forge module code generation from the ResNet-50 ONNX model.

        Steps:
        1. Load ResNet-50 from HuggingFace.
        2. Export to ONNX (opset 17).
        3. Run the Forge codegen pipeline and verify the generated module.
        """
        print_section("ResNet-50 — Forge Module Generation Test")

        model, processor = _load_resnet50_model()
        test_input = _create_test_input(processor=processor)

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=_OPSET,
        ) as onnx_model:
            run_forge_module_gen(
                onnx_model=onnx_model,
                test_input=test_input,
                graph_name=_GRAPH_NAME,
            )

        print(f"\n{SEPARATOR}\n")
