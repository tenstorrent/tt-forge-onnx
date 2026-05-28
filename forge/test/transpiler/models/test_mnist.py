# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transpiler tests for a hand-crafted MNIST CNN model.

Two focused test cases:
1. ``test_mnist_tir_graph``        – PyTorch → ONNX → TIRGraph conversion +
                                     ONNX Runtime output comparison.
2. ``test_mnist_forge_module_gen`` – ONNX → Forge module code generation.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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

_OPSET = 17
_GRAPH_NAME = "mnist_model"


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class MnistModel(nn.Module):
    """
    Small CNN for MNIST digit classification.

    Architecture (from the canonical PyTorch MNIST example):
      Conv2d(1→32, k=3) → ReLU → Conv2d(32→64, k=3) → ReLU
      → MaxPool2d(2) → Dropout(0.25) → Flatten
      → Linear(9216→128) → ReLU → Dropout(0.5) → Linear(128→10)
      → LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mnist_model() -> MnistModel:
    model = MnistModel()
    model.eval()
    return model


def _create_test_input(batch_size: int = 1) -> torch.Tensor:
    """Return a random ``(B, 1, 28, 28)`` grayscale image tensor."""
    return torch.randn(batch_size, 1, 28, 28)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.transpiler
class TestMnistTranspilation:
    """Transpiler tests for the MNIST CNN model."""

    def test_mnist_tir_graph(self):
        """
        Verify PyTorch → ONNX → TIRGraph conversion and ONNX Runtime parity.

        Steps:
        1. Instantiate the MNIST CNN.
        2. Export to ONNX (opset 17).
        3. Transpile to TIRGraph and print the graph summary.
        4. Run ONNX Runtime and compare outputs with the TIRGraph execution.
        """
        print_section("MNIST — TIRGraph Conversion Test")

        model = _create_mnist_model()
        test_input = _create_test_input()

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["input"],
            output_names=["output"],
            opset_version=_OPSET,
        ) as onnx_model:
            tir_graph = run_tir_transpilation(
                onnx_model,
                title="TIRGraph — MNIST",
            )

            input_data = {"input": test_input.detach().cpu().numpy()}
            result = run_onnx_comparison(tir_graph, onnx_model, input_data)
            assert not result["errors"], f"Comparison errors: {result['errors']}"
            assert all(result["matches"].values()), f"Output mismatch: {result['matches']}"

        print(f"\n{SEPARATOR}\n")

    def test_mnist_forge_module_gen(self):
        """
        Verify Forge module code generation from the MNIST ONNX model.

        Steps:
        1. Instantiate the MNIST CNN.
        2. Export to ONNX (opset 17).
        3. Run the Forge codegen pipeline and verify the generated module.
        """
        print_section("MNIST — Forge Module Generation Test")

        model = _create_mnist_model()
        test_input = _create_test_input()

        with export_to_onnx(
            pytorch_model=model,
            test_input=test_input,
            input_names=["input"],
            output_names=["output"],
            opset_version=_OPSET,
        ) as onnx_model:
            run_forge_module_gen(
                onnx_model=onnx_model,
                test_input=test_input,
                graph_name=_GRAPH_NAME,
            )

        print(f"\n{SEPARATOR}\n")
