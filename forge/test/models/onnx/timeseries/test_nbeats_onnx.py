# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.timeseries.nbeats.model_utils.dataset import (
    get_electricity_dataset_input,
)
from test.models.pytorch.timeseries.nbeats.model_utils.model import (
    NBeatsWithGenericBasis,
    NBeatsWithSeasonalityBasis,
    NBeatsWithTrendBasis,
)
import onnx
import torch


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [pytest.param("seasionality_basis", marks=pytest.mark.pr_models_regression)])
def test_nbeats_with_seasonality_basis_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    # Prepare model and input
    x, x_mask = get_electricity_dataset_input()

    torch_model = NBeatsWithSeasonalityBasis(
        input_size=72,
        output_size=24,
        num_of_harmonics=1,
        stacks=30,
        layers=4,
        layer_size=2048,
    )
    torch_model.eval()

    inputs = [x.contiguous(), x_mask.contiguous()]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [pytest.param("generic_basis")])
def test_nbeats_with_generic_basis(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    # Prepare model and input
    x, x_mask = get_electricity_dataset_input()

    torch_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    torch_model.eval()

    inputs = [x, x_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [pytest.param("trend_basis")])
def test_nbeats_with_trend_basis(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.NBEATS,
        variant=variant,
        task=Task.TIME_SERIES_FORECASTING,
        source=Source.GITHUB,
    )

    x, x_mask = get_electricity_dataset_input()

    torch_model = NBeatsWithTrendBasis(
        input_size=72,
        output_size=24,
        degree_of_polynomial=3,
        stacks=30,
        layers=4,
        layer_size=256,
    )
    torch_model.eval()

    inputs = [x, x_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
