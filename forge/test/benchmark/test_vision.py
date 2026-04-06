# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import forge
import pytest
from loguru import logger

from test.benchmark.benchmarks.vision_benchmark import benchmark_vision_forge_onnx
from test.benchmark.options import ForgeBenchmarkOptions

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from third_party.tt_forge_models.resnet.pytorch.loader import ModelLoader, ModelVariant
from third_party.tt_forge_models.tools.utils import export_torch_model_to_onnx

DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = True
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOOP_COUNT = 128
DEFAULT_WARMUP_COUNT = 32
DEFAULT_INPUT_SIZE = (3, 224, 224)
DEFAULT_DATA_FORMAT = "bfloat16"


def test_vision(
    model,
    model_name,
    forge_benchmark_options: ForgeBenchmarkOptions,
    load_inputs_fn,
    extract_output_tensor_fn,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled: bool = DEFAULT_TRACE_ENABLED,
    training: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_size: tuple = DEFAULT_INPUT_SIZE,
    loop_count: int = DEFAULT_LOOP_COUNT,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    data_format: str = DEFAULT_DATA_FORMAT,
    required_pcc: float = 0.99,
):
    """Run ``benchmark_vision_forge_onnx`` and optionally write JSON from CLI.

    Args:
        model: Framework module (e.g. ``forge.OnnxModule``) to compile and time.
        model_name: Logical model name for logging and result payloads.
        forge_benchmark_options: Parsed CLI overrides (batch, loops, output path, etc.).
        load_inputs_fn: Callable ``(batch_size, **kwargs) -> Tensor`` for input batches.
        extract_output_tensor_fn: Callable mapping raw model output to tensors for PCC.
        optimization_level: MLIR/compiler optimization level.
        trace_enabled: Whether to enable trace in compiler config.
        training: Training mode flag (unsupported path in benchmark; forwarded).
        batch_size: Default batch size if not overridden by ``forge_benchmark_options``.
        input_size: Logical input shape metadata passed through to reporting.
        loop_count: Default timed iteration count if not overridden by CLI.
        warmup_count: Default warmup iterations if not overridden by CLI.
        data_format: Default ``float32`` / ``bfloat16`` label if not overridden by CLI.
        required_pcc: Minimum PCC vs golden for a passing run.

    Returns:
        None. Side effect: writes JSON if ``forge_benchmark_options.output_file`` is set.
    """
    o = forge_benchmark_options
    batch_size = o.batch_size if o.batch_size is not None else batch_size
    loop_count = o.loop_count if o.loop_count is not None else loop_count
    warmup_count = o.warmup_count if o.warmup_count is not None else warmup_count
    data_format = o.data_format if o.data_format is not None else data_format
    training = o.training
    output_file = o.output_file

    logger.info(f"Running Forge ONNX vision benchmark for model: {model_name}")
    logger.info(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    training={training}
    batch_size={batch_size}
    input_size={input_size}
    loop_count={loop_count}
    warmup_count={warmup_count}
    data_format={data_format}
    required_pcc={required_pcc}
    """
    )

    results = benchmark_vision_forge_onnx(
        model=model,
        model_name=model_name,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        batch_size=batch_size,
        loop_count=loop_count,
        warmup_count=warmup_count,
        input_size=input_size,
        data_format=data_format,
        required_pcc=required_pcc,
        training=training,
    )

    if output_file:
        results["project"] = "tt-forge/tt-forge-onnx"
        results["model_rawname"] = model_name
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)


variants = [
    ModelVariant.RESNET_50_HF,
]


@pytest.mark.perf
@pytest.mark.parametrize("variant", variants)
def test_resnet50(variant, forge_benchmark_options, forge_tmp_path):
    model_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.RESNET,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    batch_size = (
        forge_benchmark_options.batch_size if forge_benchmark_options.batch_size is not None else DEFAULT_BATCH_SIZE
    )
    data_format = (
        forge_benchmark_options.data_format if forge_benchmark_options.data_format is not None else DEFAULT_DATA_FORMAT
    )

    loader = ModelLoader(variant=variant)
    pytorch_model = loader.load_model()
    pytorch_model.eval()

    inputs = loader.load_inputs(batch_size=batch_size)

    onnx_model = export_torch_model_to_onnx(
        pytorch_model,
        str(forge_tmp_path),
        inputs,
        model_name,
        opset_version=17,
    )
    framework_model = forge.OnnxModule(model_name, onnx_model)
    effective_df = forge_benchmark_options.data_format or DEFAULT_DATA_FORMAT
    if effective_df == "bfloat16":
        framework_model.set_data_format_override(forge._C.DataFormat.Float16_b)

    def load_inputs_fn(batch_size, dtype_override=None):
        return loader.load_inputs(batch_size=batch_size, dtype_override=dtype_override)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=framework_model,
        model_name=model_name,
        forge_benchmark_options=forge_benchmark_options,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
    )
