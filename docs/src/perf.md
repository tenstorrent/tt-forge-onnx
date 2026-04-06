# Performance Benchmarks

Performance benchmark tests are located under `forge/test/benchmark/` and are marked with `@pytest.mark.perf`. These tests measure end-to-end inference throughput on Tenstorrent device by compiling the model with Forge, running timed device iterations, and verifying numerical accuracy against a CPU reference.

## Directory layout

```
forge/test/benchmark/
├── conftest.py               # CLI options + forge_benchmark_options fixture
├── options.py                # ForgeBenchmarkOptions dataclass
├── utils.py                  # Console reporting + JSON result helpers
├── test_vision.py            # pytest entry points (ResNet-50, etc.)
└── benchmarks/
    └── vision_benchmark.py   # compile / warmup / timed loop / PCC logic
```

## Running benchmarks locally

To run all benchmark tests:

```sh
pytest -m perf forge/test/benchmark
```

To run a specific model:

```sh
pytest -svv -m perf forge/test/benchmark/test_vision.py::test_resnet50
```

To save results to a JSON file:

```sh
pytest -m perf forge/test/benchmark --output-file results.json
```

## CLI options

These flags are registered in `conftest.py` and apply to any test under `forge/test/benchmark`. All overrides are optional; each test defines its own defaults.

| Option | Default | Description |
|--------|---------|-------------|
| `--output-file PATH` | `None` | Path to write benchmark results as JSON. If omitted, no file is written. |
| `--batch-size N` | per-test default | Number of samples per inference call (positive integer). |
| `--loop-count N` | per-test default | Number of timed iterations after warmup (positive integer). |
| `--warmup-count N` | `min(32, loop_count)` | Number of warmup iterations before timing begins (positive integer). |
| `--data-format` | `bfloat16` | Data format for model inputs and compiler config: `float32` or `bfloat16`. |
| `--training` | `False` | Run in training mode. Not supported by current benchmarks; raises an error if set. |

**Example:**

```sh
pytest -svv -m perf forge/test/benchmark/test_vision.py::test_resnet50 \
    --batch-size 8 \
    --loop-count 128 \
    --warmup-count 32 \
    --data-format bfloat16 \
    --output-file resnet50_results.json
```

## Adding a new benchmark model

Add a parametrized test in `test_vision.py` for the same model family, or create a new `test_<family>.py` for a different category. Each test must be marked with `@pytest.mark.perf` and accept the `forge_benchmark_options` and `forge_tmp_path` fixtures.

**Example:**

```python
from third_party.tt_forge_models.<model>.pytorch.loader import ModelLoader, ModelVariant
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = [ModelVariant.MY_MODEL]

@pytest.mark.perf
@pytest.mark.parametrize("variant", variants)
def test_my_model(variant, forge_benchmark_options, forge_tmp_path):
    model_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MY_ARCH,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    batch_size = forge_benchmark_options.batch_size or DEFAULT_BATCH_SIZE
    loader = ModelLoader(variant=variant)
    pytorch_model = loader.load_model().eval()
    inputs = loader.load_inputs(batch_size=batch_size)
    onnx_model = export_torch_model_to_onnx(
        pytorch_model, str(forge_tmp_path), inputs, model_name, opset_version=17
    )

    def load_inputs_fn(batch_size, dtype_override=None):
        return loader.load_inputs(batch_size=batch_size, dtype_override=dtype_override)

    test_vision(
        model=forge.OnnxModule(model_name, onnx_model),
        model_name=model_name,
        forge_benchmark_options=forge_benchmark_options,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=lambda o: o,
        batch_size=batch_size,
    )
```
