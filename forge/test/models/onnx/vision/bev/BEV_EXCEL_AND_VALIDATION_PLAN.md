# BEV Block Benchmark — Excel Report & Output Validation Plan

---

## Task 1 — Excel Report for BEV Block Benchmarks

### Goal

Create one Excel file (`BEV_Config_Analysis.xlsx`) with **6 sub-sheets**, one per block
(A through F), using the **exact same column headers and color scheme** as
`Evo50_Config_Analysis.xlsx`.

---

### Exact column headers (taken directly from Evo50 Excel)

| Col | Header (exact) | BEV usage |
|-----|----------------|-----------|
| A | `Config ID` | C00, C01, C02 … |
| B | `Configuration Name` | e.g. `Baseline (no compiler options)` |
| C | `Setting` | Exact `CompilerConfig` / `MLIRConfig` code snippet |
| D | `PCC (before fix)` | PCC of Forge output vs golden ONNX model for this config |
| E | `FPS (before fix)` | FPS parsed from the benchmark log for this config |
| F | `Issue / Remarks` | Any compile errors, hangs, regressions observed |
| G | `Fix Implemented` | Optimization or change applied on top of this config |
| H | `PCC (after fix)` | PCC after the fix/optimization (if applicable) |
| I | `FPS (after fix)` | FPS after the fix/optimization (if applicable) |
| J | `Status` | `Retained` / `Improved` / `Regressed` / `Failed` |
| K | `Status Reason` | Explanation + source log filename + timestamp |

---

### Sheet layout (one per block)

| Sheet name | Block label | Nodes |
|------------|-------------|-------|
| `Block_A` | CameraDeformedCylinder Backbone | 460 |
| `Block_B` | CameraDeformedCylinder BEV Transform | 80 |
| `Block_C` | CameraCylinder Backbone | 129 |
| `Block_D` | CameraCylinder BEV Transform | 20 |
| `Block_E` | BEV Aggregator Backbone | 57 |
| `Block_F` | Output Heads | 24 |

---

### Baseline row (C00) — populated from the 6 log files

```
blockA_baseline_disable_program_cache.log
blockB_baseline_disable_program_cache.log
blockC_baseline_disable_program_cache.log
blockD_baseline_disable_program_cache.log
blockE_baseline_disable_program_cache.log
blockF_baseline_disable_program_cache.log
```

FPS and inference time are parsed from the result table printed at the end of each log.
Example pattern matched:

```
| baseline  [cache=OFF]   | 1234.56 ± 12.34 ms  | 1250.00 ms  | 0.80 |
```

The C00 setting cell will contain:

```python
compiler_cfg = CompilerConfig()

# no mlir_config
# no enable_optimization_passes
# no program cache
```

---

### Future config rows (pre-defined structure)

| Config ID | Configuration Name |
|-----------|-------------------|
| C00 | Baseline (no compiler options, cache OFF) |
| C01 | Forge Optimization Passes, cache OFF |
| C02 | Forge Optimization Passes, cache ON |
| C03 | Consteval, cache OFF |
| C04 | Consteval, cache ON |
| … | additional configs as tested |

---

### Color scheme (matches Evo50 exactly)

| Element | Color |
|---------|-------|
| Header row | Dark blue background `#2F4F7F`, white bold text |
| Odd data rows | Light blue `#DDEBF7` |
| Even data rows | White |
| Status = Retained / Improved | Green `#C6EFCE` |
| Status = Failed | Orange/Red `#FFC7CE` |
| Status = Regressed / No Change | Yellow `#FFEB9C` |

---

### Pending clarification

> **Log file location** — please confirm the absolute path where the 6 log files live
> so the Excel generation script can parse FPS from them.
> They are assumed to be pytest terminal output redirected externally, e.g.:
>
> ```bash
> pytest test_bev_blocks_benchmark.py \
>     -k "block_A and baseline and disable_program_cache" -s \
>     2>&1 | tee blockA_baseline_disable_program_cache.log
> ```

---

## Task 2 — Output Validation in `test_bev_blocks_benchmark.py`

### Goal

Compare each compiled block's output against the **golden ONNX model output**
(run via ONNX Runtime through `forge.OnnxModule`), using the `verify` function
already used in `test_bev_onnx_forge.py`.

### How `verify` works (from `test_bev_onnx_forge.py`)

```python
framework_model = forge.OnnxModule("onnx_bev", onnx_model)          # golden: ORT
compiled_model  = forge.compile(onnx_model, sample_inputs=...)       # compiled: TT device
verify(input_tensors, framework_model, compiled_model)               # compares outputs
```

`verify` internally runs both models on the same inputs, computes PCC on every
output tensor, and raises an assertion if any output falls below the PCC threshold.

---

### Block-level validation — same pattern, per split model

For each block the split ONNX file is used instead of the full model:

```python
# Example for Block A
split_model  = onnx.load("BEV_model/split_models/block_A_deformed_backbone.onnx")
framework_model = forge.OnnxModule("block_A_deformed_backbone", split_model)
compiled_model  = forge.compile(split_model, sample_inputs=block_a_inputs, ...)
verify(block_a_inputs, framework_model, compiled_model)
```

This applies identically to all 6 blocks — the golden reference in every case is
ONNX Runtime running the **same split ONNX file**. No ground-truth `.bin` files
or intermediate `.npy` files needed.

---

### Implementation approach

Add `validate` as a **fourth pytest parameter** in `test_opt_sweep`:

```python
@pytest.mark.parametrize(
    "validate",
    [True, False],
    ids=["with_validation", "no_validation"],
)
```

Full test ID example:

```
test_opt_sweep[enable_program_cache-baseline-block_A-with_validation]
test_opt_sweep[disable_program_cache-baseline-block_A-no_validation]
```

Selectable via `-k`:

```bash
# Benchmark + validate block A baseline:
pytest -k "block_A and baseline and disable_program_cache and with_validation"

# Benchmark only (no validation):
pytest -k "block_A and baseline and disable_program_cache and no_validation"

# Validate all blocks, baseline only:
pytest -k "baseline and disable_program_cache and with_validation"
```

---

### New helper: `_validate_block`

```python
def _validate_block(
    block_name: str,
    compiled,
    block_inputs: list[torch.Tensor],
) -> None:
    """
    Run the split ONNX model through forge.OnnxModule (ORT) as the golden
    reference, then call verify() to compare against the compiled output.
    """
    split_model_path = split_models_dir() / f"{block_name}.onnx"
    onnx_model = onnx.load(str(split_model_path))
    framework_model = forge.OnnxModule(block_name, onnx_model)

    verify(block_inputs, framework_model, compiled)
```

This is called inside `test_opt_sweep` when `validate=True`, after the benchmark
completes:

```python
if validate:
    print(f"\n[validation] Running verify() for {block_short} ...")
    _validate_block(block_name, compiled, sample_inputs)
    print(f"[validation] PASSED")
```

---

### PCC threshold

`verify` uses forge's default PCC threshold (0.99). No custom threshold needed —
the comparison is Forge vs ONNX Runtime on the same split model, so the outputs
should be numerically close.

---

## Files to create / modify

| File | Action |
|------|--------|
| `forge/test/models/onnx/vision/bev/generate_bev_excel.py` | NEW — parses log files, writes `BEV_Config_Analysis.xlsx` |
| `forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py` | MODIFY — add `validate` param + `_validate_block` helper |
| `BEV_Config_Analysis.xlsx` | GENERATED output |

---

## Execution order

```bash
# 1. Run baseline benchmarks and capture logs
pytest test_bev_blocks_benchmark.py \
    -k "baseline and disable_program_cache" -s \
    2>&1 | tee all_baseline.log

# 2. Generate Excel from log files
python forge/test/models/onnx/vision/bev/generate_bev_excel.py \
    --log-dir <path-to-logs> \
    --output BEV_Config_Analysis.xlsx

# 3. Run with validation
pytest test_bev_blocks_benchmark.py \
    -k "baseline and disable_program_cache and with_validation" -s
```
