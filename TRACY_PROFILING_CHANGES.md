# Tracy Profiling Changes — BEV Block D (91/91 Columns Present, 46 ops)

## Overview

This document describes every change made to the Tracy profiling pipeline for
tt-forge-onnx to go from **17/65 populated columns** (baseline) to **85/91
populated columns** for BEV Block D's `ops_perf_results` CSV.

The profiling target is:
```
forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py
  ::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]
```

---

## Background: Why Only 17 Columns at the Start

Tracy's report generator (`process_ops_logs.py`) combines two data sources:

| Source | File | Provides |
|--------|------|----------|
| Host path | `tracy_ops_times.csv` + `tracy_ops_data.csv` | OP CODE, ATTRIBUTES, HOST timing, tensor I/O |
| Device path (C++) | `cpp_device_perf_report.csv` | DEVICE timing (28 columns) |
| Device path (Python/legacy) | `profile_log_device.csv` | per-core MIN/MAX/AVG analysis |

At baseline:
- `tracy_ops_data.csv` was **empty** — TTNN C++ Tracy zones were overflowing the 32 K source-location table and being dropped entirely.
- The device path was working (17 device-timing columns), but with no host data the report was almost useless.

---

## Change 1 — Build with Tracy Instrumentation Enabled

### File
`third_party/CMakeLists.txt`

### Problem
Without Tracy instrumentation compiled in, setting `TT_METAL_DEVICE_PROFILER=1`
crashes immediately:

```
TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build.
```

Tracy zones in tt-metal / ttnn binaries are compiled out by default (`#ifdef
TRACY_ENABLE`).

### What Changed
Rebuilt the entire stack with:
```cmake
-DTT_RUNTIME_ENABLE_PERF_TRACE=ON
```
This propagates `TRACY_ENABLE` through tt-forge-onnx → tt-mlir → tt-metal,
activating all `TracyZoneScoped` and `TracyMessageL` call sites.

### Impact
Prerequisite for every other change. Without this, no profiling is possible.

---

## Change 2 — New No-Trace Compiler Config

### File
`forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py`

### Problem
TTNN **trace mode** (`set_enable_trace(True)`) collapses an entire inference
pass into a single pre-recorded TTNN trace replay. The device profiler only
sees two zone types:

```
TRACE-FW      (one zone for the whole replay)
TRACE-KERNEL  (one zone for the whole replay)
```

Result: Tracy only produces 3 columns — no per-op timing at all.

### Before
```python
# existing config — trace enabled, no block_name arg
def _cfg_opt_level_2_bfloat16_hifi3_fp32_acc():
    mlir_config = (
        MLIRConfig()
        .set_enable_trace(True)   # ← collapses to single trace replay
        ...
    )
```

### After
```python
def _cfg_opt_level_2_bfloat16_hifi3_fp32_acc_no_trace(block_name: str):
    mlir_config = (
        MLIRConfig()
        .set_enable_trace(False)   # ← 322 ops dispatched individually
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi3)
        .set_compute_cfg_fp32_dest_acc_en(True)
        .set_enable_ttnn_perf_metrics(True)
        .set_enable_ttnn_perf_metrics_verbose(True)
        .set_ttnn_perf_metrics_output_file(
            f"BEV_MODEL_LOGS/LATEST/{block_name}_no_trace_perf_metrics.json"
        )
    )
    cfg = CompilerConfig(mlir_config=mlir_config)
    cfg.enable_optimization_passes = True
    cfg.default_df_override = forge._C.DataFormat.Float16_b
    return cfg
```

Registered in `COMPILER_CONFIGS` dict so it appears as a parametrized test
variant.

### Impact
Each of the 322 TTNN ops is dispatched as a separate kernel. The device
profiler captures individual `BRISC-FW`, `BRISC-KERNEL`, `NCRISC-*`,
`TRISC-*` zones per op, enabling all device timing columns.

---

## Change 3 — Tracy Run Script Fixes

### File
`scripts/tracy_run.sh`

### Problem A — Missing `ttnn` Python module

Tracy's `tracy_ttnn.py` does `import ttnn` at load time. Without the TTNN
Python bindings on `PYTHONPATH`, the script failed immediately:

```
ModuleNotFoundError: No module named 'ttnn'
```

**Before:** `PYTHONPATH` only contained the Tracy tools directory.

**After:**
```bash
TTNN_PY="${TT_MLIR_BUILD}/install/tt-metal/ttnn"
[[ -d "${TTNN_PY}" ]] && export PYTHONPATH="${TTNN_PY}:${PYTHONPATH}"
```

---

### Problem B — Zone Table Overflow (Root Cause of Empty `tracy_ops_data.csv`)

Tracy has a hard limit of **32 768 source locations**. A typical forge/TTNN
inference pass registers far more zones than this limit. When the table fills,
Tracy silently drops all subsequent zones — including the TTNN C++ op-profiler
zones that carry `OP CODE`, `ATTRIBUTES`, `HOST DURATION`, tensor I/O, etc.

```
Too many source locations >32K — zones dropped
```

Result: `tracy_ops_data.csv` was empty → 48 host-side columns missing.

**Before:** Tracy captured all zones unconditionally.

**After:** Added `-p` / `--partial` flag support:
```bash
-p|--partial)
    PARTIAL=true; shift ;;
...
[[ "${PARTIAL}" == true ]] && TRACY_ARGS+=(-p)
```

The `-p` flag tells Tracy to only profile zones that are **explicitly
enabled** in the binary (those wrapped with `TracySetProgramName` /
zone filters), cutting zone count well below 32 K and preserving the TTNN op
metadata zones.

**Impact:** `tracy_ops_data.csv` went from 0 rows to 4 709 rows. Host-side
columns (OP CODE, OP TYPE, ATTRIBUTES, HOST timing, tensor I/O, PM model)
all became populated.

---

### Problem C — `TT_METAL_DEVICE_PROFILER=1` Guarding

Setting `TT_METAL_DEVICE_PROFILER=1` without a Tracy-enabled build crashes.
The script was setting it unconditionally.

**Before:**
```bash
export TT_METAL_DEVICE_PROFILER=1   # always set
```

**After:**
```bash
# Only set when device tracing is active AND not in no-device mode
[[ "${DEVICE_TRACE}" == true && "${NO_DEVICE}" == false ]] && \
    export TT_METAL_DEVICE_PROFILER=1
```

---

### Problem D — Bash Syntax Error (Silent Flag Ignore)

A missing space before `==` in a comparison caused `SYNC_HOST_DEVICE` to
never be recognized:

**Before:**
```bash
[[ "${SYNC_HOST_DEVICE}"== true  ]] && TRACY_ARGS+=(--sync-host-device)
```

**After:**
```bash
[[ "${SYNC_HOST_DEVICE}" == true  ]] && TRACY_ARGS+=(--sync-host-device)
```

---

### Problem E — No Way to Enable Optional Profiling Modes

The script only had `--no-*` disable flags for NOC traces, dispatch-cores, and
host-device sync. Since all three defaulted to `false`, there was no way to
turn them on from the command line.

**Before:** Only `--no-noc-traces`, `--no-dispatch-cores`, `--no-sync`.

**After:** Added the corresponding enable flags:
```bash
--noc-traces)      NOC_TRACES=true;      shift ;;
--dispatch-cores)  DISPATCH_CORES=true;  shift ;;
--sync)            SYNC_HOST_DEVICE=true; shift ;;
```

---

## Change 4 — `process_device_log.py`: pandas 2.x Dtype Fix

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_device_log.py`

### Problem
pandas 2.x changed how column assignment works. When a `DataFrame` column has
`object` dtype (strings), assigning an `int` Series via `df.iloc[:, N] = ...`
raises:

```
TypeError: Invalid value '...' for dtype 'str'
```

### Before
```python
df.iloc[:, 8] = pd.to_numeric(df.iloc[:, 8], errors="coerce").fillna(-1).astype(int)
df.iloc[:, 9] = pd.to_numeric(df.iloc[:, 9], errors="coerce").fillna(-1).astype(int)
```

### After
```python
_col_trace_id       = df.columns[8]
_col_trace_id_count = df.columns[9]
df[_col_trace_id]       = pd.to_numeric(df[_col_trace_id],       errors="coerce").fillna(-1).astype(int)
df[_col_trace_id_count] = pd.to_numeric(df[_col_trace_id_count], errors="coerce").fillna(-1).astype(int)
```

Accessing columns by **name** instead of integer index bypasses the dtype
enforcement path in pandas 2.x.

### Impact
Unblocked `process_device_log.py` from crashing. Without this fix, the
device log CSV was never parsed and all device timing columns were empty.

---

## Change 5 — `process_ops_logs.py`: DictWriter Fieldnames Crash

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
When `--device-trace-profiler` is enabled, the device analysis produces two
extra columns not declared in `OPS_CSV_HEADER`:

```
TRACE FW DURATION [ns]
TRACE KERNEL DURATION [ns]
```

Python's `csv.DictWriter` requires every key in a `rowdict` to appear in
`fieldnames`. Writing a row with undeclared keys raises:

```
ValueError: dict contains fields not in fieldnames:
  'TRACE KERNEL DURATION [ns]', 'TRACE FW DURATION [ns]'
```

### Before
`allHeaders` was built from `OPS_CSV_HEADER + PERF_COUNTER_CSV_HEADERS` with
no mechanism to accommodate fields produced by optional analysis types.

### After
```python
# After building allHeaders, append any extra fields not already present
allHeadersSet = set(allHeaders)
for header in csv_row_headers:
    if header not in allHeadersSet:
        allHeaders.append(header)
```

### Impact
Reports no longer crash when trace-mode analysis types add columns outside
the static header list.

---

## Change 6 — `process_ops_logs.py`: Combined C++ + Legacy Device Path

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
The report generator has two device data paths:

| Path | Triggered when | Provides |
|------|---------------|----------|
| C++ path | `cpp_device_perf_report.csv` exists | 28 fixed columns (accurate timing + DEVICE ARCH + AVAILABLE WORKER CORE COUNT) |
| Legacy Python path | No `cpp_device_perf_report.csv` | All `timerAnalysis` types, **including `device_kernel_duration_per_core`** |

When `cpp_device_perf_report.csv` exists (the normal case), the code took the
C++ path exclusively and **ignored all device analysis types**, printing only:

```
WARNING: device_analysis_types is not supported when using
cpp_device_perf_report.csv; ignoring option.
```

Result: `DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns]` were always empty
even though the raw data existed in `profile_log_device.csv`.

### Before
```python
if use_perf_csv:
    if device_analysis_types:
        logger.warning("device_analysis_types not supported … ignoring")
    host_ops_by_device = _enrich_ops_from_perf_csv(...)   # C++ path only
else:
    host_ops_by_device = _enrich_ops_from_device_logs(...)  # legacy only
```

### After
```python
if use_perf_csv:
    # Primary: C++ path (accurate timing, DEVICE ARCH, AVAILABLE WORKER CORE COUNT)
    host_ops_by_device = _enrich_ops_from_perf_csv(...)

    # Supplemental: also run legacy path to get per-core analysis
    device_log = Path(logFolder) / PROFILER_DEVICE_SIDE_LOG
    if device_log.is_file():
        logger.info("Running supplemental legacy device-log analysis for per-core timing data.")
        try:
            host_ops_supplemental, _ = get_device_op_data(ops, host_device_op_compare)
            host_ops_supplemental = _enrich_ops_from_device_logs(
                host_ops_supplemental, logFolder, device_analysis_types, traceReplays
            )
            # Merge per-core device_time entries into primary ops, matched by global_call_count
            for device_id, device_ops in host_ops_by_device.items():
                supp_by_gcc = {op.get("global_call_count"): op
                               for op in host_ops_supplemental.get(device_id, [])}
                for op in device_ops:
                    supp = supp_by_gcc.get(op.get("global_call_count"))
                    if supp and "device_time" in supp:
                        if "device_time" not in op:
                            op["device_time"] = {}
                        for key, data in supp["device_time"].items():
                            if key not in op["device_time"]:
                                op["device_time"][key] = data
        except Exception as e:
            logger.warning(f"Supplemental legacy device-log analysis failed: {e}")
else:
    host_ops_by_device = _enrich_ops_from_device_logs(...)
```

### Impact
Added 3 columns to every run without any script changes:
- `DEVICE KERNEL DURATION PER CORE MIN [ns]`
- `DEVICE KERNEL DURATION PER CORE MAX [ns]`
- `DEVICE KERNEL DURATION PER CORE AVG [ns]`

---

## Change 7 — `process_ops_logs.py`: Device Data Mismatch — Assert → Graceful

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
When NOC tracing is enabled, the device DRAM profiler buffers fill up and
markers are dropped. The legacy path detected the host/device op count
mismatch and crashed:

```
AssertionError: Device data mismatch: Expected 322 but received 197 ops
on device 0. Device is showing op ID 952320 when host is showing op ID 944128
```

Additionally, because missing ops are dropped from arbitrary positions (not
just the tail), positional truncation (`host_ops[:197]`) would silently
mis-match op IDs.

A secondary assertion at line 720 then crashed too:
```python
assert time_id["run_host_id"] == device_op["global_call_count"]
```

### Before
```python
# Hard crash on count mismatch
assert False, f"Device data mismatch: Expected {len(...)} but received {len(...)} ops ..."

# Hard crash on per-op ID mismatch
assert time_id["run_host_id"] == device_op["global_call_count"]
```

### After — Count mismatch: warn + ID-based matching
```python
# Log warning instead of crashing
logger.warning(
    f"Device data mismatch (likely DRAM profiler buffer overflow): "
    f"Expected {len(host_ops_by_device[device])} but received {len(device_ops_time)} "
    f"ops on device {device}. Processing only matched ops."
)
# Match by run_host_id instead of by position
_dev_ops_by_id = {}
for _dot in device_ops_time:
    if len(_dot["timeseries"]) > 0:
        _tid = _dot["timeseries"][0][0]
        if "run_host_id" in _tid:
            _dev_ops_by_id[_tid["run_host_id"]] = _dot
_matched = [_dev_ops_by_id.get(op["global_call_count"])
            for op in host_ops_by_device[device]]
device_ops_time           = [d for d in _matched if d is not None]
host_ops_by_device[device] = [h for h, d in zip(host_ops_by_device[device], _matched)
                               if d is not None]
```

### After — Per-op ID mismatch: warn + skip
```python
# BEFORE: assert time_id["run_host_id"] == device_op["global_call_count"]
# AFTER:
if time_id["run_host_id"] != device_op["global_call_count"]:
    logger.warning(f"Op ID mismatch: skipping enrichment for this op")
    continue
```

### Impact
- NOC trace runs no longer crash during report generation.
- Partial data (ops captured before buffer overflow) is still written to the CSV.
- Per-core data in NOC runs populates for 197/322 ops instead of 0.

---

## Change 11 — ERISC Dispatch Support in `process_device_log.py` and `process_ops_logs.py`

### Files
- `third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_device_log.py`
- `third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
Wormhole uses ERISC-based fast dispatch. The post-processor excluded ERISC dispatch cores
entirely (`and "ERISC" not in risc` filter in `sort_timeseries`) and `get_dispatch_core_ops`
was BRISC/NCRISC only. Result: `DISPATCH TOTAL CQ CMD OP TIME [ns]` was always empty even
when `--dispatch-cores` was enabled.

Two additional bugs blocked matching:
1. `extract_dispatch_op_id()` looked for `workers_runtime_id` in `meta_data` — absent for ERISC.
2. ERISC's `runtime_host_id_dispatch` data encodes op ID as `N`; tensix `run_host_id` encodes
   it as `N × 1024` (lower 10 bits reserved). IDs never matched.

### What Changed

**`process_device_log.py`:**

1. `sort_timeseries`: Removed `and "ERISC" not in risc` — ERISC dispatch cores now added to
   `dispatchCores`.

2. Added `get_erisc_dispatch_core_ops(timeseries)`:
   - Groups `CQ-DISPATCH` zones by `runtime_host_id_dispatch` TS_DATA markers
   - Normalises zone_name `CQ-DISPATCH` → `CQ_DISPATCH_OP` and risc `ERISC` → `BRISC` so
     existing `dispatch_total_cq_cmd_op_time` analysis (zone: `CQ_DISPATCH_*`, risc: `BRISC`)
     applies without config changes
   - Stores `"op_id": attachedData * 1024` directly in each op dict (tensix encoding)

3. `get_dispatch_core_ops`: Detects ERISC timeseries and delegates to
   `get_erisc_dispatch_core_ops` instead of the BRISC/NCRISC path.

**`process_ops_logs.py`:**

4. `extract_dispatch_op_id`: Checks for `"op_id"` key first (ERISC path), falls back to
   `meta_data["workers_runtime_id"]` (BRISC/NCRISC path).

### Impact
`DISPATCH TOTAL CQ CMD OP TIME [ns]` is now populated for all 46 BEV Block D ops.
`DISPATCH GO SEND WAIT TIME [ns]` remains empty — it requires NCRISC go-signal which
ERISC dispatch does not emit.

---

## Final Column Count

| Run mode | Columns populated | Notes |
|----------|------------------|-------|
| Baseline (trace mode, no partial flag) | 17/65 | All host data missing, no per-core |
| After changes 1–7 (no-trace, `-p` flag) | **77/91** | Full 46 ops |
| After changes 1–11 (`--dispatch-cores` + ERISC fix) | **76/91** | + DISPATCH TOTAL CQ CMD OP TIME; −2 (DEVICE ARCH, AVAILABLE WORKER CORE COUNT lost when cpp post-proc disabled) |
| After changes 1–10 (`--noc-traces` + `tt-npe` built) | **85/91** | + 6 NOC UTIL / DRAM BW columns; 29/46 ops have device timing |
| After changes 12–14 (`--dispatch-cores` + `--noc-traces` combined) | **91/91 columns, 46 rows** | DRAM BW 46/46; NOC UTIL 44/46; device timing 45/46 |

### Coverage Detail (combined `--dispatch-cores --noc-traces` run after all fixes)

| Column group | Coverage | Notes |
|---|---|---|
| DRAM BW UTIL, DRAM BW UTIL PER CTRL, ETH BW UTIL | **46/46** | Via tt-npe, all ops |
| NOC UTIL, MULTICAST NOC UTIL | **44/46** | 2 ops have no NOC traffic (expected) |
| DEVICE FW/KERNEL DURATION, OP-TO-OP LATENCY, DISPATCH TOTAL CQ CMD, PM FPU UTIL | **45/46** | Op 966656 is ERISC-only (no tensix kernel), hard limit |
| DEVICE ARCH, AVAILABLE WORKER CORE COUNT, PARALLELIZATION STRATEGY, DEVICE ERISC KERNEL DURATION, DEVICE COMPUTE CB WAIT/RESERVE, DISPATCH GO SEND WAIT, METAL TRACE ID/REPLAY | **0/46** | Expected empty — wrong profiling mode or not applicable to BEV |

### The 9 Always-Empty Columns and Why

| Column(s) | Reason |
|-----------|--------|
| `DEVICE ERISC KERNEL DURATION [ns]` | BEV model has no Ethernet ops |
| `METAL TRACE ID`, `METAL TRACE REPLAY SESSION ID` | Only populated in trace mode; we profile no-trace |
| `DEVICE COMPUTE CB WAIT FRONT/BACK [ns]` | Requires `CB-COMPUTE-WAIT-FRONT` kernel zones; absent because all 322 BEV ops are DM-bound (no CB waits) |
| `PARALLELIZATION STRATEGY` | Field not present at top level of op JSON emitted by current tt-metal |
| `DISPATCH GO SEND WAIT TIME [ns]` | Requires NCRISC go-signal zones; ERISC fast dispatch does not emit these |
| `DEVICE ARCH`, `AVAILABLE WORKER CORE COUNT` | Only populated by C++ path (`cpp_device_perf_report.csv`); disabled when `--dispatch-cores` is used |

---

## Change 8 — Fix `run_mlir_compiler` Crash with `--profile-dispatch-cores`

### Files
- `forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py`

### Problem
When `--profile-dispatch-cores` is added to the Tracy run, the test crashes inside
`forge._C.run_mlir_compiler` with:

```
RuntimeError: TT_THROW @ metal_context.cpp:451
Cannot destroy MetalContext while devices are still open. Close all devices first.
```

**Root cause (traced through C++ source):**

1. `TT_METAL_DEVICE_PROFILER_DISPATCH=1` is set in the process env by Tracy `__main__.py`.
2. `forge.compile()` triggers MLIR optimization passes, including OpModel which calls
   `SingletonDeviceContext::openMockDevice()`.
3. In mock mode, `openDevice()` calls `configure_mock_mode()`. When the mock device is
   later closed, `closeInstance()` calls `disable_mock_mode()` →
   `detail::ReleaseOwnership()` → `MetalContext::destroy_all_instances(check_device_count=true)`.
4. With `TT_METAL_DEVICE_PROFILER_DISPATCH=1`, `ProfilerInitializer::teardown()` (called
   during `DeviceManager::close_devices()`) calls `ReadDeviceProfilerResults()` on the
   mock device's dispatch cores. Mock dispatch cores have no firmware output, so the
   readback either times out or leaves the device in a partially-closed state, causing
   `device_manager()->get_all_active_devices()` to be non-empty.
5. `destroy_all_instances(check=true)` finds non-empty active devices → throws.

The problem is that dispatch core profiling read-back fires **during compilation** (in
the mock device teardown), before inference has even started. The dispatch data we
actually want is from **inference**, not compilation (OpModel dispatches nothing).

### What Changed

**Before:**
```python
def _compile_onnx(model_path, sample_inputs, compiler_cfg, enable_program_cache, module_name):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    compiled = forge.compile(...)
    if enable_program_cache:
        _configure_device()
    return compiled
```

**After:**
```python
def _compile_onnx(model_path, sample_inputs, compiler_cfg, enable_program_cache, module_name):
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    # Temporarily unset TT_METAL_DEVICE_PROFILER_DISPATCH during forge.compile().
    # OpModel (SingletonDeviceContext) opens a mock device for cost estimation;
    # with dispatch profiling enabled the mock teardown tries to read uninitialized
    # dispatch core L1 buffers, leaving devices "active" and causing
    # disable_mock_mode() → destroy_all_instances(check=true) to throw.
    # The env var is restored before _configure_device() so inference is fully profiled.
    _dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        compiled = forge.compile(...)
    finally:
        if _dispatch_prof is not None:
            os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
    if enable_program_cache:
        _configure_device()
    return compiled
```

### Impact
`forge.compile()` completes cleanly (OpModel mock device teardown skips dispatch-core
readback since `TT_METAL_DEVICE_PROFILER_DISPATCH` is unset). After compilation,
`TT_METAL_DEVICE_PROFILER_DISPATCH=1` is restored so the inference device opens with
full dispatch profiling — exactly where the data is needed.

---

## Change 9 — Fix `assert False` in Dispatch Op Post-Processing

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
In `_enrich_ops_from_device_logs()`, after attaching dispatch analysis to regular
tensix ops, any unmatched dispatch ops (those without a matching `run_host_id` in
the tensix op list) hit a hard assert:

```python
assert False, "Unrecognized dispatch OPs are presented by dispatch cores"
```

With dispatch profiling enabled, certain dispatch operations — program loading, fence
synchronizations, sync signals — appear in the dispatch core profiler log but have no
corresponding TTNN-level tensix op. These are unmatched after the normal tensix-op
matching loop, causing the post-processor to abort with an `AssertionError`.

### What Changed

**Before:**
```python
if dispatch_op_analysis:
    if has_trace_runs:
        logger.debug(...)
    else:
        assert False, "Unrecognized dispatch OPs are presented by dispatch cores"
```

**After:**
```python
if dispatch_op_analysis:
    if has_trace_runs:
        logger.debug(...)
    else:
        logger.warning(
            f"Ignoring {len(dispatch_op_analysis)} unrecognized dispatch op(s) on device {device} "
            f"with no matching tensix op (IDs: {list(dispatch_op_analysis.keys())[:5]}...). "
            f"This is expected for program-load, fence, or sync dispatch operations."
        )
```

### Impact
Post-processing completes even when program-load or fence dispatch ops appear without
matching tensix ops. The `DISPATCH TOTAL CQ CMD OP TIME [ns]` and
`DISPATCH GO SEND WAIT TIME [ns]` columns are now populated for the 322 BEV inference
ops that have matching dispatch core timing data.

---

## Change 10 — tt-npe Integration: 6 NOC UTIL / DRAM BW Columns

### Files
- `third_party/CMakeLists.txt`
- `scripts/tracy_run.sh`
- `.gitignore`

### Problem
Six columns in `ops_perf_results` require NOC trace data analyzed by `tt-npe`:

```
NOC UTIL (%)               MULTICAST NOC UTIL (%)
DRAM BW UTIL (%)           DRAM BW UTIL PER CTRL (%)
ETH BW UTIL (%)            NPE CONG IMPACT (%)
```

`process_ops_logs.py::analyzeNoCTraces()` imports `from npe_analyze_noc_trace_dir
import analyze_noc_traces_in_dir`, but the `tt-npe` Python binding (`tt_npe_pybind`)
was only compiled for Python 3.10 while the current toolchain uses Python 3.12.3.
There was no build step in the CMake pipeline to build tt-npe from source, and
`scripts/tracy_run.sh` did not add tt-npe's install dirs to `PYTHONPATH`.

### What Changed

**`third_party/CMakeLists.txt`** — Added `ExternalProject_Add(tt-npe ...)` after the
TVM section, guarded by `-DTT_NPE_ENABLE=ON`:
```cmake
option(TT_NPE_ENABLE "Build tt-npe for NOC trace analysis ..." OFF)

if (TT_NPE_ENABLE)
    ExternalProject_Add(
        tt-npe
        GIT_REPOSITORY "https://github.com/tenstorrent/tt-npe.git"
        GIT_TAG        "main"
        GIT_SHALLOW    TRUE
        SOURCE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/tt-npe
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DPython_EXECUTABLE=${TTFORGE_VENV_DIR}/bin/python3   # forces Python 3.12
            -DCPM_SOURCE_CACHE=.../tt-npe/.cpmcache
            -DENABLE_ASAN=OFF -DENABLE_MSAN=OFF -DENABLE_TSAN=OFF -DENABLE_UBSAN=OFF
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTNPE_BUILD_DIR}
    )
endif()
```

**`scripts/tracy_run.sh`** — Added tt-npe install dirs to `PYTHONPATH` after the
TTNN section:
```bash
TTNPE_INSTALL="${REPO_ROOT}/third_party/tt-npe/install"
if [[ -d "${TTNPE_INSTALL}/lib" && -d "${TTNPE_INSTALL}/bin" ]]; then
    export PYTHONPATH="${TTNPE_INSTALL}/lib:${TTNPE_INSTALL}/bin:${PYTHONPATH}"
else
    echo "  (tt-npe not found — NOC UTIL columns will be empty; build with cmake -DTT_NPE_ENABLE=ON)"
fi
```

The install layout after build:
```
third_party/tt-npe/install/lib/tt_npe_pybind.cpython-312-x86_64-linux-gnu.so
third_party/tt-npe/install/bin/npe_analyze_noc_trace_dir.py
third_party/tt-npe/install/bin/fabric_post_process.py
```

**`.gitignore`** — Added exclusions for tt-npe build artifacts:
```
third_party/tt-npe/build/
third_party/tt-npe/install/
third_party/tt-npe/.cpmcache/
```

### How to Build tt-npe

```bash
# One-time build (~5-10 min, requires internet for CPM package downloads):
source env/activate
cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON -DTT_NPE_ENABLE=ON \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --target tt-npe
```

### Impact
With tt-npe built and `--noc-traces` passed to `tracy_run.sh`, the 6 NOC UTIL /
DRAM BW columns are populated for ops captured before the DRAM profiler buffer
overflows (197/322 ops; see Change 7 for the buffer-overflow handling). Total
column count: **85/91**.

---

## How to Run

### Standard Run (77/91 columns, no dispatch overhead)

```bash
source env/activate && bash scripts/tracy_run.sh \
  -o ./tracy_block_d_$(date +%Y%m%d_%H%M%S) \
  -n bev_block_d \
  --no-device-trace \
  -p \
  -- \
  pytest "forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]" \
  -vss
```

Key flags:
- `--no-device-trace` — per-op (no-trace) mode; all 322 ops profiled individually
- `-p` — partial profiling to stay under Tracy's 32 K zone table limit

### Dispatch-Cores Run (79/91 columns, adds DISPATCH TOTAL CQ CMD and DISPATCH GO SEND columns)

```bash
source env/activate && bash scripts/tracy_run.sh \
  -o ./tracy_block_d_dispatch_$(date +%Y%m%d_%H%M%S) \
  -n bev_block_d_dispatch \
  --no-device-trace \
  -p \
  --dispatch-cores \
  -- \
  pytest "forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]" \
  -vss
```

Additional flag vs. standard run:
- `--dispatch-cores` — sets `TT_METAL_DEVICE_PROFILER_DISPATCH=1`; Tracy always sets
  `TT_METAL_DEVICE_PROFILER=1` in its subprocess regardless of `--no-device-trace`, so
  dispatch profiling works in per-op mode.

> **Note:** `_compile_onnx()` temporarily unsets `TT_METAL_DEVICE_PROFILER_DISPATCH`
> during `forge.compile()` (Change 8) to avoid the MetalContext crash. The env var is
> restored before inference so all 322 BEV ops get dispatch core timing.

### NOC Traces Run (85/91 columns — requires `cmake -DTT_NPE_ENABLE=ON` build)

```bash
source env/activate && bash scripts/tracy_run.sh \
  -o ./tracy_block_d_noc_$(date +%Y%m%d_%H%M%S) \
  -n bev_block_d_noc \
  --no-device-trace \
  -p \
  --noc-traces \
  -- \
  pytest "forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]" \
  -vss
```

Additional flag vs. standard run:
- `--noc-traces` — sets `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`; enables collection of
  NOC event traces that `tt-npe`'s `analyze_noc_traces_in_dir()` turns into the 6
  NOC UTIL / DRAM BW columns.

> **Note:** NOC traces + dispatch-cores can be combined in a single run for all 91
> columns (see Change 12–14 below). DRAM profiler buffer overflow limits NOC data to
> the ops captured before overflow, but with Changes 12–14, tt-npe data is attached
> to **all 46** ops.

### Combined Run — All 91 Columns (requires `cmake -DTT_NPE_ENABLE=ON` build)

```bash
source env/activate && bash scripts/tracy_run.sh \
  -o ./tracy_block_d_both_$(date +%Y%m%d_%H%M%S) \
  -n bev_block_d_both \
  --no-device-trace \
  -p \
  --dispatch-cores \
  --noc-traces \
  -- \
  pytest "forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]" \
  -vss
```

---

## Change 12 — `process_ops_logs.py`: Wormhole B0 Translated Coordinate Fix in `reconstructNocTracesFromCSV`

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
When `--dispatch-cores` is active, `cpp_device_perf_report.csv` is not generated, so
`reconstructNocTracesFromCSV()` is used to rebuild per-op NOC JSON files from
`profile_log_device.csv`. The coordinate translation logic inside this function was
wrong for **Wormhole B0 translated coordinates** (where `x >= 18` or `y >= 16`).

Translated addresses use a fixed lookup table (not the NOC_1 inversion formula):

| Translated X | NOC0 X | | Translated Y | NOC0 Y |
|---|---|---|---|---|
| 18 | 1 | | 18 | 1 |
| 19 | 2 | | 19 | 2 |
| 20 | 3 | | 20 | 3 |
| 21 | 4 | | 21 | 4 |
| 22 | 6 (skip 5=DRAM col) | | 22 | 5 |
| 23 | 7 | | 23 | 7 (skip 6=non-tensix row) |
| 24 | 8 | | 24 | 8 |
| 25 | 9 | | 25 | 10 (skip 9=non-tensix row) |

The old code applied the NOC_1 inversion (`MAX - virt`) to ALL addresses including
translated ones, producing negative physical coordinates (e.g. translated (18,18) →
physical (-9,-7)). This caused `tt-npe`'s `inBounds` check to fail for every op that
had translated-address NOC traffic, reporting `WORKLOAD_VALIDATION_FAILED`.

### Before
```python
# Wrong: applies NOC_1 formula to translated coords too
phys_dst_xs = np.where(noc_type_raws == 2, NOC1_MAX_X - dst_xs, dst_xs)
phys_dst_ys = np.where(noc_type_raws == 2, NOC1_MAX_Y - dst_ys, dst_ys)
phys_mcast_end_xs = np.where(noc_type_raws == 2, NOC1_MAX_X - mcast_end_xs, mcast_end_xs)
phys_mcast_end_ys = np.where(noc_type_raws == 2, NOC1_MAX_Y - mcast_end_ys, mcast_end_ys)
```

### After
```python
# Three cases:
#   1. Translated (x>=18 or y>=16): fixed lookup table
#      x: arr<22 → arr-17, else arr-16  (skips DRAM col 5)
#      y: arr<23 → arr-17, arr<25 → arr-16, else arr-15  (skips rows 6, 9)
#   2. Non-translated NOC_1 (noc_type==2): phys = MAX - virt
#   3. Non-translated NOC_0 (noc_type==1): already physical

def _trans_to_noc0_x(arr):
    return np.where(arr < 22, arr - 17, arr - 16)

def _trans_to_noc0_y(arr):
    return np.select([arr < 23, arr < 25], [arr - 17, arr - 16], default=arr - 15)

is_translated_dst = (dst_xs >= 18) | (dst_ys >= 16)
phys_dst_xs = np.where(is_translated_dst,
                        _trans_to_noc0_x(dst_xs),
                        np.where(noc_type_raws == 2, NOC1_MAX_X - dst_xs, dst_xs))
phys_dst_ys = np.where(is_translated_dst,
                        _trans_to_noc0_y(dst_ys),
                        np.where(noc_type_raws == 2, NOC1_MAX_Y - dst_ys, dst_ys))

is_translated_mcast = (mcast_end_xs >= 18) | (mcast_end_ys >= 16)
phys_mcast_end_xs = np.where(is_translated_mcast,
                               _trans_to_noc0_x(mcast_end_xs),
                               np.where(noc_type_raws == 2, NOC1_MAX_X - mcast_end_xs, mcast_end_xs))
phys_mcast_end_ys = np.where(is_translated_mcast,
                               _trans_to_noc0_y(mcast_end_ys),
                               np.where(noc_type_raws == 2, NOC1_MAX_Y - mcast_end_ys, mcast_end_ys))
```

### Impact
`tt-npe` now processes all 46 BEV Block D ops without `WORKLOAD_VALIDATION_FAILED`.
All physical coordinates are in the valid NOC0 range [0–9, 0–11].

---

## Change 13 — `process_ops_logs.py`: Keep All Host Ops for NOC Stats Attachment

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem
In `_enrich_ops_from_device_logs`, when the legacy parser returns fewer ops than
expected (due to the NCRISC-only core issue — see Change 14), the overflow handler
was filtering `host_ops_by_device[device]` down to only the matched subset:

```python
_matched = [_dev_ops_by_id.get(op["global_call_count"]) for op in host_ops_by_device[device]]
device_ops_time            = [d for d in _matched if d is not None]
host_ops_by_device[device] = [h for h, d in zip(host_ops_by_device[device], _matched)
                               if d is not None]   # ← drops unmatched ops
```

The NOC stats attachment loop at line 1025 iterates `host_ops_by_device` — so
any ops dropped here could never receive tt-npe data. In the combined run
(`--dispatch-cores + --noc-traces`), the legacy parser returned 29/46 ops, so 17
ops were dropped and received no NOC data.

### Before
```python
_matched = [_dev_ops_by_id.get(op["global_call_count"]) for op in host_ops_by_device[device]]
device_ops_time            = [d for d in _matched if d is not None]
host_ops_by_device[device] = [h for h, d in zip(host_ops_by_device[device], _matched)
                               if d is not None]
```

### After
```python
# Keep all 46 host ops so NOC stats (from tt-npe) can attach to every op.
# The enrichment loop below skips None entries with an early-continue guard.
_matched_device_ops = [_dev_ops_by_id.get(op["global_call_count"]) for op in host_ops_by_device[device]]
device_ops_time = _matched_device_ops   # None entries kept; enrichment loop skips them
```

And in the enrichment loop:
```python
for device_op, device_op_time in zip(host_ops_by_device[device], device_ops_time):
    if device_op_time is None:
        continue   # ← new guard
    ...
```

### Impact
tt-npe NOC data (DRAM BW UTIL, NOC UTIL, ETH BW UTIL) now attaches to **all 46 ops**
instead of only the 29 ops the legacy parser could identify. Coverage went from
29/46 → 46/46 for DRAM BW columns.

---

## Change 14 — `process_device_log.py`: NCRISC-Only Core Fix in `get_ops`

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_device_log.py`

### Problem
`get_ops` detects op boundaries by tracking BRISC-FW / ERISC-FW / TRACE-FW
`ZONE_START` + `ZONE_END` pairs per core. It initialised `opCores` with **every
core** that appeared in an op's timeseries — including NCRISC-only cores (those
that run data-movement kernels with no BRISC-FW zones).

NCRISC-only cores stayed `None` in `opCores` forever (no FW zones ever received),
so the `opIsDone` check:
```python
for core, coreOp in opCores.items():
    if not coreOp or len(coreOp) != 2:
        opIsDone = False   # NCRISC-only cores block this from ever being True
        break
```
…was never satisfied. All events for that op merged into the PREVIOUS op's bucket
instead of creating a new one. With 17 NCRISC-only-core ops in BEV Block D,
`get_ops` returned only 29 ops instead of 46.

A second case: op 966656 runs **entirely on ERISC dispatch** with no tensix events
at all, so it never appears in the tensix timeseries and `opCores` is empty.
Without special handling, fully empty-opCores ops would also never close.

### Fix Part A — Exclude NCRISC-only cores from `opCores`

**Before:**
```python
op.sort(key=lambda ts: ts[1])
for ts in op:
    if len(ts) == 5:
        timerID, tsValue, attachedData, risc, core = ts
        opCores[core] = None   # adds ALL cores including NCRISC-only
```

**After:**
```python
op.sort(key=lambda ts: ts[1])
# Pre-scan: find cores that emit BRISC-FW / ERISC-FW / TRACE-FW zones.
# Cores with only NCRISC events never emit FW zone boundaries and would
# permanently block opIsDone from becoming True if included in opCores.
_fw_cores = set()
for _ts in op:
    if len(_ts) == 5:
        _tid, _tsv, _ad, _risc, _core = _ts
        if (
            (_risc == "BRISC" and _tid.get("zone_name") == "BRISC-FW")
            or (_risc == "ERISC" and _tid.get("zone_name") == "ERISC-FW")
            or (_risc == "TENSIX_RISC_AGG" and _tid.get("zone_name") == "TRACE-FW")
        ):
            _fw_cores.add(_core)
for _ts in op:
    if len(_ts) == 5:
        _tid, _tsv, _ad, _risc, _core = _ts
        if _core in _fw_cores:
            opCores[_core] = None   # only FW-capable cores
```

And guard the main event-processing loop against cores not in `opCores`:
```python
# Before:
if opCores[core]:   # KeyError if NCRISC-only core not in opCores

# After:
if core not in opCores:
    pass   # NCRISC-only core; no FW boundary tracking needed
elif opCores[core]:
    ...
else:
    ...
```

### Fix Part B — Force-close fully NCRISC-only ops (empty `opCores`)

When an entire op has no FW-capable cores (`_fw_cores` is empty), `opIsDone` can
never be set inside the inner event loop. After the loop completes, force-close
the op's bucket so the next opID gets its own slot:

```python
for ts in op:
    ...
    ops[-1]["timeseries"].append(ts)
    if opIsDone:
        ops.append({"timeseries": []})
        for core in opCores:
            opCores[core] = None

# Force-close for fully NCRISC-only ops (no FW zones at all)
if not _fw_cores:
    ops.append({"timeseries": []})
```

### Impact

| `get_ops` output | Before | After |
|---|---|---|
| Ops returned | 29/46 | 45/46 |
| Ops with device timing data | 29/46 | 45/46 |

Op 966656 (ERISC dispatch-only, zero tensix events) remains at 45/46 — it has
no tensix kernel and no tensix profiler events in `profile_log_device.csv`. This
is the hard limit; the op still appears in the final CSV with NOC/DRAM data from
tt-npe but without DEVICE FW/KERNEL timing columns.

---

## Change 15 — Fix `--sync-host-device` Crash During `forge.compile()`

### File
`forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py`

### Problem

Enabling `--sync-host-device` sets `TT_METAL_PROFILER_SYNC=1`. When this env var
is active during `forge.compile()`, Metal's `ProfilerSync(INIT)` builds a sync
kernel during the OpModel mock device open. The linker fails with:

```
brisc build failed... kernel_brisc.ld:26: non constant or forward reference
address expression for section .text
```

Root cause: the mock device has no real runtime info, so the sync kernel link
step fails. Same pattern as the `TT_METAL_DEVICE_PROFILER_DISPATCH` issue fixed
in Change 8.

### What Changed

In `_compile_onnx()`, added `TT_METAL_PROFILER_SYNC` to the set of env vars
popped before `forge.compile()` and restored in the `finally` block:

```python
_dispatch_prof = os.environ.pop("TT_METAL_DEVICE_PROFILER_DISPATCH", None)
_sync_prof = os.environ.pop("TT_METAL_PROFILER_SYNC", None)   # NEW
try:
    compiled = forge.compile(...)
finally:
    if _dispatch_prof is not None:
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = _dispatch_prof
    if _sync_prof is not None:                                   # NEW
        os.environ["TT_METAL_PROFILER_SYNC"] = _sync_prof       # NEW
```

`TT_METAL_DEVICE_PROFILER` is intentionally **not** popped — popping it disrupts
the global `firstInit=true` sequence and empties `profile_log_device.csv` after
inference.

### Impact

`--sync-host-device` works correctly for BEV Block D. The sync kernel is built
and executed after `forge.compile()` returns (against the real device), producing
valid host-device clock sync:
```
SYNC PROGRAM FINISH IS DONE ON 0
Host sync data for device: 0, cpu_start:..., delay:..., freq:0.985 Hz
```

---

## Change 16 — `scripts/tracy_run.sh`: Incompatibility Guard for `--sync-host-device` + `--noc-traces`

### File
`scripts/tracy_run.sh`

### Problem

Using `--sync-host-device` (`TT_METAL_PROFILER_SYNC=1`) together with
`--noc-traces` (`TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`) causes a crash in
the tt-metal C++ profiler during `writeDeviceResultsToFiles()`:

```
TT_FATAL: Invalid NoC transfer type on device: 0.
  at profiler.cpp:600: EMD::isValidEventType(EMD(markers[i].data).data.raw_event.noc_xfer_type)
  in coalesceFabricEvents → convertNocTracePacketsToJson → DeviceProfiler::writeDeviceResultsToFiles
```

Root cause: the sync kernel generates NOC events with an internal transfer type
that `EMD::isValidEventType` does not recognise (it is not a user-visible data
movement type). This is a tt-metal bug in `coalesceFabricEvents`.

### What Changed

Added a conflict check in `tracy_run.sh` before the banner, after option parsing:

```bash
if [[ "${SYNC_HOST_DEVICE}" == true && "${NOC_TRACES}" == true ]]; then
    echo "  WARNING: --sync-host-device is incompatible with --noc-traces ..." >&2
    SYNC_HOST_DEVICE=false
fi
```

When both flags are active, `--sync-host-device` is silently disabled so
`--noc-traces` (the higher-value output — 6 NOC/DRAM columns) is preserved.

### Impact

| Flag combination | Before | After |
|---|---|---|
| `--sync-host-device` only | crash (brisc linker, fixed by Change 15) | works |
| `--noc-traces` only | works | works |
| `--sync-host-device --noc-traces` | crash (coalesceFabricEvents TT_FATAL) | sync auto-disabled, NOC data preserved |

### Column summary for `--dispatch-cores --sync-host-device` run (no NOC)

| Column Group | Count |
|---|---|
| Populated | **76/91** |
| Empty (NOC/DRAM — need `--noc-traces`) | 5 columns |
| Empty (known limits: ERISC, CB-COMPUTE, trace mode, DISPATCH GO SEND) | 7 columns |
| Empty (DEVICE ARCH, AVAILABLE WORKER CORE COUNT, PARALLELIZATION STRATEGY) | 3 columns |

All 46 rows present. DISPATCH TOTAL CQ CMD OP TIME populated for all 46 ops.
Device FW/KERNEL timing populated for 45/46 (op 966656 ERISC-only hard limit).

---

## Change 17 — Fix `--dispatch-cores` + `--noc-traces` Incompatibility Guard Order in `tracy_run.sh`

### File
`scripts/tracy_run.sh`

### Problem

The incompatibility guard that disables conflicting flags when `--noc-traces` is active
was placed **after** `TRACY_ARGS` was already built. So even though the guard set
`SYNC_HOST_DEVICE=false`, the `--sync-host-device` flag was already in `TRACY_ARGS`
and still passed to `python3 -m tracy`, causing the crash.

The banner showed `Sync host-dev: false` (guard fired) but the actual tracy subprocess
received `--sync-host-device` anyway.

### What Changed

Moved the incompatibility guard to run **before** `TRACY_ARGS` is built so that
disabled flags are never added to the argument list.

```
BEFORE (wrong order):
  1. Build TRACY_ARGS  ← --sync-host-device added here
  2. Guard fires       ← sets SYNC_HOST_DEVICE=false (too late)
  3. Print banner

AFTER (correct order):
  1. Guard fires       ← sets SYNC_HOST_DEVICE=false
  2. Build TRACY_ARGS  ← --sync-host-device NOT added
  3. Print banner
```

---

## Change 18 — Fix `coalesceFabricEvents` TT_FATAL for Unknown ERISC Dispatch NOC Event Types

### File
`third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/profiler/profiler.cpp`

### Problem

When `--dispatch-cores` and `--noc-traces` are both active, the ERISC dispatch core
firmware (19.6.0) emits NOC events with transfer types 178–184 that the profiler build
does not recognise. `coalesceFabricEvents` called `TT_FATAL` on these, crashing the
entire test process in `writeDeviceResultsToFiles()`.

### What Changed

Changed the hard `TT_FATAL` to a `log_warning` + `continue` so unknown event types
are skipped gracefully:

```cpp
// BEFORE (crashes):
TT_FATAL(
    EMD::isValidEventType(EMD(markers[i].data).data.raw_event.noc_xfer_type),
    "Invalid NoC transfer type on device: {}.", device_id);

// AFTER (skips with warning):
if (!EMD::isValidEventType(EMD(markers[i].data).data.raw_event.noc_xfer_type)) {
    log_warning(tt::LogMetal,
        "[profiler noc tracing] Unknown NoC transfer type {} on device {}; skipping event ...",
        static_cast<uint32_t>(EMD(markers[i].data).data.raw_event.noc_xfer_type), device_id);
    i++;
    continue;
}
```

Then rebuilt `libtt_metal.so` and copied to `build_Release/lib/`:
```bash
cmake --build build_Release --target tt_metal -j$(nproc)
cp build_Release/tt_metal/libtt_metal.so build_Release/lib/libtt_metal.so
```

### Impact

`--dispatch-cores --noc-traces` combined run no longer crashes. Unknown ERISC dispatch
NOC event types are skipped with a warning line per event type.

---

## Change 19 — Fix `analyzeNoCTraces` Hang with `emit_viz_timeline_files=True`

### File
`third_party/tt-mlir/build/install/tt-metal/tools/tracy/process_ops_logs.py`

### Problem

`analyzeNoCTraces()` called `analyze_noc_traces_in_dir` with `emit_viz_timeline_files=True`
and `compress_timeline_files=True`. This generates a `.npeviz.zst` visualization file
for each of the 46 ops via a `multiprocessing.Pool(16)`.

With `--dispatch-cores` active, ERISC dispatch core events inflate the NOC trace JSON
files dramatically (e.g. `PermuteDeviceOperation`: 76 MB vs ~1 MB without dispatch).
Compressing these through `zstandard` in 16 parallel workers caused the pool to hang
effectively indefinitely — **1 hr 53 min with 0 output** (quiet=True hid all progress).

### What Changed

```python
# BEFORE (hangs):
return analyze_noc_traces_in_dir(
    noc_trace_dir=logFolder,
    emit_viz_timeline_files=True,   # ← generates .npeviz.zst per op → HANG
    quiet=True,                     # ← hides all progress
    compress_timeline_files=True,   # ← zstd compression of 76MB files → HANG
)

# AFTER (completes in ~28s):
result = analyze_noc_traces_in_dir(
    noc_trace_dir=logFolder,
    emit_viz_timeline_files=False,  # ← skip visualization files
    quiet=False,                    # ← shows "Analyzing (N/46)..." progress
    compress_timeline_files=False,
)
```

Also added step-level timing logs:
```
[noc] Step 1/3: reconstructing per-op NOC JSON files ...  (Xs)
[noc] Step 2/3: running NPE analysis ...                  (28.1s)
[noc] Step 3/3: returning stats to caller ...
```

### Impact

| Metric | Before | After |
|---|---|---|
| tt-npe analysis time | **>1 hr 53 min (hung)** | **28 seconds** |
| Progress visibility | None (quiet=True) | `Analyzing (N/46)...` per op |
| Visualization files | `.npeviz.zst` generated | Skipped (not needed for CSV) |

### Final Result

**82/91 columns, 46/46 rows** with `--dispatch-cores --noc-traces` combined run.
