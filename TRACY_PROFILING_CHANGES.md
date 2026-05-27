# Tracy Profiling Changes — BEV Block D (85/91 Columns)

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

## Final Column Count

| Run mode | Columns populated | Notes |
|----------|------------------|-------|
| Baseline (trace mode, no partial flag) | 17/65 | All host data missing, no per-core |
| After changes 1–7 (no-trace, `-p` flag) | **77/91** | Full 322 rows |
| After changes 1–9 (`--dispatch-cores` enabled) | **79/91** | + DISPATCH TOTAL CQ CMD OP TIME + DISPATCH GO SEND WAIT TIME |
| After changes 1–10 (`--noc-traces` + `tt-npe` built) | **85/91** | + 6 NOC UTIL / DRAM BW columns (197/322 ops due to buffer overflow) |

### The 6 Still-Empty Columns and Why

| Column(s) | Reason |
|-----------|--------|
| `DEVICE ERISC KERNEL DURATION [ns]` | BEV model has no Ethernet ops |
| `METAL TRACE ID`, `METAL TRACE REPLAY SESSION ID` | Only populated in trace mode; we profile no-trace |
| `DEVICE COMPUTE CB WAIT FRONT/BACK [ns]` | Requires `CB-COMPUTE-WAIT-FRONT` kernel zones; absent because all 322 BEV ops are DM-bound (no CB waits) |
| `PARALLELIZATION STRATEGY` | Field not present at top level of op JSON emitted by current tt-metal; only `move_op_parallelization_strategy` appears nested inside `attributes` for move ops |

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

> **Note:** NOC traces + dispatch-cores can be combined in a single run for 85+/91
> columns, but DRAM profiler buffer overflow limits per-core data to ~197/322 ops when
> NOC traces are active.
