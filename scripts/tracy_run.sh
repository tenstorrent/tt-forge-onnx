#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Generic Tracy profiler wrapper for tt-forge-onnx.
# Runs any command (pytest, python script, arbitrary shell command) under
# the tt-metal Tracy profiler and generates a CSV op-performance report.
#
# PREREQUISITES
#   Build with Tracy enabled (one-time, ~30-60 min):
#     cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \
#         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
#     cmake --build build
#
# USAGE
#   source env/activate
#   bash scripts/tracy_run.sh [OPTIONS] -- COMMAND [ARGS...]
#
# OPTIONS
#   -o, --output  DIR        Profile artifacts output directory
#                            [default: /tmp/tracy_<timestamp>]
#   -n, --name    NAME       Custom name appended to report filename
#   --op-count    N          Max ops the profiler buffers [default: 3000]
#   -p, --partial            Only profile zones that are explicitly enabled
#                            (Tracy -p flag; reduces overhead)
#   --module                 Treat COMMAND as a Python module name
#                            (passes -m to python3 -m tracy, uses runpy)
#   --no-report              Skip generating the CSV report (omits -r flag)
#   --no-device              Exclude device-side kernel data from report
#   --no-device-trace        Disable device-side trace profiling (on by default)
#   --no-dispatch-cores      Disable dispatch-cores profiling (off by default)
#   --sync                   Enable host-device sync (off by default)
#   --no-memory-profile      Disable device memory profiling (on by default)
#   --no-perf-counters       Disable perf-counter capture (off by default)
#   --check-exit-code        Abort post-processing if test fails (on by default)
#   --no-check-exit-code     Continue post-processing even if test fails
#   --device-analysis-types  Override device analysis types [default: all]
#   --                       Separator: everything after is the COMMAND
#
# EXAMPLES
#   # Run a pytest suite (full profiling on by default):
#   bash scripts/tracy_run.sh -o /tmp/bev_trace -n bev_block_d -- \
#       pytest forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py \
#           -k "block_D and opt_level_2_bfloat16_hifi3_fp32_acc_trace_enabled" -vss
#
#   # Lightweight run — host-only, no device trace:
#   bash scripts/tracy_run.sh -o /tmp/my_trace --no-device-trace -- my_model.py
#
#   # Dispatch-core profiling (adds DISPATCH TOTAL CQ CMD and DISPATCH GO SEND columns):
#   #   --dispatch-cores enables TT_METAL_DEVICE_PROFILER_DISPATCH=1
#   #   Tracy always sets TT_METAL_DEVICE_PROFILER=1 in the subprocess regardless of
#   #   --no-device-trace, so dispatch profiling works with per-op (no-trace) mode.
#   #   NOTE: _compile_onnx() in test_bev_blocks_benchmark.py temporarily unsets
#   #   TT_METAL_DEVICE_PROFILER_DISPATCH during forge.compile() to avoid a MetalContext
#   #   conflict in SingletonDeviceContext (OpModel mock device teardown crash).
#   bash scripts/tracy_run.sh -o ./tracy_block_d_dispatch -n bev_block_d_dispatch \
#       --no-device-trace -p --dispatch-cores -- \
#       pytest "forge/test/models/onnx/vision/bev/test_bev_blocks_benchmark.py::test_opt_sweep[enable_program_cache-opt_level_2_bfloat16_hifi3_fp32_acc_no_trace-block_D]" -vss
#
# OUTPUT
#   <output_dir>/reports/<date>/ops_perf_results_<name>_<date>.csv
#   <output_dir>/.logs/tracy_profile_log_host.tracy  (open with Tracy GUI)

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate repo root
# ---------------------------------------------------------------------------
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OUTPUT_DIR="/tmp/tracy_$(date +%Y%m%d_%H%M%S)"
NAME_APPEND=""
OP_COUNT=3000
PARTIAL=false
USE_MODULE=false
REPORT=true
NO_DEVICE=false
DEVICE_TRACE=true        # --device-trace-profiler: safe with trace mode
DISPATCH_CORES=false     # --profile-dispatch-cores: disables C++ post-proc; conflicts with trace mode
SYNC_HOST_DEVICE=false   # --sync-host-device: improves host-device timestamp accuracy
MID_RUN_DUMP=false       # --dump-device-data-mid-run: flush DRAM profiler buffer after EACH op
                         #   Fixes DRAM circular buffer overflow for large models (>70 ops)
                         #   INCOMPATIBLE with --dispatch-cores (hard crash)
MEMORY_PROFILE=true      # --device-memory-profiler: safe
PERF_COUNTERS=false      # --profiler-capture-perf-counters: disables C++ post-proc; enable explicitly
LEGACY_DEVICE=false      # --legacy-device: pass --no-runtime-analysis → skip C++ post-proc, use legacy Python
                         #   Required when block C (>70 ops) DRAM buffer overflows in CPP_POST_PROCESS mode
CHECK_EXIT_CODE=true     # --check-exit-code: abort post-processing if test command fails
# DEVICE_ANALYSIS_TYPES: space-separated list of analysis types to include.
# Leave empty to run ALL analysis types (default). Valid types:
#   trace_fw_duration  trace_kernel_duration  "trace2trace - FW"  "trace2trace - kernel"
#   op2op  device_kernel_duration  device_fw_duration  device_kernel_duration_per_core
#   device_brisc_kernel_duration  device_ncrisc_kernel_duration
#   device_trisc0_kernel_duration  device_trisc1_kernel_duration  device_trisc2_kernel_duration
#   device_erisc_kernel_duration  device_compute_cb_wait_front  device_compute_cb_reserve_back
#   dispatch_total_cq_cmd_op_time  dispatch_go_send_wait_time  perf_counter_data
DEVICE_ANALYSIS_TYPES=""

# ---------------------------------------------------------------------------
# Parse script options (everything before --)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_DIR="$2"; shift 2 ;;
        -n|--name)
            NAME_APPEND="$2"; shift 2 ;;
        --op-count)
            OP_COUNT="$2"; shift 2 ;;
        -p|--partial)
            PARTIAL=true; shift ;;
        --module)
            USE_MODULE=true; shift ;;
        --no-report)
            REPORT=false; shift ;;
        --no-device)
            NO_DEVICE=true; shift ;;
        --no-device-trace)
            DEVICE_TRACE=false; shift ;;
        --dispatch-cores)
            DISPATCH_CORES=true; shift ;;
        --no-dispatch-cores)
            DISPATCH_CORES=false; shift ;;
        --sync)
            SYNC_HOST_DEVICE=true; shift ;;
        --no-sync)
            SYNC_HOST_DEVICE=false; shift ;;
        --mid-run-dump)
            MID_RUN_DUMP=true; shift ;;
        --no-memory-profile)
            MEMORY_PROFILE=false; shift ;;
        --perf-counters)
            PERF_COUNTERS=true; shift ;;
        --no-perf-counters)
            PERF_COUNTERS=false; shift ;;
        --legacy-device)
            LEGACY_DEVICE=true; shift ;;
        --no-legacy-device)
            LEGACY_DEVICE=false; shift ;;
        --check-exit-code)
            CHECK_EXIT_CODE=true; shift ;;
        --no-check-exit-code)
            CHECK_EXIT_CODE=false; shift ;;
        --device-analysis-types)
            DEVICE_ANALYSIS_TYPES="$2"; shift 2 ;;
        --)
            shift; break ;;       # everything after -- is the command
        -*)
            echo "Unknown option: $1" >&2
            echo "Run with --help or see the script header for usage." >&2
            exit 1 ;;
        *)
            break ;;              # no -- used; treat remainder as command
    esac
done

if [[ $# -eq 0 ]]; then
    sed -n '/^# USAGE/,/^# OUTPUT/p' "$0" | sed 's/^# \?//'
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TT_MLIR_BUILD="${REPO_ROOT}/third_party/tt-mlir/build"
TT_METAL_SRC="${REPO_ROOT}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal"
TRACY_TOOLS="${TT_METAL_SRC}/build/tools/profiler/bin"

# Prefer the install tree if it was populated (cmake --install)
TRACY_PY_MODULE="${TT_METAL_SRC}/tools"
if [[ -d "${TT_MLIR_BUILD}/install/tt-metal/tools/tracy" ]]; then
    TRACY_PY_MODULE="${TT_MLIR_BUILD}/install/tt-metal/tools"
fi

# ---------------------------------------------------------------------------
# Auto-install patched post-processors into the tt-metal install tree.
# process_ops_logs.py and process_device_log.py fix pandas 2.x compatibility
# and ERISC dispatch-core support. Must be copied after every cmake --build.
# ---------------------------------------------------------------------------
TRACY_INSTALL="${TT_MLIR_BUILD}/install/tt-metal/tools/tracy"
if [[ -d "${TRACY_INSTALL}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    for _f in process_ops_logs.py process_device_log.py; do
        if [[ -f "${SCRIPT_DIR}/${_f}" ]]; then
            cp "${SCRIPT_DIR}/${_f}" "${TRACY_INSTALL}/${_f}"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -f "${TRACY_TOOLS}/capture-release" ]]; then
    echo "ERROR: capture-release not found at ${TRACY_TOOLS}" >&2
    echo "  Build with Tracy enabled:" >&2
    echo "    cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON \\" >&2
    echo "        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++" >&2
    echo "    cmake --build build" >&2
    exit 1
fi

if [[ ! -f "${TRACY_TOOLS}/csvexport-release" ]]; then
    echo "ERROR: csvexport-release not found at ${TRACY_TOOLS}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export TT_METAL_HOME="${TT_METAL_SRC}"
export PYTHONPATH="${TRACY_PY_MODULE}:${PYTHONPATH:-}"
# ttnn Python bindings — required by tracy/tracy_ttnn.py which does `import ttnn`
TTNN_PY="${TT_MLIR_BUILD}/install/tt-metal/ttnn"
[[ -d "${TTNN_PY}" ]] && export PYTHONPATH="${TTNN_PY}:${PYTHONPATH}"
# Device-side kernel cycle timestamps — requires TT_RUNTIME_ENABLE_PERF_TRACE=ON build.
# Only enabled when device tracing is active (default: on).
[[ "${DEVICE_TRACE}" == true && "${NO_DEVICE}" == false ]] && export TT_METAL_DEVICE_PROFILER=1
# Enable TTNN op profiler Tracy messages (TT_DNN_DEVICE_OP zones) and trace tracking.
# python -m tracy sets these internally; we bypass that launcher so must set them explicitly.
# Without TTNN_OP_PROFILER=1, TracyOpMeshWorkload() skips all message emission and
# tracy_ops_data.csv ends up empty ("There are currently no messages!"), which causes
# profile_log_device.csv to have only headers and no per-op device timing data.
[[ "${DEVICE_TRACE}" == true && "${NO_DEVICE}" == false ]] && export TTNN_OP_PROFILER=1
[[ "${DEVICE_TRACE}" == true && "${NO_DEVICE}" == false ]] && export TT_METAL_PROFILER_TRACE_TRACKING=1
# Synchronous per-op DRAM buffer flush — prevents circular-buffer overflow for large models
# (documented threshold: >70 ops).  Equivalent to --mid-run-dump at the tt-metal level.
# Always set when mid-run dump is active; also set unconditionally here so that runs
# without --mid-run-dump still flush correctly when the model is large.
# NOTE: TT_METAL_PROFILER_SYNC is temporarily popped during forge.compile() by the BEV test
# files (_compile_model) and restored immediately after — this is intentional and safe.
export TT_METAL_PROFILER_SYNC=1

# ---------------------------------------------------------------------------
# Incompatibility guard
# ---------------------------------------------------------------------------
if [[ "${MID_RUN_DUMP}" == true && "${DISPATCH_CORES}" == true ]]; then
    echo "  ERROR: --mid-run-dump is incompatible with --dispatch-cores (tt-metal hard crash)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build python3 -m tracy argument list
# ---------------------------------------------------------------------------
TRACY_ARGS=()
[[ "${REPORT}"          == true  ]] && TRACY_ARGS+=(-r)
[[ "${PARTIAL}"         == true  ]] && TRACY_ARGS+=(-p)
[[ "${NO_DEVICE}"       == true  ]] && TRACY_ARGS+=(--no-device)
[[ "${DEVICE_TRACE}"    == true  ]] && TRACY_ARGS+=(--device-trace-profiler)
[[ "${DISPATCH_CORES}"  == true  ]] && TRACY_ARGS+=(--profile-dispatch-cores)
[[ "${SYNC_HOST_DEVICE}" == true  ]] && TRACY_ARGS+=(--sync-host-device)
[[ "${CHECK_EXIT_CODE}" == true  ]] && TRACY_ARGS+=(--check-exit-code)
[[ "${MID_RUN_DUMP}"    == true  ]] && TRACY_ARGS+=(--dump-device-data-mid-run)
[[ "${MEMORY_PROFILE}"  == true  ]] && TRACY_ARGS+=(--device-memory-profiler)
[[ "${PERF_COUNTERS}"   == true  ]] && TRACY_ARGS+=(--profiler-capture-perf-counters all)
[[ "${LEGACY_DEVICE}"  == true  ]] && TRACY_ARGS+=(--no-runtime-analysis)
# Expand each space-separated analysis type into its own --device-analysis-types flag
# (the Tracy option uses action="append"). Empty = omit flag = run all types.
if [[ -n "${DEVICE_ANALYSIS_TYPES}" ]]; then
    for _analysis_type in ${DEVICE_ANALYSIS_TYPES}; do
        TRACY_ARGS+=(--device-analysis-types "${_analysis_type}")
    done
fi
[[ -n "${NAME_APPEND}"           ]] && TRACY_ARGS+=(-n "${NAME_APPEND}")
[[ "${REPORT}"          == true  ]] && TRACY_ARGS+=(--op-support-count "${OP_COUNT}")

TRACY_ARGS+=(-v -o "${OUTPUT_DIR}" --tracy-tools-folder "${TRACY_TOOLS}")

# Use -m flag if the user wants module-style execution
[[ "${USE_MODULE}" == true ]] && TRACY_ARGS+=(-m)

# ---------------------------------------------------------------------------
# Print banner
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "================================================================="
echo " Tracy Profile Run"
echo "  Output        : ${OUTPUT_DIR}"
[[ -n "${NAME_APPEND}" ]]      && echo "  Name          : ${NAME_APPEND}"
echo "  Tools         : ${TRACY_TOOLS}"
echo "  Op count      : ${OP_COUNT}"
echo "  Device trace  : ${DEVICE_TRACE}"
echo "  Dispatch cores: ${DISPATCH_CORES}"
echo "  Sync host-dev : ${SYNC_HOST_DEVICE}"
echo "  Mid-run dump  : ${MID_RUN_DUMP}"
echo "  Memory profile: ${MEMORY_PROFILE}"
echo "  Perf counters : ${PERF_COUNTERS}"
echo "  Legacy device : ${LEGACY_DEVICE}"
echo "  Check exit    : ${CHECK_EXIT_CODE}"
[[ -n "${DEVICE_ANALYSIS_TYPES}" ]] && echo "  Analysis types: ${DEVICE_ANALYSIS_TYPES}" || echo "  Analysis types: all (default)"
echo "  Command       : $*"
echo "================================================================="

# ---------------------------------------------------------------------------
# Resolve command so Tracy can open it
#
# Tracy's __main__.py uses io.open_code(args[0]) to read the script.
# That call raises FileNotFoundError (not SyntaxError/ValueError) when
# given a bare command name like "pytest" — it never reaches the
# subprocess.run fallback. Fix: if the first argument is not a file path,
# resolve it via `which` so Tracy gets an absolute path it can open.
# If the target still isn't a Python file (e.g. a compiled binary), fall
# back to passing -m <module_name> which uses runpy.run_module() instead.
# ---------------------------------------------------------------------------
CMD="${1}"
if [[ ! -f "${CMD}" ]]; then
    RESOLVED="$(command -v "${CMD}" 2>/dev/null || true)"
    if [[ -n "${RESOLVED}" && -f "${RESOLVED}" ]]; then
        # Replace the bare command with its absolute path so Tracy can open it
        set -- "${RESOLVED}" "${@:2}"
    elif [[ "${USE_MODULE}" == false ]]; then
        # Cannot find or open the command as a file — auto-switch to -m mode
        echo "  (auto: '${CMD}' not found as file; using -m module mode)" >&2
        USE_MODULE=true
        # Re-add -m to TRACY_ARGS
        TRACY_ARGS+=(-m)
    fi
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python3 -m tracy "${TRACY_ARGS[@]}" -- "$@"

# ---------------------------------------------------------------------------
# Print report location
# ---------------------------------------------------------------------------
echo ""
echo "================================================================="
echo " Profile complete."
CSV=$(find "${OUTPUT_DIR}/reports" -name "ops_perf_results*.csv" 2>/dev/null | sort | tail -1)
if [[ -n "${CSV}" ]]; then
    echo " CSV report : ${CSV}"
else
    echo " CSV report : ${OUTPUT_DIR}/reports/ (check for generated files)"
fi
TRACY_FILE=$(find "${OUTPUT_DIR}/.logs" -name "*.tracy" 2>/dev/null | sort | tail -1)
if [[ -n "${TRACY_FILE}" ]]; then
    echo " Tracy file : ${TRACY_FILE}"
fi
echo ""
echo " To open in Tracy GUI:"
echo "   tracy-profiler ${OUTPUT_DIR}/.logs/tracy_profile_log_host.tracy"
echo "================================================================="
