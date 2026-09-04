#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

: << 'COMMENT'
Stage the craq-sim Quasar simulator and export the environment forge needs to
run against it instead of silicon.

    source ./scripts/quasar_sim_env.sh
    pytest -svv forge/test/mlir/test_quasar_sim.py

MUST be sourced, not executed -- the whole point is the exports, and they have to
be in place before the pytest process starts. tt-metal reads these once, when the
MetalContext singleton is first constructed at the first device touch, and forge's
TTSystem is a function-local static evaluated once as well. Whichever test runs
first therefore decides hardware-vs-simulator for the entire process, and there is
no way to change it afterwards. That is also why Quasar tests must run in their own
pytest invocation rather than mixed in with the ordinary suite.

INPUTS (all optional):
    QUASAR_SIM_DIR   directory holding the QSR libttsim.so
                     default: /proj_sw/user_dev/ctr-lelanchelian/craq-sim/src/_out/release_qsr

If libttsim.so is missing, build it from a craq-sim checkout with:
    TT_VERSION=2 ./make.py src/_out/release_qsr/libttsim.so
Public ttsim releases ship wormhole and blackhole builds only; the Quasar one has
to be built locally.
COMMENT

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "ERROR: this script must be sourced, not executed:" >&2
    echo "    source ./scripts/quasar_sim_env.sh" >&2
    exit 1
fi

_qsr_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_qsr_sim_dir="${QUASAR_SIM_DIR:-/proj_sw/user_dev/ctr-lelanchelian/craq-sim/src/_out/release_qsr}"
_qsr_so="${_qsr_sim_dir}/libttsim.so"
_qsr_soc_src="${_qsr_repo_root}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors/quasar_32_arch.yaml"
_qsr_soc_dst="${_qsr_sim_dir}/soc_descriptor.yaml"

if [ ! -f "${_qsr_so}" ]; then
    echo "ERROR: no Quasar simulator at ${_qsr_so}" >&2
    echo "Build one from a craq-sim checkout:" >&2
    echo "    TT_VERSION=2 ./make.py src/_out/release_qsr/libttsim.so" >&2
    echo "or point QUASAR_SIM_DIR at a directory that has one." >&2
    return 1
fi

# A wormhole/blackhole libttsim.so loads far enough to be confusing and then fails
# with 'MissingSpecification: libttsim_pci_mem_wr_bytes'. Catch it here instead.
if ! strings "${_qsr_so}" 2>/dev/null | grep -q 'libttsim_pci_mem_wr_bytes'; then
    echo "ERROR: ${_qsr_so} does not look like a QSR build." >&2
    echo "Rebuild it with TT_VERSION=2, or point QUASAR_SIM_DIR elsewhere." >&2
    return 1
fi

# UMD derives the SoC descriptor path from the .so's parent directory and expects
# the generic name -- see get_soc_descriptor_path_from_simulator_path in
# umd/device/simulation/simulation_chip.cpp. The arch-specific filename is not
# consulted, so the yaml has to be copied AND renamed.
if [ ! -f "${_qsr_soc_src}" ]; then
    echo "ERROR: missing SoC descriptor ${_qsr_soc_src}" >&2
    echo "Is third_party/tt-mlir/third_party/tt-metal checked out?" >&2
    return 1
fi
if [ ! -f "${_qsr_soc_dst}" ] || [ "${_qsr_soc_src}" -nt "${_qsr_soc_dst}" ]; then
    cp "${_qsr_soc_src}" "${_qsr_soc_dst}" || return 1
    echo "staged soc_descriptor.yaml <- quasar_32_arch.yaml"
fi

# Pointing this at a SoC descriptor makes UMD fail with 'Invalid YAML': it expects
# a cluster descriptor, which is a different schema. It is also not needed for a
# single-chip simulator, so make sure an inherited value cannot break the run.
if [ -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]; then
    echo "WARNING: unsetting inherited TT_METAL_MOCK_CLUSTER_DESC_PATH=${TT_METAL_MOCK_CLUSTER_DESC_PATH}" >&2
    unset TT_METAL_MOCK_CLUSTER_DESC_PATH
fi

export TT_METAL_SIMULATOR="${_qsr_so}"
export TT_METAL_SIMULATOR_HOME="${_qsr_sim_dir}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export ARCH_NAME=quasar
export CHIP_ARCH=quasar

# Not load-bearing -- rtoptions resolves its root dir from TT_METAL_RUNTIME_ROOT,
# which forge sets at import. Exported for parity with tt-metal's own Quasar
# qualification harness, some of whose tooling reads TT_METAL_HOME directly.
export TT_METAL_HOME="${_qsr_repo_root}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal"

# Without these a run is completely opaque: a deadlocked core and a simulator
# legitimately grinding through a kernel both sit at 100% CPU printing nothing.
# Both are read by libttsim.so itself, so they need no tt-metal patch -- unlike
# TT_METAL_SIM_CORE_WAIT_TIMEOUT_MS, which only exists on an unpinned tt-metal
# branch and is therefore inert here.
#
# The watchdog fires only on genuine no-progress: pending work, and no RISC-V
# instruction progress and no Tensix retirement for N simulated clocks. It does
# not catch a firmware loop that keeps retiring instructions -- use a wall-clock
# `timeout` for that.
export TTSIM_HANG_WATCHDOG_CLOCKS="${TTSIM_HANG_WATCHDOG_CLOCKS:-1000000}"
export TTSIM_PROGRESS_HEARTBEAT_CLOCKS="${TTSIM_PROGRESS_HEARTBEAT_CLOCKS:-2000000}"

echo "Quasar simulator environment ready:"
echo "  TT_METAL_SIMULATOR             = ${TT_METAL_SIMULATOR}"
echo "  TT_METAL_SIMULATOR_HOME        = ${TT_METAL_SIMULATOR_HOME}"
echo "  TT_METAL_SLOW_DISPATCH_MODE    = ${TT_METAL_SLOW_DISPATCH_MODE}"
echo "  ARCH_NAME / CHIP_ARCH          = ${ARCH_NAME} / ${CHIP_ARCH}"
echo "  TTSIM_HANG_WATCHDOG_CLOCKS     = ${TTSIM_HANG_WATCHDOG_CLOCKS}"
echo "  TTSIM_PROGRESS_HEARTBEAT_CLOCKS= ${TTSIM_PROGRESS_HEARTBEAT_CLOCKS}"
echo
echo "Run Quasar tests in their own pytest process:"
echo "  pytest -svv forge/test/mlir/test_quasar_sim.py"
echo
echo "Execution is slow -- minutes to tens of minutes per op. Wrap long runs in"
echo "\`timeout\`, and use \`py-spy dump --pid <pid> --native\` to tell compiling"
echo "from executing from stuck."

unset _qsr_repo_root _qsr_sim_dir _qsr_so _qsr_soc_src _qsr_soc_dst
