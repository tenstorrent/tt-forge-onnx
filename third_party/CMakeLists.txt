### Build tt-mlir ###
include(ExternalProject)

# Tracy profiling option. Set to ON to build tt-metal and tt-mlir with
# TRACY_ENABLE instrumentation for device-side op timing and host call stacks.
# Requires a full rebuild of tt-mlir and tt-metal (~30-60 min).
# Usage: cmake -DTT_RUNTIME_ENABLE_PERF_TRACE=ON ...
option(TT_RUNTIME_ENABLE_PERF_TRACE "Enable Tracy profiling in tt-metal/tt-mlir" OFF)

set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib")

set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/build")
set(TTMLIR_INSTALL_PREFIX "${TTMLIR_BUILD_DIR}/install")

ExternalProject_Add(
    tt-mlir
    SOURCE_DIR ${TTMLIR_SOURCE_DIR}
    BINARY_DIR ${TTMLIR_BUILD_DIR} # for some reason BUILD_DIR for ExternalProject_Add doesn't exist, but they use BINARY_DIR instead
    INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component SharedLib && ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component TTNNStandalone
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
        -DTTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR}
        -DTTMLIR_ENABLE_RUNTIME=ON
        -DTTMLIR_ENABLE_RUNTIME_TESTS=ON  # used for emitc testing
        -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF
        -DTT_RUNTIME_DEBUG=${TTMLIR_RUNTIME_DEBUG}
        -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
        -DTTMLIR_ENABLE_OPMODEL=ON
        -DTTMLIR_ENABLE_EXPLORER=OFF
        -DTT_USE_SYSTEM_SFPI=OFF
        -DTT_RUNTIME_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
)

install(DIRECTORY ${TTMLIR_INSTALL_PREFIX}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

### end build tt-mlir ###

### Build TVM ###

add_custom_target(build_tvm ALL
    COMMAND bash -c ${CMAKE_CURRENT_SOURCE_DIR}/tvm/install.sh
    COMMENT "Installing TVM"
    USES_TERMINAL
)

### end build TVM ###

### Build tt-npe (NOC perf estimator — adds 6 NOC UTIL / DRAM BW columns to Tracy CSV) ###
#
# tt-npe provides the Python binding tt_npe_pybind and the helper script
# npe_analyze_noc_trace_dir.py.  process_ops_logs.py imports these (via
# analyzeNoCTraces()) to populate the following ops_perf_results columns:
#   NOC UTIL (%)  MULTICAST NOC UTIL (%)  DRAM BW UTIL (%)
#   DRAM BW UTIL PER CTRL (%)  ETH BW UTIL (%)  NPE CONG IMPACT (%)
#
# Enable: cmake -DTT_NPE_ENABLE=ON ...
# Run:    bash scripts/tracy_run.sh --noc-traces ...
#
# Install layout after build:
#   third_party/tt-npe/install/lib/tt_npe_pybind.cpython-312-x86_64-linux-gnu.so
#   third_party/tt-npe/install/bin/npe_analyze_noc_trace_dir.py
#   third_party/tt-npe/install/bin/fabric_post_process.py
# scripts/tracy_run.sh automatically prepends both dirs to PYTHONPATH when they exist.

option(TT_NPE_ENABLE "Build tt-npe for NOC trace analysis (adds 6 NOC UTIL / DRAM BW columns to Tracy ops_perf_results CSV); use with --noc-traces in scripts/tracy_run.sh; requires TT_RUNTIME_ENABLE_PERF_TRACE=ON" OFF)

if (TT_NPE_ENABLE)
    set(TTNPE_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/tt-npe")
    set(TTNPE_BUILD_DIR   "${TTNPE_SOURCE_DIR}/build")
    set(TTNPE_INSTALL_DIR "${TTNPE_SOURCE_DIR}/install")
    set(TTNPE_PYTHON      "${TTFORGE_VENV_DIR}/bin/python3")

    file(MAKE_DIRECTORY ${TTNPE_BUILD_DIR})

    ExternalProject_Add(
        tt-npe
        GIT_REPOSITORY "https://github.com/tenstorrent/tt-npe.git"
        GIT_TAG        "main"
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SOURCE_DIR     ${TTNPE_SOURCE_DIR}
        BINARY_DIR     ${TTNPE_BUILD_DIR}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_INSTALL_PREFIX=${TTNPE_INSTALL_DIR}
            -DPython_EXECUTABLE=${TTNPE_PYTHON}
            -DCPM_SOURCE_CACHE=${TTNPE_SOURCE_DIR}/.cpmcache
            -DENABLE_ASAN=OFF
            -DENABLE_MSAN=OFF
            -DENABLE_TSAN=OFF
            -DENABLE_UBSAN=OFF
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTNPE_BUILD_DIR}
    )

    message(STATUS "tt-npe enabled — NOC UTIL / DRAM BW columns will be populated.")
    message(STATUS "  Source dir  : ${TTNPE_SOURCE_DIR}")
    message(STATUS "  Install dir : ${TTNPE_INSTALL_DIR}")
    message(STATUS "  Python      : ${TTNPE_PYTHON}")
    message(STATUS "  Run with    : bash scripts/tracy_run.sh --noc-traces ...")
endif()

### end build tt-npe ###
