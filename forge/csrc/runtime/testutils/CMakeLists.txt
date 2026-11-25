add_library(runtime_testutils STATIC testutils.cpp)
add_dependencies(runtime_testutils runtime tt-mlir)

set(_TTMLIR_INSTALL_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib")
set(_TTMLIR_TEST_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/runtime/test/ttnn")
set(_METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib")
set(_METAL_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal")

# Tracy client headers ship in the tt-metal install tree, not under any of the
# tt-mlir include dirs, so `#include "tracy/Tracy.hpp"` needs this explicitly.
set(_METAL_TRACY_INC
    "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/tt-metal/tt_metal/third_party/tracy/public")

# Whether the installed libtracy really carries the Tracy client. Built with
# ENABLE_TRACY=OFF (TT_RUNTIME_ENABLE_PERF_TRACE=OFF), tt-metal still installs a
# libtracy.so, but as a stub without the client entry point, so compiling with
# TRACY_ENABLE and linking it fails on tracy::GetProfiler_CFG_...(). Probe the
# symbol instead of assuming, so the runners build in both configurations.
set(_METAL_TRACY_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib/libtracy.so")
set(_TRACY_CLIENT_AVAILABLE OFF)
if(EXISTS "${_METAL_TRACY_LIB}")
    find_program(_NM_EXECUTABLE nm)
    if(_NM_EXECUTABLE)
        execute_process(
            COMMAND ${_NM_EXECUTABLE} -D --defined-only "${_METAL_TRACY_LIB}"
            OUTPUT_VARIABLE _TRACY_SYMS
            ERROR_QUIET)
        string(FIND "${_TRACY_SYMS}" "GetProfiler_CFG" _TRACY_SYM_POS)
        if(NOT _TRACY_SYM_POS EQUAL -1)
            set(_TRACY_CLIENT_AVAILABLE ON)
        endif()
    endif()
endif()
if(_TRACY_CLIENT_AVAILABLE)
    message(STATUS "emitc runners: Tracy client found - model_benchmark_profile signposts enabled")
else()
    message(STATUS "emitc runners: Tracy client absent (tt-metal built with ENABLE_TRACY=OFF) - "
                   "model_benchmark_profile signposts compile to no-ops")
endif()

target_include_directories(runtime_testutils PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/tt-mlir/runtime/include
    ${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/runtime/include
)

target_link_libraries(runtime_testutils PUBLIC coverage_config)
target_link_libraries(runtime_testutils PRIVATE TTMLIRRuntime TTRuntimeTTNNTestLib)

target_link_directories(runtime_testutils PRIVATE
    ${_TTMLIR_INSTALL_LIB}
    ${_TTMLIR_TEST_LIB}
)

target_compile_options(runtime_testutils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})

# ---------------------------------------------------------------------------
# emitc_pipeline — generic EmitC inference and benchmark runners
# ---------------------------------------------------------------------------

set(_EMITC_DIR ${CMAKE_SOURCE_DIR}/emitc_pipeline)
set(TT_METAL_ULFM_MPI_LIB_DIR "/opt/openmpi-v5.0.7-ulfm/lib" CACHE PATH
    "ULFM Open MPI lib dir libtt_metal links against")

# Python3::Python is overridden in forge/csrc/CMakeLists.txt to point to the
# uv-managed Python 3.12 (not the system 3.10 that FindPython3 resolves by default).
# Derive _PY312_LIBDIR for RPATH by querying the IMPORTED_LOCATION of that target.
get_target_property(_PY312_IMPORTED_LOC Python3::Python IMPORTED_LOCATION)
get_filename_component(_PY312_LIBDIR "${_PY312_IMPORTED_LOC}" DIRECTORY)
message(STATUS "emitc runners: Python 3.12 lib = ${_PY312_IMPORTED_LOC} (dir ${_PY312_LIBDIR})")

add_library(emitc_utils STATIC ${_EMITC_DIR}/emitc_utils.cpp ${_EMITC_DIR}/emitc_device_sync.cpp)
add_dependencies(emitc_utils runtime_testutils ttforge_csrc tt-mlir)

target_include_directories(emitc_utils PRIVATE ${_EMITC_DIR})
# emitc_device_sync.cpp declares tt::tt_metal::distributed::Synchronize itself; it
# needs only the header-only SubDeviceId / StrongType types from these two dirs.
target_include_directories(emitc_utils SYSTEM PRIVATE
    ${_METAL_SRC_DIR}/tt_metal/api
    ${_METAL_SRC_DIR}/tt_stl
)
# tt_stl needs C++20 (operator<=>, std::span); the rest of forge builds as C++17.
# The appended -std wins over the target's; std::optional/std::span mangle the same.
set_source_files_properties(${_EMITC_DIR}/emitc_device_sync.cpp PROPERTIES COMPILE_OPTIONS "-std=c++20")
target_compile_options(emitc_utils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})

target_link_libraries(emitc_utils PRIVATE
    runtime_testutils TTMLIRRuntime ttforge_csrc
    torch_python c10 ${TORCH_LIBRARIES} Python3::Python
)
target_link_directories(emitc_utils PRIVATE
    ${TORCH_LIB_DIR} ${_TTMLIR_INSTALL_LIB} ${_METAL_LIB_DIR} ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${_PY312_LIBDIR}
)

# Shared setup for the emitc runners. Pass WITH_TRACY after the source for a runner
# that emits Tracy signposts (model_benchmark_profile); the others stay free of any
# profiler dependency.
macro(add_emitc_runner _target _src)
    set(_runner_tracy OFF)
    foreach(_runner_arg ${ARGN})
        if(_runner_arg STREQUAL "WITH_TRACY")
            set(_runner_tracy ON)
        endif()
    endforeach()
    add_executable(${_target} ${_EMITC_DIR}/${_src})
    add_dependencies(${_target} emitc_utils runtime_testutils ttforge_csrc tt-mlir)
    target_include_directories(${_target} PRIVATE ${_EMITC_DIR})
    target_compile_options(${_target} PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS} ${CXXFLAGS})
    target_link_libraries(${_target} PRIVATE
        emitc_utils runtime_testutils TTMLIRRuntime
        ttforge_csrc tt_metal tt_stl
        torch_python c10 ${TORCH_LIBRARIES} Python3::Python
    )
    if(_runner_tracy)
        # SYSTEM: the Tracy client headers do not build clean under -Wextra -Werror.
        target_include_directories(${_target} SYSTEM PRIVATE ${_METAL_TRACY_INC})
        if(_TRACY_CLIENT_AVAILABLE)
            # The inlined client must match libtracy's build config: Tracy 0.13 mangles
            # it into the entry point (GetProfiler_CFG_E1_OD0_DI0_ML0_F0_DHT0_TF1), so a
            # mismatch is a link error, not a silent ABI break. TRACY_TIMER_FALLBACK is
            # the one non-default, forced ON by tt-metal's third_party CMake.
            target_compile_definitions(${_target} PRIVATE TRACY_ENABLE TRACY_TIMER_FALLBACK)
            target_link_libraries(${_target} PRIVATE tracy)
        endif()
    endif()
    target_link_options(${_target} PRIVATE -Wl,--copy-dt-needed-entries)
    # libtt_metal needs the ULFM Open MPI (MPIX_Comm_shrink/revoke). Without this,
    # ld can resolve libmpi.so.40 from the system Open MPI 4.x first, through a
    # library whose RUNPATH lacks the ULFM dir, and the link fails.
    if(EXISTS "${TT_METAL_ULFM_MPI_LIB_DIR}")
        target_link_options(${_target} PRIVATE "-Wl,-rpath-link,${TT_METAL_ULFM_MPI_LIB_DIR}")
    endif()
    target_link_directories(${_target} PRIVATE
        ${TORCH_LIB_DIR} ${_TTMLIR_INSTALL_LIB} ${_METAL_LIB_DIR} ${TTMLIR_TOOLCHAIN_DIR}/lib
        ${_PY312_LIBDIR}
    )
    set_target_properties(${_target} PROPERTIES
        INSTALL_RPATH "$ORIGIN:${TORCH_LIB_DIR}:${_TTMLIR_INSTALL_LIB}:${_METAL_LIB_DIR}:${TTMLIR_TOOLCHAIN_DIR}/lib:${_PY312_LIBDIR}"
        BUILD_RPATH   "${TORCH_LIB_DIR}:${_TTMLIR_INSTALL_LIB}:${_METAL_LIB_DIR}:${TTMLIR_TOOLCHAIN_DIR}/lib:${_PY312_LIBDIR}"
    )
endmacro()

add_emitc_runner(model_inference model_inference.cpp)
add_emitc_runner(model_benchmark model_benchmark.cpp)
add_emitc_runner(model_benchmark_profile model_benchmark_profile.cpp WITH_TRACY)
