add_library(runtime_testutils STATIC testutils.cpp)
add_dependencies(runtime_testutils runtime tt-mlir)

set(_TTMLIR_INSTALL_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib")
set(_TTMLIR_TEST_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/runtime/test/ttnn")
set(_METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib")
# Tracy client headers ship in the tt-metal install tree, not under any of the
# tt-mlir include dirs, so `#include "tracy/Tracy.hpp"` needs this explicitly.
set(_METAL_TRACY_INC
    "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/tt-metal/tt_metal/third_party/tracy/public"
)

# Whether the installed libtracy actually carries the Tracy client, rather than
# whether a tracy library merely exists. Built with ENABLE_TRACY=OFF, tt-metal
# still installs libtracy.so -- but as a stub with ~8 exported symbols and no
# client entry point -- so linking against it while compiling with TRACY_ENABLE
# fails on `tracy::GetProfiler_CFG_..._TF1()`. Probe the symbol instead of
# assuming, so the emitc runners build in both configurations.
set(_METAL_TRACY_LIB
    "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib/libtracy.so")
set(_TRACY_CLIENT_AVAILABLE OFF)
if(EXISTS "${_METAL_TRACY_LIB}")
    find_program(_NM_EXECUTABLE nm)
    if(_NM_EXECUTABLE)
        execute_process(
            COMMAND ${_NM_EXECUTABLE} -D --defined-only "${_METAL_TRACY_LIB}"
            OUTPUT_VARIABLE _TRACY_SYMS
            ERROR_QUIET
        )
        string(FIND "${_TRACY_SYMS}" "GetProfiler_CFG" _TRACY_SYM_POS)
        if(NOT _TRACY_SYM_POS EQUAL -1)
            set(_TRACY_CLIENT_AVAILABLE ON)
        endif()
    endif()
endif()
if(_TRACY_CLIENT_AVAILABLE)
    message(STATUS "emitc runners: Tracy client found - signposts enabled")
else()
    message(STATUS "emitc runners: Tracy client absent (tt-metal built with "
                   "ENABLE_TRACY=OFF) - signposts compile to no-ops")
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

# Python3::Python is overridden in forge/csrc/CMakeLists.txt to point to the
# uv-managed Python 3.12 (not the system 3.10 that FindPython3 resolves by default).
# Derive _PY312_LIBDIR for RPATH by querying the IMPORTED_LOCATION of that target.
get_target_property(_PY312_IMPORTED_LOC Python3::Python IMPORTED_LOCATION)
get_filename_component(_PY312_LIBDIR "${_PY312_IMPORTED_LOC}" DIRECTORY)
message(STATUS "emitc runners: Python 3.12 lib = ${_PY312_IMPORTED_LOC} (dir ${_PY312_LIBDIR})")

add_library(emitc_utils STATIC ${_EMITC_DIR}/emitc_utils.cpp)
add_dependencies(emitc_utils runtime_testutils ttforge_csrc tt-mlir)

target_include_directories(emitc_utils PRIVATE ${_EMITC_DIR})
target_compile_options(emitc_utils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})

target_link_libraries(emitc_utils PRIVATE
    runtime_testutils TTMLIRRuntime ttforge_csrc
    torch_python c10 ${TORCH_LIBRARIES} Python3::Python
)
target_link_directories(emitc_utils PRIVATE
    ${TORCH_LIB_DIR} ${_TTMLIR_INSTALL_LIB} ${_METAL_LIB_DIR} ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${_PY312_LIBDIR}
)

# Shared setup for model_inference and model_benchmark
macro(add_emitc_runner _target _src)
    add_executable(${_target} ${_EMITC_DIR}/${_src})
    add_dependencies(${_target} emitc_utils runtime_testutils ttforge_csrc tt-mlir)
    target_include_directories(${_target} PRIVATE ${_EMITC_DIR})
    # SYSTEM: the tracy client headers do not build clean under this
    # target's -Wextra -Werror (e.g. -Wsign-compare in TracyProfiler.hpp).
    target_include_directories(${_target} SYSTEM PRIVATE ${_METAL_TRACY_INC})
    target_compile_options(${_target} PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS} ${CXXFLAGS})
    # Tracy's macros compile to nothing without TRACY_ENABLE, and the client code
    # Tracy.hpp inlines here has to agree with the libtracy it calls into: Tracy
    # 0.13 mangles its build config into the entry point
    # (GetProfiler_CFG_E1_OD0_DI0_ML0_F0_DHT0_TF1), so a mismatched define is a
    # link error rather than a silent ABI break. TRACY_TIMER_FALLBACK is the one
    # non-default: tt_metal/third_party/CMakeLists.txt forces it ON. Everything
    # else (ON_DEMAND, DELAYED_INIT, MANUAL_LIFETIME, FIBERS, DISALLOW_HW_TIMER)
    # is at its default, hence the bare TRACY_ENABLE.
    # Only when the client is really there; Tracy's macros compile to nothing
    # without TRACY_ENABLE, which is exactly the right fallback.
    if(_TRACY_CLIENT_AVAILABLE)
        target_compile_definitions(${_target} PRIVATE TRACY_ENABLE TRACY_TIMER_FALLBACK)
    endif()
    target_link_libraries(${_target} PRIVATE
        emitc_utils runtime_testutils TTMLIRRuntime
        ttforge_csrc tt_metal tt_stl
        torch_python c10 ${TORCH_LIBRARIES} Python3::Python
    )
    if(_TRACY_CLIENT_AVAILABLE)
        target_link_libraries(${_target} PRIVATE tracy)
    endif()
    target_link_options(${_target} PRIVATE -Wl,--copy-dt-needed-entries)
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
