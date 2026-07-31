add_library(runtime_testutils STATIC testutils.cpp)
add_dependencies(runtime_testutils runtime tt-mlir)

set(_TTMLIR_INSTALL_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib")
set(_TTMLIR_TEST_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/runtime/test/ttnn")
set(_METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib")

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
    target_compile_options(${_target} PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS} ${CXXFLAGS})
    target_link_libraries(${_target} PRIVATE
        emitc_utils runtime_testutils TTMLIRRuntime
        ttforge_csrc tt_metal tt_stl
        torch_python c10 ${TORCH_LIBRARIES} Python3::Python
    )
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
