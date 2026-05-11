add_library(runtime_testutils STATIC testutils.cpp)
add_dependencies(runtime_testutils runtime)

target_link_libraries(runtime_testutils PUBLIC coverage_config)

target_compile_options(runtime_testutils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})

set(_TTMLIR_INSTALL_LIB "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/install/lib")

# ---------------------------------------------------------------------------
# Standalone EmitC runner — loads evo50_forward.so and runs forward()
# ---------------------------------------------------------------------------
add_executable(run_evo50_forward
    ${CMAKE_SOURCE_DIR}/evo50_model/run_evo50_forward.cpp
)

add_dependencies(run_evo50_forward runtime_testutils ttforge_csrc)

target_compile_options(run_evo50_forward PRIVATE
    ${STATIC_LIB_FLAGS}
    ${TTFORGE_CSRC_CFLAGS}
    ${CXXFLAGS}
)

target_link_libraries(run_evo50_forward
    PRIVATE
        runtime_testutils
        ${_TTMLIR_INSTALL_LIB}/libTTMLIRRuntime.so
        ttforge_csrc
        torch_python
        c10
        ${TORCH_LIBRARIES}
)

target_link_directories(run_evo50_forward PRIVATE
    ${TORCH_LIB_DIR}
    ${_TTMLIR_INSTALL_LIB}
    ${TTMLIR_TOOLCHAIN_DIR}/lib
)

set_target_properties(run_evo50_forward PROPERTIES
    INSTALL_RPATH "$ORIGIN:${TORCH_LIB_DIR}:${_TTMLIR_INSTALL_LIB}:${TTMLIR_TOOLCHAIN_DIR}/lib"
    BUILD_RPATH   "${TORCH_LIB_DIR}:${_TTMLIR_INSTALL_LIB}:${TTMLIR_TOOLCHAIN_DIR}/lib"
)
