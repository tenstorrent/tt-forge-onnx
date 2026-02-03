### Build tt-mlir ###
include(ExternalProject)

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
