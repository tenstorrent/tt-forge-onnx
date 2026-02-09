cmake_minimum_required(VERSION 3.24.0)

project(ttforge LANGUAGES CXX)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "17" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18"))
    message(STATUS "CXX compiler in use: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(WARNING "The compiler you are using is not officially tested - you may run into issues. Official builds are created with Clang-17.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(Utils)

check_required_env_var(TTFORGE_PYTHON_VERSION)
check_required_env_var(TTFORGE_TOOLCHAIN_DIR)
check_required_env_var(TTFORGE_VENV_DIR)

set(TTFORGE_PYTHON_VERSION $ENV{TTFORGE_PYTHON_VERSION})
set(TTFORGE_TOOLCHAIN_DIR $ENV{TTFORGE_TOOLCHAIN_DIR})
set(TTFORGE_VENV_DIR $ENV{TTFORGE_VENV_DIR})

set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(TTMLIR_VENV_DIR $ENV{TTMLIR_VENV_DIR})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-DDEBUG)
endif()

add_library(coverage_config INTERFACE)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(coverage_config INTERFACE
    -O0        # no optimization
    -g         # generate debug info
    --coverage # sets all required flags
  )
  target_link_options(coverage_config INTERFACE --coverage)
endif(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

set(TTFORGE_CSRC_WARNINGS -Wall -Wextra -Wno-pragmas -Wno-unused-parameter)
set(CFLAGS_NO_WARN -DFMT_HEADER_ONLY)
set(TTFORGE_CSRC_CFLAGS ${CFLAGS_NO_WARN} ${TTFORGE_CSRC_WARNINGS} -DUTILS_LOGGER_PYTHON_OSTREAM_REDIRECT=1)

set(CONFIG_LDFLAGS "")

set(STATIC_LIB_FLAGS -fPIC)
set(SHARED_LIB_FLAGS -fPIC)

# Add GoogleTest if unit tests are enabled
option(TTFORGE_UNITTESTS_ENABLED "Enable unit tests with GoogleTest" ON)
include(GoogleTest)

add_subdirectory(third_party)
add_subdirectory(forge)
add_subdirectory(docs)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
### Install _C.so directly into the install directory (not ${CMAKE_INSTALL_PREFIX}/lib/)
### so that the python can find it. When we package this in the wheel, and it is installed
### the ${CMAKE_INSTALL_PREFIX} directory will end up in `venv/lib/python3.12/site-packages/forge`.
install(TARGETS ttforge_csrc LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

### Generate stubs for ttforge
### Run `cmake --build build -- make_stubs` to generate stubs
add_custom_target(make_stubs
    COMMAND pip install mypy==1.10
    COMMAND stubgen -m forge._C -m forge._C.autograd -m forge._C.graph -m forge._C.torch_device -m forge._C.runtime -m forge._C.ops -o forge -v
    COMMENT "Generating stubs for ttforge"
    USES_TERMINAL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
