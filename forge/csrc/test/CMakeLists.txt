if(TTFORGE_UNITTESTS_ENABLED)
    # GoogleTest is fetched via FetchContent in cmake/GoogleTest.cmake
    enable_testing()
else()
    message(STATUS "Unit tests are disabled. Enable with -DTTFORGE_UNITTESTS_ENABLED=ON")
    return()
endif()

# This will hold list of all executable test names.
set(TEST_NAMES)

# Picks up all the files in a given test dir and creates
# a test executable for the given directory.
function (add_unittest TEST_DIR)
    file(GLOB TEST_FILES "${TEST_DIR}/*.cpp")
    set(TEST_NAME "test_${TEST_DIR}")

    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_FILES})
    target_compile_options(${TEST_NAME} PRIVATE
        ${TTFORGE_CSRC_CFLAGS}
        ${CXXFLAGS}
        # Suppress deprecation warnings for pybind11 embed issues with Python 3.12 Issue Link:  https://github.com/tenstorrent/tt-forge-onnx/issues/2935#issuecomment-3297927503
        -Wno-deprecated-declarations
    )
    target_link_libraries(
        ${TEST_NAME}
        PRIVATE
            gtest
            gtest_main
            ttforge_csrc
            torch_python
            ${TORCH_LIBRARIES}
            ${TTFORGE_PYTHON_VERSION}
    )

    target_link_directories(${TEST_NAME} PRIVATE
        ${TORCH_LIB_DIR}
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    list(APPEND TEST_NAMES ${TEST_NAME})
    set(TEST_NAMES ${TEST_NAMES} PARENT_SCOPE)
endfunction()

# Add the unit tests (directories)
add_unittest("passes")
add_unittest("graph_lib")
add_unittest("verif")
add_unittest("ops")

add_custom_target(build_unit_tests
    COMMENT "Building unit tests..."
    DEPENDS
        ttforge_csrc
        ${TEST_NAMES}
)

add_custom_target(run_unit_tests
    COMMENT "Running unit tests..."
    COMMAND
        ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        build_unit_tests
)
