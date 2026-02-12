# CMakeLists.txt for C++ Cache Deletion Interceptor Library
# This builds libcache_interceptor.so for LD_PRELOAD-based monitoring

cmake_minimum_required(VERSION 3.16)

# Library target
add_library(cache_interceptor SHARED
    cache_deletion_interceptor.cpp
)

# Compiler flags for interceptor
target_compile_options(cache_interceptor PRIVATE
    -Wall
    -Wextra
    -O2
    -fPIC
    -g                  # Debug symbols for backtrace
)

# Linker flag: export symbols for backtrace (must be linker flag, not compile flag)
target_link_options(cache_interceptor PRIVATE
    -rdynamic
)

# C++17 standard
set_target_properties(cache_interceptor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(cache_interceptor PRIVATE
    dl          # dlsym, RTLD_NEXT
    pthread     # Thread safety
)

# Install to a known location
install(TARGETS cache_interceptor
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/lib
)

# Create a symlink in the root build directory for easy access
add_custom_command(TARGET cache_interceptor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE:cache_interceptor>
        ${CMAKE_BINARY_DIR}/libcache_interceptor.so
    COMMENT "Creating symlink to libcache_interceptor.so in build directory"
)

# Print information
message(STATUS "Cache Interceptor: Will be built at ${CMAKE_BINARY_DIR}/libcache_interceptor.so")
message(STATUS "Cache Interceptor: Use with LD_PRELOAD for automatic C++ deletion monitoring")
