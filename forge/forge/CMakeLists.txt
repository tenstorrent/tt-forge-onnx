add_custom_target(install_ttforge ALL
    COMMAND cd "${CMAKE_SOURCE_DIR}" && uv pip install --no-deps -e .
    COMMENT "Installing ttforge module")

add_dependencies(install_ttforge run_after_ttforge_csrc)
