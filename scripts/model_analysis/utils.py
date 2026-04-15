# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import subprocess
from typing import Union, List, Optional
import shutil
from git import Repo
import ast

from forge.tvm_unique_op_generation import UniqueOperations


def check_path(directory_or_file_path: str):

    # Check if a file or directory exists, return True otherwise return False
    if os.path.exists(directory_or_file_path):
        logger.info(f"{directory_or_file_path} exists!")
        return True

    logger.info(f"{directory_or_file_path} does not exist.")
    return False


def create_python_package(package_path: str, package_name: str):
    package_path = os.path.join(package_path, package_name)
    if not check_path(package_path):
        os.makedirs(package_path, exist_ok=True)
    if not check_path(os.path.join(package_path, "__init__.py")):
        try:
            f = open(os.path.join(package_path, "__init__.py"), "x")
            f.close()
            logger.info(f"Created package in this path {package_path}")
        except FileExistsError:
            logger.info(f"__init__.py file already exists inside {package_path} path")


def dump_logs(log_files: Union[str, List[str]], content: str):
    if isinstance(log_files, str):
        log_files = [log_files]
    for log_file in log_files:
        log_file_dir_path = "/".join(log_file.split("/")[:-1])
        os.makedirs(log_file_dir_path, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(content)
            logger.info(f"Dumped test logs in {log_file}")


def collect_all_model_analysis_test(directory_or_file_path: str, marker: str, output_directory_path: str):
    """
    Collect all the tests that matches with the specified marker in a specified directory or file.
    """

    # Ensure the directory or file path exists
    assert check_path(
        directory_or_file_path
    ), f"The directory path for collecting test {directory_or_file_path} doesn't exists"

    logger.info(f"Collecting all the tests that matches with {marker} marker in {directory_or_file_path}")

    collected_test_outputs = ""
    try:
        # Run pytest to collect all the tests that matches with the specified marker
        result = subprocess.run(
            ["pytest", directory_or_file_path, "-m", marker, "--collect-only"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Append stdout and stderr to the collected outputs
        collected_test_outputs += "STDOUT:\n" + result.stdout
        collected_test_outputs += "STDERR:\n" + result.stderr

    except subprocess.CalledProcessError as e:
        collected_test_outputs += e.output

    # Save the collected test outputs to a file
    collected_test_file_path = os.path.join(output_directory_path, "collected_tests.txt")
    dump_logs(collected_test_file_path, collected_test_outputs)

    # Extract tests from the collected test outputs
    test_list = []
    with open(collected_test_file_path, "r") as collected_test_file:
        lines = collected_test_file.readlines()
        test_lines = False
        for line in lines:
            if "Automatic Model Analysis Collected tests:" in line:
                test_lines = True
            elif "Automatic Model Analysis Collected test count:" in line:
                test_lines = False
                break
            elif test_lines:
                test_list.append(str(line).replace("\n", ""))

    return test_list


def run_command(command: str):
    command_outputs = ""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        # Append stdout and stderr to the command outputs
        command_outputs += result.stdout + result.stderr

    except subprocess.CalledProcessError as e:
        command_outputs += e.output

    logger.info(f"Running the {command}\n{command_outputs}")


def run_precommit(directory_path: str):
    """Checks if pre-commit is installed and runs it on all files in the given directory."""

    # Check if pre-commit is installed
    if shutil.which("pre-commit") is None:
        logger.info("pre-commit is not installed. Installing...")
        run_command("uv pip install pre-commit")
    else:
        logger.info("pre-commit is already installed.")

    # Run pre-commit on all files in the directory
    run_command(f"pre-commit run --files $(find {directory_path} -type f)")


def remove_directory(directory_path: str):
    if check_path(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"The directory path '{directory_path}' and its contents have been removed.")
        except Exception as e:
            print(f"An error occurred while removing the directory path {directory_path}: {e}")


def extract_framework_from_test_file_path(test_file_path: str):
    if "forge/test/models/pytorch" in test_file_path:
        framework = "pytorch"
    elif "forge/test/models/onnx" in test_file_path:
        framework = "onnx"
    elif "forge/test/models/paddlepaddle" in test_file_path:
        framework = "paddlepaddle"
    elif "forge/test/models/jax" in test_file_path:
        framework = "jax"
    elif "forge/test/models/tensorflow" in test_file_path:
        framework = "tensorflow"
    else:
        framework = "unknown"
    return framework


def extract_test_file_path_and_test_case_func(test_case: str):
    """
    Extracts the test file path and test case function name from a given test case string.

    Args:
        test_case (str): A test case string in the format 'file_path::test_function'.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - test_file_path (str or None): The extracted file path if present.
            - test_case_func (str or None): The extracted test case function name if present.
    """
    test_file_path = None
    test_case_func = None

    # Check if test case contains '::' separator
    if "::" in test_case:
        test_file_path, test_case_func = test_case.split("::", 1)  # Splitting into two parts

    return test_file_path, test_case_func


def filter_unique_operations(unique_operations: UniqueOperations, ops_to_filter: Optional[List[str]] = None):
    """
    Select and return only the unique operations whose forge operation names match those specified in `ops_to_filter`.
    """
    if not unique_operations or ops_to_filter is None or not ops_to_filter:
        return unique_operations

    if not any(
        [
            True if forge_op_function_name.split(".")[-1] in ops_to_filter else False
            for forge_op_function_name in unique_operations.keys()
        ]
    ):
        raise ValueError(f"Provided op names {filter_ops} are not found in unique ops extracted across all models")

    return {
        forge_op_function_name: unique_operands_and_opargs_opmetadata
        for forge_op_function_name, unique_operands_and_opargs_opmetadata in unique_operations.items()
        if forge_op_function_name.split(".")[-1] in ops_to_filter
    }


def filter_tests(tests: List[str], tests_to_filter: Optional[List[str]] = None) -> List[str]:
    """
    Return a list of tests that match any of the filter tests list.
    """
    if not tests or tests_to_filter is None or not tests_to_filter:
        return tests

    filtered_tests = [test for test in tests if any(filter_test in test for filter_test in tests_to_filter)]
    if not filtered_tests:
        raise ValueError("None of the specified tests match the collected model analysis tests.")

    filtered_tests.sort()

    return filtered_tests


def extract_models_ops_test_params(pytest_file_path: str):
    """
    Parse a pytest file to extract test parameters for Forge modules and their operations.

    For each entry in 'forge_modules_and_shapes_dtypes_list', returns a dict containing:
      - module_name: Name of the ForgeModule subclass
      - forge_op_name: The fully-qualified forge.op.* call string
      - operand_names: Names of operands (variable or constant/parameter key)
      - operand_shapes: List of tuple shapes passed to each operand
      - operand_types: Classification: 'Activation', 'Constant', or 'Parameter'
      - operand_dtypes: Corresponding dtype expressions or strings
      - op_args: Keyword arguments passed to the operation call
      - model_names: List of model names from test metadata
      - pcc: Allowed post-conversion change threshold from metadata
      - max_int: Maximum integer value from metadata
      - markers: pytest.mark annotations with name and optional reason

    Args:
        pytest_file_path (str): Path to the pytest file (.py) to parse.

    Returns:
        List[dict]: Each dict contains all extracted fields for one test parameter.
    """

    # Read file content and parse into an AST for static analysis
    with open(pytest_file_path, "r") as f:
        file_content = f.read()
    ast_tree = ast.parse(file_content)

    # Build a mapping from class names to their AST.ClassDef nodes
    class_defs = {node.name: node for node in ast_tree.body if isinstance(node, ast.ClassDef)}

    # Pre-extract constants and parameters from each class's __init__ method
    init_data = {}
    for class_name, class_node in class_defs.items():
        constants_map = {}
        parameters_map = {}

        # Locate the __init__ FunctionDef inside the class
        for stmt in class_node.body:
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"):
                continue
            # Iterate over statements in __init__ to find add_constant / add_parameter calls
            for init_call in stmt.body:
                if not (isinstance(init_call, ast.Expr) and isinstance(init_call.value, ast.Call)):
                    continue
                call_node = init_call.value

                # Handle add_constant(...) calls
                if isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "add_constant":
                    const_name = ast.literal_eval(call_node.args[0])
                    # Extract 'shape' keyword argument
                    const_shape = next(
                        (ast.literal_eval(kw.value) for kw in call_node.keywords if kw.arg == "shape"), None
                    )
                    # Extract 'dtype' keyword argument as source code string
                    const_dtype = next((ast.unparse(kw.value) for kw in call_node.keywords if kw.arg == "dtype"), None)
                    constants_map[const_name] = (const_shape, const_dtype)

                # Handle add_parameter(name, forge.Parameter(*shape, dev_data_format=...))
                elif isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "add_parameter":
                    param_name = ast.literal_eval(call_node.args[0])
                    parameter_ctor = call_node.args[1]
                    # Starred args hold the shape tuple
                    param_shape = next(
                        (ast.literal_eval(arg.value) for arg in parameter_ctor.args if isinstance(arg, ast.Starred)),
                        None,
                    )
                    # Extract 'dev_data_format' keyword from Parameter constructor
                    param_dtype = next(
                        (ast.unparse(kw.value) for kw in parameter_ctor.keywords if kw.arg == "dev_data_format"), None
                    )
                    parameters_map[param_name] = (param_shape, param_dtype)

        # Store extracted init data per class
        init_data[class_name] = {
            "constants": constants_map,
            "parameters": parameters_map,
        }

    # List to collect extracted parameter info
    results = []

    # Find the assignment to forge_modules_and_shapes_dtypes_list
    for node in ast_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "forge_modules_and_shapes_dtypes_list"):
                continue

            # node.value should be a List of tuples or pytest.param() calls
            for list_element in node.value.elts:
                # Collect pytest.mark decorators
                marker_list = []

                # Unwrap pytest.param(...) if used
                if (
                    isinstance(list_element, ast.Call)
                    and isinstance(list_element.func, ast.Attribute)
                    and list_element.func.attr == "param"
                ):
                    param_call = list_element
                    # Extract any marks kwarg
                    for kw in param_call.keywords:
                        if kw.arg == "marks":
                            mark_nodes = kw.value.elts if isinstance(kw.value, (ast.List, ast.Tuple)) else [kw.value]
                            for mark_node in mark_nodes:
                                # Handle mark with reason arg
                                if isinstance(mark_node, ast.Call) and isinstance(mark_node.func, ast.Attribute):
                                    mark_name = mark_node.func.attr
                                    mark_reason = None
                                    # Check for reason=...
                                    for mk_kw in mark_node.keywords:
                                        if mk_kw.arg == "reason":
                                            mark_reason = ast.literal_eval(mk_kw.value)
                                    marker_list.append(
                                        {
                                            "marker_name": mark_name,
                                            "reason": mark_reason,
                                        }
                                    )
                                elif isinstance(mark_node, ast.Attribute):
                                    mark_name = mark_node.attr
                                    marker_list.append({"marker_name": mark_name, "reason": None})

                    # The test tuple is the first argument to param()
                    param_tuple = param_call.args[0]
                elif isinstance(list_element, ast.Tuple):
                    param_tuple = list_element
                else:
                    # Skip unexpected elements
                    continue

                # Unpack the tuple: (ModuleClass, [ (shape,dtype)... ], metadata_dict)
                module_class_name = ast.unparse(param_tuple.elts[0])
                shapes_and_dtypes = [
                    (ast.literal_eval(s.elts[0]), ast.unparse(s.elts[1])) for s in param_tuple.elts[1].elts
                ]
                metadata = ast.literal_eval(param_tuple.elts[2])

                # Extract common metadata fields
                model_names = metadata.get("model_names", [])
                pcc_value = metadata.get("pcc", None)
                max_int_value = metadata.get("max_int", None)
                op_args = metadata.get("args", {})
                default_df_override = metadata.get("default_df_override", None)
                if default_df_override is not None:
                    op_args["default_df_override"] = default_df_override

                # Locate the forward() method to find the forge.op.* call
                class_node = class_defs.get(module_class_name)
                op_call_node = None
                for fn in class_node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                        # Walk AST under forward() to find the op invocation
                        for stmt in ast.walk(fn):
                            if (
                                isinstance(stmt, ast.Call)
                                and isinstance(stmt.func, ast.Attribute)
                                and isinstance(stmt.func.value, ast.Attribute)
                                and isinstance(stmt.func.value.value, ast.Name)
                                and stmt.func.value.value.id == "forge"
                                and stmt.func.value.attr == "op"
                            ):
                                op_call_node = stmt
                                break
                        break

                # Prepare to extract operand details from the op call
                shape_iter = iter(shapes_and_dtypes)
                operand_shapes = []
                operand_types = []
                operand_dtypes = []
                operand_names = []
                const_map = init_data[module_class_name]["constants"]
                param_map = init_data[module_class_name]["parameters"]

                for operand in op_call_node.args[1:]:
                    # Activation inputs are plain variables
                    if isinstance(operand, ast.Name):
                        operand_types.append("Activation")
                        operand_names.append(operand.id)
                        shape, dtype = next(shape_iter)
                        operand_shapes.append(tuple(shape))
                        operand_dtypes.append(dtype)

                    # Constants or Parameters fetched via self.get_*()
                    elif isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute):
                        if operand.func.attr == "get_constant":
                            operand_types.append("Constant")
                            const_key = ast.literal_eval(operand.args[0])
                            operand_names.append(const_key)
                            shape, dtype = const_map.get(const_key, (None, None))
                            operand_shapes.append(tuple(shape))
                            operand_dtypes.append(dtype)
                        elif operand.func.attr == "get_parameter":
                            operand_types.append("Parameter")
                            param_key = ast.literal_eval(operand.args[0])
                            operand_names.append(param_key)
                            shape, dtype = param_map.get(param_key, (None, None))
                            operand_shapes.append(tuple(shape))
                            operand_dtypes.append(dtype)

                # Aggregate all extracted details
                results.append(
                    {
                        "module_name": module_class_name,
                        "forge_op_name": ast.unparse(op_call_node.func),
                        "operand_types": operand_types,
                        "operand_shapes": operand_shapes,
                        "operand_dtypes": operand_dtypes,
                        "operand_names": operand_names,
                        "op_args": op_args,
                        "model_names": model_names,
                        "pcc": pcc_value,
                        "max_int": max_int_value,
                        "markers": marker_list,
                    }
                )

    return results


def find_dirs_with_files(root: str, extensions: List[str] = None) -> List[str]:
    """
    Recursively search under `root`, returning directories that contain
    at least one file ending with any of the `extensions`.
    """
    if extensions is None:
        extensions = [".json", ".xlsx"]

    found = []
    for dirpath, _, filenames in os.walk(root):
        # Check if any filename ends with one of our extensions
        if any(fname.lower().endswith(ext) for fname in filenames for ext in extensions):
            found.append(dirpath)

    return sorted(found)
