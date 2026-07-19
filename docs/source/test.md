# Testing

This page describes how to run different kinds of tests in the `tt-forge-onnx` project. If you haven't built the project yet,
please refer to the [Build](getting_started_build_from_source.md) page.

## Unit tests

To build the unit tests, run the following command:

```sh
cmake --build build -- build_unit_tests
```

To run the unit tests (this will also build the tests if they are not built):

```sh
cmake --build build -- run_unit_tests
```

> **Note:** The unit tests are built in the `build/forge/csrc/test` directory. From there, you can run targeted tests directly.
> - For example, to run all the tests defined in `forge/csrc/test/passes/` use: `./build/forge/csrc/test/test_passes`
> - You can further filter the tests by using the `--gtest_filter` flag:
>   ```sh
>   ./build/forge/csrc/test/test_passes --gtest_filter=MMFuseBias/MMFuseBias.mm_fuse_bias/3
>   ```

## End to end tests

For running the end-to-end tests we use the `pytest` framework. To run these tests, you need to be on a machine with a Tenstorrent
Wormhole device. Also, we are still in the process of cleaning up the old tests, so not all tests are working. For a list of green
tests, consult `pytest.ini`.

> **Note:** Make sure that you have activated the python environment before running the tests.

To run all tests defined in `/test/mlir/test_ops.py` use:

```sh
pytest -svv forge/test/mlir/test_ops.py
```

To run a specific test, use the following:

```sh
pytest -svv forge/test/mlir/test_ops.py::test_add
```

> - The `-svv` flag is optional and used to display more information about the test run.

## Single operator E2E tests

Single operator E2E tests consists of pre configured collections of in-depth tests for each operator according to test plan.
Tests include small models consisting of single operator with or without combination with few other operators.
More details about test plan available on [Test template page](https://github.com/tenstorrent/tt-forge-onnx/blob/main/forge/test/operators/test_plan_template.md)

To start interacting with test sweeps framework load helper commands via

```sh
source forge/test/operators/pytorch/test_commands.sh
```

Available commands

| Command               | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `print_help`          | Print commands and current query parameters.                          |
| `print_query_docs`    | Print docs for all available query parameters.                        |
| `print_params`        | Print current query parameters values.                                |
| `select_test_query`   | Select test_query pytest function.                                    |
| `select_test_push`    | Select test_push pytest function.                                     |
| `pytest`              | Run all tests or subset of test plan based on a query parameters.     |
| `with-params pytest`  | Print params before and after test run.                               |
| `export_tests`        | Export tests from test plan to JSON file based on a query parameters. |

Full list of supported query parameters

| Parameter             | Description                                                                                   | Supported by commands                 |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------- |
| OPERATORS             | List of operators                                                                             | test_query                            |
| FILTERS               | List of lambda filters                                                                        | test_query                            |
| INPUT_SOURCES         | List of input sources                                                                         | test_query                            |
| INPUT_SHAPES          | List of input shapes                                                                          | test_query                            |
| DEV_DATA_FORMATS      | List of dev data formats                                                                      | test_query                            |
| MATH_FIDELITIES       | List of math fidelities                                                                       | test_query                            |
| KWARGS                | List of kwargs dictionaries.                                                                  | test_query                            |
| FAILING_REASONS       | List of failing reasons                                                                       | test_query                            |
| SKIP_REASONS          | List of skip reasons                                                                          | test_query                            |
| RANGE                 | Limit number of results                                                                       | test_query                            |
| RANDOM_SEED           | Seed for random number generator                                                              | test_query                            |
| SAMPLE                | Percentage of results to sample                                                               | test_query                            |
| TEST_ID               | Id of a single test to run containing all test parameters                                     | test_query                            |
| ID_FILES              | Paths to files containing test ids instead of tests from test plan                            | test_query                            |
| ID_FILES_IGNORE       | Paths to files containing test ids to be ignored                                              | test_query                            |

Test configuration parameters

| Parameter                 | Description                                                                                   | Supported by commands                 |
| ------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------- |
| SKIP_FORGE_VERIFICATION   | Skip Forge model verification including model compiling and inference                         | all                                   |

To check supported values and options for each query parameter please run command `print_query_docs`.


Usage examples

Run all tests

```sh
with-params pytest
```

Run all tests for few operators

```sh
export OPERATORS=add,div
with-params pytest
```

Run subset of tests based on query criteria

```sh
export OPERATORS=div
export FILTERS=HAS_DATA_FORMAT,QUICK
export INPUT_SOURCES=FROM_HOST,FROM_DRAM_QUEUE
export DEV_DATA_FORMATS=Float16_b,Int8
export MATH_FIDELITIES=HiFi4,HiFi3
export KWARGS="[{'rounding_mode': 'trunc'},{'rounding_mode': 'floor'}]"
with-params pytest
```

Print representative tests ids of all operators with examples for kwargs values

```sh
FILTERS=UNIQUE_KWARGS with-params pytest --collect-only
```

Print representative tests ids of few operators

```sh
OPERATORS=add,div FILTERS=UNIQUE_KWARGS with-params pytest --collect-only
```

Each test can be uniquely identified via a test id. Format of test id is `{operator}-{input_source}-{kwargs}-{input_shape}[-{number_of_operands)-]{dev_data_format}-{math_fidelity}`.

Kwarg is a mandatory or optional attribute of an operator. See framework (PyTorch, Forge, ...) operator documentation for each operator or use filter `UNIQUE_KWARGS` to find examples.

Run single test based on a test id. Test id may be from a test plan or constructed custom by specifying custom values for kwargs and input_shapes.

```sh
TEST_ID='ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4' with-params pytest
```
