# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import time
import pytest
import psutil
import shutil
from loguru import logger
from datetime import datetime
from forge.forge_property_utils import (
    ForgePropertyHandler,
    forge_property_handler_var,
    get_device_arch,
    get_device_count,
    get_device_type,
)
from forge._C.verif import malloc_trim


def pytest_addoption(parser):
    parser.addoption(
        "--log-memory-usage",
        action="store_true",
        default=False,
        help="log per-test memory usage into pytest-memory-usage.csv",
    )
    parser.addoption(
        "--tests_to_filter", nargs="+", type=str, help="List of test patterns to include (file paths or full test IDs)"
    )


@pytest.fixture(scope="function")
def forge_tmp_path(tmp_path):
    """
    Yield a temporary directory path and remove it immediately after the test execution complete.

    This fixture wraps pytest's built-in tmp_path fixture to ensure that
    the temporary directory is cleaned up as soon as the test finishes,
    regardless of the test outcome.
    """
    yield tmp_path
    # After the test, delete the entire temporary directory and its contents
    shutil.rmtree(str(tmp_path), ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("end_timestamp", end_timestamp)


@pytest.fixture(scope="function", autouse=True)
def forge_property_recorder(request, record_property):
    """
    Pytest fixture to initialize and manage a ForgePropertyHandler instance for recording test properties.
    ForgePropertyHandler is created as a context variable and functions from forge_property_utils.py should be used
    for storing various properties during test execution.

    After the test runs, the fixture extracts any refined error message and failure category attached to the test item
    (if available) and records these details. Finally, the properties are stored using the provided record_property
    function regardless of the test outcome.
    """

    # Create a handler that uses the property store; the handler is responsible for recording and managing property details.
    forge_property_handler = ForgePropertyHandler()
    token = forge_property_handler_var.set(forge_property_handler)

    # Auto-detect and record the hardware architecture label (e.g. n150, n300, p150).
    arch = get_device_arch()
    device_count = get_device_count()
    device_type = get_device_type(arch, device_count)
    if device_type and device_type != "unknown":
        forge_property_handler.record_arch(device_type)
    elif arch:
        forge_property_handler.record_arch(arch)

    yield

    forge_property_handler.record_error(request)
    forge_property_handler.record_all_properties(record_property)
    forge_property_handler_var.reset(token)


@pytest.fixture(autouse=True)
def memory_usage_tracker(request):
    """
    A pytest fixture that tracks memory usage during the execution of a test.

    This fixture automatically tracks the memory usage of the process running the tests.
    It starts tracking before the test runs, continues tracking in a background thread during the test,
    and stops tracking after the test completes. It logs the memory usage statistics including the
    minimum, maximum, average, and total memory usage by the test.

    The memory usage is measured in megabytes (MB).

    Note:
        - This fixture is automatically used for all tests due to the `autouse=True` parameter.
        - The interval for memory readings can be adjusted by changing the sleep duration in the `track_memory` function.
        - Min, max, and avg memory usage are calculated based on the recorded memory readings from system memory.
    """
    process = psutil.Process()

    # Initialize memory tracking variables
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = start_mem
    max_mem = start_mem
    total_mem = start_mem
    count = 1

    # Start a background thread or loop to collect memory usage over time
    tracking = True

    def track_memory():
        nonlocal min_mem, max_mem, total_mem, count
        while tracking:
            current_mem = process.memory_info().rss / (1024 * 1024)
            min_mem = min(min_mem, current_mem)
            max_mem = max(max_mem, current_mem)
            total_mem += current_mem
            count += 1
            time.sleep(0.1)  # Adjust the interval as needed

    # Start tracking in a background thread
    import threading

    tracker_thread = threading.Thread(target=track_memory)
    tracker_thread.start()

    # Run the test
    yield

    # Stop tracking and wait for the thread to finish
    tracking = False
    tracker_thread.join()

    # Calculate end memory and memory usage stats
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = min(min_mem, end_mem)
    max_mem = max(max_mem, end_mem)
    total_mem += end_mem
    count += 1
    avg_mem = total_mem / count

    by_test = max_mem - start_mem

    # Log memory usage statistics
    logger.info(f"Test memory usage:")
    logger.info(f"    By test: {by_test:.2f} MB")
    logger.info(f"    Minimum: {min_mem:.2f} MB")
    logger.info(f"    Maximum: {max_mem:.2f} MB")
    logger.info(f"    Average: {avg_mem:.2f} MB")

    gc.collect()  # Force garbage collection
    after_gc = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory usage after garbage collection: {after_gc:.2f} MB")

    malloc_trim()
    after_trim = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory usage after malloc_trim: {after_trim:.2f} MB")

    should_log = request.config.getoption("--log-memory-usage")
    if not should_log:
        return

    # Get the current test name
    test_name = request.node.name

    # Store memory usage stats into a CSV file
    file_name = "pytest-memory-usage.csv"
    with open(file_name, "a") as f:
        if f.tell() == 0:
            # Write header if file is empty
            f.write("test_name,start_mem,end_mem,min_memory,max_memory,by_test (approx), after_gc, after_trim\n")
        # NOTE: escape test_name in double quotes because some tests have commas in their parameter list...
        f.write(
            f'"{test_name}",{start_mem:.2f},{end_mem:.2f},{min_mem:.2f},{max_mem:.2f},{by_test:2f},{after_gc:2f},{after_trim:2f}\n'
        )
