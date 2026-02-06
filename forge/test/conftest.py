# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from typing import List, Dict, Tuple
from loguru import logger
import subprocess
import fnmatch
import signal
import threading

import numpy as np
import pytest
import _pytest.skipping
import torch.multiprocessing as mp
import torch
import tensorflow as tf


# This is a workaround to set RTLD_GLOBAL flag to load emulation ZeBu library.
# Essentially symbol names have to be unique in global scope to work with ZeBu,
# hence need to be set as GLOBAL. This is a requirement for ZeBu.
import sys

original_flags = sys.getdlopenflags()
if os.environ.get("FORGE_ENABLE_EMULATION_DEVICE") == "1":
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
# Import code that requires os.RTLD_GLOBAL goes here
# Reset the flags to their original value
if os.environ.get("FORGE_ENABLE_EMULATION_DEVICE") == "1":
    sys.setdlopenflags(original_flags)

import forge
from forge.config import CompilerConfig
from forge.verify.config import TestKind
from forge.tvm_to_python import ExitTest

import test.utils
from test.exception_utils import extract_refined_error_message, extract_failure_category
import json

collect_ignore = ["legacy_tests"]

watchdog_timer_default = 1800  # default expiration in seconds (test not in .test_durations)
watchdog_timer_minimum = 300  # minimum expiration in seconds
watchdog_abort_timer = None
watchdog_test_durations = None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    def send_abort_signal():
        # commenting this out for now, as it is currently not working with pytest
        # Suspend pytest capturing (if active) so that we can actually print out the message.
        # capmanager = item.session.config.pluginmanager.get_plugin("capturemanager")
        # if capmanager and capmanager.is_capturing():
        #     capmanager.suspend()

        # print("WATCHDOG timeout reached! Killing test process.")
        os.kill(os.getpid(), signal.SIGABRT)

    def reset_abort_timer(timeout=watchdog_timer_default):
        global watchdog_abort_timer
        if watchdog_abort_timer:
            watchdog_abort_timer.cancel()
        watchdog_abort_timer = threading.Timer(timeout, send_abort_signal)  # 10 minute timeout
        watchdog_abort_timer.daemon = True
        watchdog_abort_timer.start()

    def check_test_durations(tst):
        global watchdog_test_durations, watchdog_timer_default, watchdog_timer_minimum
        if watchdog_test_durations is None:
            try:
                with open(".test_durations", "r") as f:
                    watchdog_test_durations = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                watchdog_test_durations = {}

        if tst in watchdog_test_durations:
            duration = watchdog_test_durations[tst] * 2
            if duration < watchdog_timer_minimum:
                duration = watchdog_timer_minimum
        else:
            duration = watchdog_timer_default

        return duration

    reset_abort_timer(check_test_durations(item.nodeid))
    with open(".pytest_current_test_executing", "w") as f:
        f.write(item.nodeid)
        f.flush()


def pytest_sessionstart(session):
    # See: https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    mp.set_start_method("spawn")
    num_threads = 8
    if "FORGE_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["FORGE_NUM_THREADS"])
    torch.set_num_threads(num_threads)
    mp.set_sharing_strategy("file_system")
    os.environ["TVM_NUM_THREADS"] = f"{num_threads}"
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    torch._dynamo.reset()
    # If specified by env variable, print the environment variables
    # It can be useful in CI jobs to get the state of the enviroment variables before test session starts
    print_env_variables = bool(int(os.environ.get("PYTEST_PRINT_ENV_VARIABLES", "0")))
    if print_env_variables:
        forge_specific_vars = {}
        tt_backend_specific_vars = {}
        print(f"####### Environment variables - Count: {len(os.environ)} #######")
        for key, value in os.environ.items():
            print(f"{key}={value}")
            if key.startswith("FORGE_") or key.startswith("GOLDEN_"):
                forge_specific_vars[key] = value
            elif key.startswith("TT_BACKEND_"):
                tt_backend_specific_vars[key] = value

        print(f"####### FORGE specific enviroment variables - Count: {len(forge_specific_vars)} #######")
        for key, value in forge_specific_vars.items():
            print(f"{key}={value}")

        print(f"####### TT_BACKEND specific enviroment variables - Count: {len(tt_backend_specific_vars)} #######")
        for key, value in tt_backend_specific_vars.items():
            print(f"{key}={value}")


@pytest.fixture(autouse=True)
def reset_seeds_fixture():
    test.utils.reset_seeds()


@pytest.fixture(autouse=True)
def reset_device():
    if "FORGE_RESET_DEV_BEFORE_TEST" in os.environ:
        # Reset device between tests
        # For this to work, pytest must be called with --forked
        subprocess.check_call(["device/bin/silicon/reset.sh"], cwd=os.environ["FORGE_HOME"])


def run_command(cmd: List[str]) -> str:
    """
    Executes a shell command using subprocess.run and returns the command's output.

    Args:
        cmd (List[str]): The command to execute as a list of arguments.

    Returns:
        str: The standard output from the command execution.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Command '{' '.join(cmd)}' executed successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(cmd)
        logger.error(f"Error executing '{cmd_str}': {e}")
        return ""


def parse_pip_freeze(freeze_output: str) -> Dict[str, str]:
    """
    Parses the output of 'pip freeze' and returns a dictionary of package names and versions.

    Args:
        freeze_output (str): The output string from 'pip freeze'.

    Returns:
        Dict[str, str]: A dictionary where keys are package names and values are their versions.
    """
    packages = {}
    for line in freeze_output.splitlines():
        if "==" in line:
            pkg, version = line.split("==", maxsplit=1)
            packages[pkg.strip()] = version.strip()
    return packages


@pytest.fixture(scope="function")
def restore_package_versions():
    """
    A pytest fixture that ensures package versions remain consistent during tests.

    This fixture captures the installed packages before the test runs, and after the test,
    compares the versions. If any package version has changed, it logs the difference and attempts
    to revert the package back to its original version.
    """
    # Capture the state of installed packages before test execution.
    logger.info("Capturing the initial state of installed packages using 'pip freeze'.")
    before_freeze = run_command(["pip", "freeze"])
    before_packages = parse_pip_freeze(before_freeze)

    yield

    # Capture the state after test execution.
    logger.info("Capturing the final state of installed packages using 'pip freeze'.")
    after_freeze = run_command(["pip", "freeze"])
    after_packages = parse_pip_freeze(after_freeze)

    # Determine which packages have changed versions.
    diff_packages: Dict[str, Tuple[str, str]] = {}
    for pkg, orig_version in before_packages.items():
        if pkg in after_packages:
            new_version = after_packages[pkg]
            if new_version != orig_version:
                diff_packages[pkg] = (orig_version, new_version)

    # If differences are detected, log and revert the changed packages.
    if diff_packages:
        logger.info("Detected changes in package versions during the test:")
        for pkg, (orig_version, new_version) in diff_packages.items():
            logger.info(f"Package '{pkg}': current version {new_version}; reverting to {orig_version}")
            cmd_output = run_command(["pip", "install", f"{pkg}=={orig_version}"])
            logger.info(cmd_output)
    else:
        logger.info("No package version changes detected after the test.")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    This hook is intended to be executed during the 'call' phase.
    It performs the following actions:
      - If the test is either a genuine failure or an expected failure marked as xfail,
        it extracts the test's error message.
      - It uses helper functions to refine the error message and to determine an appropriate failure category.
      - These details are attached to the test item as attributes, which can later be accessed by forge_property_recorder fixture for additional property recording.
    """
    outcome = yield
    report = outcome.get_result()

    if call.excinfo and call.excinfo.errisinstance(ExitTest):

        reason = getattr(call.excinfo.value, "reason", None)

        report.outcome = "passed"
        report.longrepr = None
        call.excinfo = None

        if reason:
            report.user_properties.append(("exit_reason", reason))

    # Only process reports that are generated during the execution phase of the test ("call")
    if not report or report.when != "call":
        return

    # Determine if the test is expected to fail (xfail) or actually failed.
    xfail = hasattr(report, "wasxfail")
    is_xfailed = report.skipped and xfail
    is_failed = report.failed and not xfail
    if not (is_xfailed or is_failed):
        return

    # Extract the error message from the test report; exit if no message is found.
    error_message = getattr(report, "longreprtext", None)
    if not error_message:
        return

    # Refine the error message using a helper function to remove unnecessary details and extract relevant info.
    refined_error_message = extract_refined_error_message(error_message)
    if refined_error_message is None:
        return

    # Attach the refined error message to the test item so that other hooks or fixtures can access it.
    setattr(item, "refined_error_message", refined_error_message)

    # Extract a failure category from the refined error message and attach it to the test item. if one is determined.
    failure_category = extract_failure_category(refined_error_message)
    if failure_category is not None:
        setattr(item, "failure_category", failure_category)


def pytest_addoption(parser):
    parser.addoption(
        "--silicon-only", action="store_true", default=False, help="run silicon tests only, skip golden/model"
    )
    parser.addoption("--no-silicon", action="store_true", default=False, help="skip silicon tests")
    parser.addoption(
        "--compile-only", action="store_true", default=False, help="only compiles the model and generate TTI"
    )
    parser.addoption(
        "--run-only", action="store_true", default=False, help="load the generated TTI and only runs the model"
    )
    parser.addoption(
        "--tti-path",
        default=None,
        type=str,
        help="Valid only if either --compile-only or --run-only is specified. Save/load TTI from the path",
    )
    parser.addoption(
        "--device-config", default=None, type=str, help="Runtime yaml is automatically configured based on the value"
    )
    parser.addoption(
        "--devtype",
        default=None,
        type=str,
        choices=("golden", "silicon"),
        help="Valid only if --compile-only is specified. Set the backend device type between Golden or Silicon",
    )
    parser.addoption(
        "--no-skips", action="store_true", default=False, help="ignore pytest.skip() calls, and continue on with test"
    )


"""
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--silicon", action="store_true", default=False, help="run silicon tests"
    )
    parser.addoption(
        "--versim", action="store_true", default=False, help="run versim tests"
    )
    parser.addoption(
        "--arch", action="store", default="wormhole_b0", help="run tests on different arch"
    )
    parser.addoption(
        "--microbatch-size", action="store", default=8,
    )
    parser.addoption(
        "--num-microbatches", action="store", default=2,
    )
    parser.addoption(
        "--batch-size", action="store", default=32,
    )
    parser.addoption(
        "--num-batches", action="store", default=4,
    )
    parser.addoption(
        "--num-chips", action="store", default=1,
    )

# Use this to stop slow tests from running by default
@pytest.fixture(name="runslow", scope="session", autouse=True)
def runslow(request):
   return request.config.getoption("--runslow")
"""


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):

    if "--no-skips" not in args:
        return

    def no_skip(*args, **kwargs):
        return

    pytest.skip = no_skip
    _pytest.skipping.skip = no_skip  # can't run skipped tests with decorator @pytest.mark.skip without this


# DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE = {
#     "gs_e150": BackendDevice.Grayskull,
#     "gs_e300": BackendDevice.Grayskull,
#     "wh_n150": BackendDevice.Wormhole_B0,
#     "wh_n300": BackendDevice.Wormhole_B0,
#     "galaxy": BackendDevice.Wormhole_B0,
# }

# @dataclass
# class TestDevice:
#     devtype: BackendType
#     arch: BackendDevice
#     devmode: DeviceMode
#     tti_path: str = None

#     @classmethod
#     def from_str(cls, name: str, devmode: DeviceMode, tti_path: str = None, device_config=None) -> "TestDevice":
#         if name == "Golden":
#             if device_config and DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE.get(device_config, None):
#                 return TestDevice(devtype=BackendType.Golden, arch=DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE[device_config], devmode=devmode, tti_path=tti_path)
#             elif "GOLDEN_WORMHOLE_B0" in os.environ:
#                 return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
#             elif "FORGE_GOLDEN_BLACKHOLE" in os.environ:
#                 return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
#             return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Model":
#             return TestDevice(devtype=BackendType.Model, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Versim":
#             # Set default versim device arch to Grayskull
#             versim_backend_device = BackendDevice.Grayskull
#             # If FORGE_VERSIM_DEVICE_ARCH is set, use that arch for Versim device
#             versim_arch_name = os.environ.get("FORGE_VERSIM_DEVICE_ARCH", None)
#             if versim_arch_name != None:
#                 versim_backend_device = BackendDevice.from_string(versim_arch_name)
#             return TestDevice(devtype=BackendType.Versim, arch=versim_backend_device, devmode=devmode, tti_path=tti_path)
#         if name == "Emulation":
#             # Set default emulation device arch to Grayskull
#             emulation_backend_device = BackendDevice.Grayskull
#             # If FORGE_EMULATION_DEVICE_ARCH is set, use that arch for Emulation device
#             emulation_arch_name = os.environ.get("FORGE_EMULATION_DEVICE_ARCH", None)
#             if emulation_arch_name != None:
#                 emulation_backend_device = BackendDevice.from_string(emulation_arch_name)
#             return TestDevice(devtype=BackendType.Emulation, arch=emulation_backend_device, devmode=devmode, tti_path=tti_path)
#         if name == "Grayskull":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Wormhole_B0":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
#         if name == "Blackhole":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
#         raise RuntimeError("Unknown test device: " + name)

#     def is_available(self, device_list: List[BackendDevice], silicon_only: bool, no_silicon: bool, devtype: Optional[BackendType], devmode: DeviceMode) -> bool:
#         """
#         Return true if this kind of device is available on the current host. Expect a list of devices from
#         `detect_available_devices`.
#         """
#         if devtype is not None and self.devtype != devtype:
#             return False

#         if self.devtype == BackendType.Golden:
#             return not silicon_only

#         if self.devtype == BackendType.Model:
#             return bool(int(os.environ.get("FORGE_ENABLE_MODEL_DEVICE", "0")))

#         if self.devtype == BackendType.Versim:
#             return bool(int(os.environ.get("FORGE_ENABLE_VERSIM_DEVICE", "0")))

#         if self.devtype == BackendType.Emulation:
#             return bool(int(os.environ.get("FORGE_ENABLE_EMULATION_DEVICE", "0")))

#         if self.devtype == BackendType.Silicon:
#             compiled_arch_name = os.environ.get("BACKEND_ARCH_NAME", None) or os.environ.get("ARCH_NAME", None)
#             if compiled_arch_name == "wormhole_b0":
#                 compiled_arch = BackendDevice.Wormhole_B0
#             elif compiled_arch_name == "blackhole":
#                 compiled_arch = BackendDevice.Blackhole
#             else:
#                 compiled_arch = BackendDevice.Grayskull

#             is_offline_silicon_compile = devmode == DeviceMode.CompileOnly and self.arch == compiled_arch
#             return (self.arch in device_list and not no_silicon) or is_offline_silicon_compile

#         return False

#     def is_silicon(self):
#         return self.devtype == BackendType.Silicon

#     def is_grayskull(self):
#         return self.arch == BackendDevice.Grayskull

#     def is_wormhole_b0(self):
#         return self.arch == BackendDevice.Wormhole_B0

#     def is_blackhole(self):
#         return self.arch == BackendDevice.Blackhole

device_cfg_global = None


def pytest_generate_tests(metafunc):
    global device_cfg_global

    if "test_kind" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_kind",
            (TestKind.INFERENCE, TestKind.TRAINING, TestKind.TRAINING_RECOMPUTE),
            ids=["inference", "training", "training_with_recompute"],
        )

    if "training" in metafunc.fixturenames:
        metafunc.parametrize("training", (False, True), ids=["inference", "training"])

    if "test_device" in metafunc.fixturenames:
        # Temporary work arround to provide dummy test_device
        # TODO remove workarround https://github.com/tenstorrent/tt-forge-onnx/issues/342
        metafunc.parametrize("test_device", (None,), ids=["no_device"])

    if "_test_device_not_implemented" in metafunc.fixturenames:
        # if "test_device" in metafunc.fixturenames:
        names = ["Golden", "Model", "Versim", "Emulation", "Wormhole_B0", "Blackhole"]

        # Set device-mode for the test
        compile_only = metafunc.config.getoption("--compile-only")
        run_only = metafunc.config.getoption("--run-only")
        devtype = metafunc.config.getoption("--devtype")
        devtype = BackendType.from_string(devtype.capitalize()) if devtype else None

        devmode = DeviceMode.CompileAndRun
        if compile_only:
            devmode = DeviceMode.CompileOnly
            if devtype is None:
                assert False, "Backend device type needs to be specified when running tests with compile-only mode"
        elif run_only:
            devmode = DeviceMode.RunOnly

        # Configure TTI-path only if compile/run-only is set
        tti_path = None
        if compile_only or run_only:
            tti_path = metafunc.config.getoption("--tti-path")

        devices = [(TestDevice.from_str(s, devmode, tti_path, device_cfg_global), s) for s in names]
        silicon_only = metafunc.config.getoption("--silicon-only")
        no_silicon = metafunc.config.getoption("--no-silicon")
        device_list = []
        if not no_silicon:
            device_list = detect_available_devices()
        enabled_devices = [
            (d, name)
            for (d, name) in devices
            if d.is_available(device_list, silicon_only, no_silicon, devtype, devmode)
        ]
        params = [pytest.param(d) for (d, _) in enabled_devices]
        ids = [name for (_, name) in enabled_devices]

        metafunc.parametrize("test_device", params, ids=ids)

    # Configure backend runtime yaml
    device_cfg_global = metafunc.config.getoption("--device-config")


environ_before_test = None


def pytest_runtest_logreport(report):
    if report.when == "setup":
        global environ_before_test
        environ_before_test = os.environ.copy()

        if "FORGE_OVERRIDES_VETO" in os.environ:
            from forge.config import _set_forge_override_veto

            # This functionality represents one way to control general and env based compiler configuration (enable us to
            # add/update/remove existing configs in each test with ease during runtime). In sum, it uses a dict of key-value pairs
            # that control all Forge specific overrides set in test. Have in  mind that this doesn't apply for everything set
            # outside of the test itself (e.g. env vars set before calling the specific pytest).
            #
            # Input to this function is represented as two dicts:
            # - first one is a dict of keys/value pairs that controls general compiler config settings
            # - second one is a dict of keys/value pairs that controls configurations set through environment variables
            #
            # Also, few general notes of how to use this these dicts to control the general and env based compiler configurations:
            # - overriding value with "" will use the value set in test itself
            # - overriding with some specific value will use that config and override it (ignoring the test config)
            # - not including the key and value here, will use default compiler config value (discarding the test config if set there)
            #
            # Description of override levels:
            # - Level 0 - set by compiler;      we want to keep them            (defined during compiler runtime; not in test itself)
            # - Level 1 - set by user in test;  we want to keep them            (defined in test, but easy to use and understandable for end user)
            # - Level 2 - set by dev in test;   we want to remove them          (e.g. enable/disable by default, redefine as more user friendly, etc.)
            # - Level 3 - set by dev in test;   we want to remove them entirely (purely for testing purposes)
            #
            if "FORGE_OVERRIDES_VETO_CUSTOM_SETUP" in os.environ:
                _set_forge_override_veto(
                    {
                        "backend_output_dir": "",
                    },
                    {},
                )
            else:
                _set_forge_override_veto(
                    {
                        "backend_output_dir": "",
                        "backend_runtime_params_path": "",
                        "harvesting_mask": "",
                        "cpu_fallback_ops": "",
                        # Level 1 overrides
                        "balancer_policy": "",
                        "enable_t_streaming": "",
                        "default_df_override": "",
                    },
                    {
                        # Level 2 overrides
                        "FORGE_RIBBON2": "",
                        "FORGE_DISABLE_STREAM_OUTPUT": "",
                        "FORGE_OVERRIDE_DEVICE_YAML": "",  # Mostly used for 1x1 model overrides
                    },
                )

    elif report.when == "teardown":
        environ_before_test_keys = set(environ_before_test.keys())
        environ_after_test_keys = set(os.environ.keys())

        # remove
        added_flags = environ_before_test_keys ^ environ_after_test_keys
        for f in added_flags:
            os.environ.pop(f, None)

        # reset
        for key, default_value in environ_before_test.items():
            if os.environ.get(key, "") != default_value:
                os.environ[key] = default_value


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):

    patterns = config.getoption("--tests_to_filter")
    if patterns:
        selected = []
        deselected = []
        seen_files = set()

        # Precompile pattern types
        file_patterns = []
        test_patterns = []
        for p in patterns:
            if "::" in p:
                test_patterns.append(p)
            else:
                # Normalize file paths
                file_patterns.append(os.path.normpath(p))

        for item in items:
            # Extract file path and full test ID
            file_path = os.path.normpath(str(item.fspath))
            full_test_id = item.nodeid

            # Check for file matches
            file_match = any(
                fnmatch.fnmatch(file_path, pattern)
                or fnmatch.fnmatch(file_path, pattern + ".py")
                or pattern in file_path
                for pattern in file_patterns
            )

            # Check for full test ID matches
            test_match = any(
                fnmatch.fnmatch(full_test_id, pattern) or pattern in full_test_id for pattern in test_patterns
            )

            if file_match or test_match:
                selected.append(item)
                # Track which files had matches
                seen_files.add(file_path)
            else:
                deselected.append(item)

        # Handle partial file patterns (e.g., directory/*.py)
        for pattern in file_patterns:
            if not any(pattern in f for f in seen_files):
                pytest.exit(f"No tests found matching file pattern: {pattern}", returncode=2)

        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    yield

    is_collect_enabled = config.getoption("--collect-only")
    marker = config.getoption("-m")
    if marker and "skip_model_analysis" in marker and is_collect_enabled:  # If a marker is specified
        print("\nAutomatic Model Analysis Collected tests: ")
        test_count = 0
        for item in items:
            test_file_path = item.location[0]
            test_name = item.location[2]
            print(f"{test_file_path}::{test_name}")
            test_count += 1
        print(f"Automatic Model Analysis Collected test count: {test_count}")
        if test_count == 0:  # Warn if no tests match the marker
            print(f"Warning: No tests found with marker '{marker}'.")

    splits = config.getoption("splits")
    group = config.getoption("group")
    if splits is not None and group is not None:
        nodeids = [item.nodeid for item in items]
        if nodeids:
            print("\nCollected tests after splits and group:")
            for n in nodeids:
                print(n)

    for item in items:
        callspec = getattr(item, "callspec", None)
        if not callspec:
            continue

        training_val = callspec.params.get("training_test", None)

        # xfail_if_models_ops_training
        marker = item.get_closest_marker("xfail_if_models_ops_training")
        if marker and training_val is True:
            reason = marker.kwargs.get("reason", "no reason")
            item.add_marker(pytest.mark.xfail(reason=reason))

        # skip_if_models_ops_training
        marker = item.get_closest_marker("skip_if_models_ops_training")
        if marker and training_val is True:
            reason = marker.kwargs.get("reason", "no reason")
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item):
    """
    Wrap the normal test call.  If ExitTest is raised, swallow it
    and return early—pytest will see the test as having passed.
    """
    # Run the actual test function
    try:
        yield
    except ExitTest:
        # Prevent pytest from treating this as an error
        # the exception is swallowed, so the call phase ends cleanly
        pass
