"""
Pytest configuration and fixtures for ONNX transpiler tests.
"""
import os
from loguru import logger
import pytest
import numpy as np
import torch


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Setup test environment for transpiler tests.
    This fixture runs once per test session. 
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set number of threads for consistent behavior
    if "FORGE_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["FORGE_NUM_THREADS"])
        torch.set_num_threads(num_threads)
        os.environ["TVM_NUM_THREADS"] = str(num_threads)
    
    logger.info("Test environment setup complete")
    yield
    logger.info("Test environment teardown complete")


@pytest.fixture(autouse=True)
def reset_seeds():
    """
    Reset random seeds before each test for reproducibility.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def default_rtol():
    """
    Default relative tolerance for numerical comparisons.
    """
    return 1e-5


@pytest.fixture
def default_atol():
    """
    Default absolute tolerance for numerical comparisons.
    """
    return 1e-6


@pytest.fixture
def debug_mode():
    """
    Fixture to control debug mode for transpiler.
    Can be overridden by environment variable FORGE_TRANSPILER_DEBUG.
    """
    debug = os.environ.get("FORGE_TRANSPILER_DEBUG", "false").lower() == "true"
    return debug


@pytest.fixture
def validate_model():
    """
    Fixture to control model validation for transpiler.
    Can be overridden by environment variable FORGE_TRANSPILER_VALIDATE.
    """
    validate = os.environ.get("FORGE_TRANSPILER_VALIDATE", "true").lower() == "true"
    return validate


def pytest_addoption(parser):
    """
    Add custom command-line options for transpiler tests.
    """
    parser.addoption(
        "--transpiler-debug",
        action="store_true",
        default=False,
        help="Enable debug mode for transpiler tests"
    )
    parser.addoption(
        "--transpiler-no-validate",
        action="store_true",
        default=False,
        help="Disable model validation in transpiler tests"
    )
    parser.addoption(
        "--transpiler-verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging for transpiler tests"
    )


def pytest_configure(config):
    """
    Configure pytest with custom options.
    """
    # Set environment variables based on command-line options
    if config.getoption("--transpiler-debug"):
        os.environ["FORGE_TRANSPILER_DEBUG"] = "true"
    
    if config.getoption("--transpiler-no-validate"):
        os.environ["FORGE_TRANSPILER_VALIDATE"] = "false"
    
    if config.getoption("--transpiler-verbose"):
        logger.remove()  # Remove default handler
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")  # Set debug level

