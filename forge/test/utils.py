# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Optional
from loguru import logger
import random
import requests
import time
import os
import shutil
import urllib
from filelock import FileLock
import tarfile
import importlib
import subprocess
import sys
from typing import Optional

import numpy as np
import torch
import paddle
import tensorflow as tf
from forge.module import FrameworkModule


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torch/models", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torchxrayvision/models_data", ignore_errors=True)
            os.mkdir(os.path.expanduser("~") + "/.cache")
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."


def get_cache_dir() -> str:
    """Get models cache directory from env var or use local default."""
    cache_dir = os.environ.get("FORGE_MODELS_CACHE")
    if not cache_dir:
        cache_dir = os.path.join(os.getcwd(), ".forge_models_cache")
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_loader(path: str):
    """Load model with PyTorch."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"PyTorch loading error: {e}")
        return None


def yolov5_loader(path: str, variant: str = "ultralytics/yolov5"):
    try:
        model = download_model(torch.hub.load, variant, "custom", path=path)
        return model
    except Exception as e:
        print(f"YOLOv5 loading error: {e}")
        return None


def fetch_model(
    model_name: str,
    url: str,
    loader: Optional[Callable] = default_loader,
    max_retries: int = 3,
    timeout: int = 30,
    **kwargs: Any,
) -> FrameworkModule:
    """Fetch model from URL, cache it, and load it."""

    model_file = model_name + ".pt"

    model_path = os.path.join(get_cache_dir(), model_file)

    # Download if needed
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lock_path = model_path + ".lock"
        lock = FileLock(lock_path)

        with lock:
            # Check again after acquiring the lock to handle concurrent processes
            if not os.path.exists(model_path):
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"Downloading {model_name}, attempt {attempt}/{max_retries}...")
                        response = requests.get(url, timeout=timeout)
                        response.raise_for_status()

                        # Write to temporary file first
                        temp_path = model_path + ".tmp"
                        with open(temp_path, "wb") as f:
                            f.write(response.content)

                        # Atomic rename after successful download
                        try:
                            os.rename(temp_path, model_path)
                        except OSError as e:
                            print(f"Error during rename: {e}")
                            # Clean up the temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            raise  # Let this be caught by the outer exception handler

                        break  # Successfully downloaded and renamed

                    except (requests.exceptions.RequestException, OSError) as e:
                        print(f"Attempt {attempt} failed: {e}")
                        # Clean up temp file if it exists
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                        if attempt < max_retries:
                            time.sleep(2**attempt)  # Exponential backoff
                        else:
                            raise RuntimeError(f"Failed to download {model_name} after {max_retries} attempts.")

    # Load model
    model = loader(model_path, **kwargs) if loader else None
    return model


def paddlenlp_from_pretrained(
    load_fn,
    variant,
    max_retries=3,
    base_delay_sec=2.0,
    max_delay_sec=60.0,
    jitter=True,
    **kwargs,
):
    """
    Load PaddleNLP model/tokenizer with retry for transient network failures
    (e.g. RemoteDisconnected, connection timeouts).
    """
    retriable = (
        OSError,
        ConnectionError,
        TimeoutError,
    )
    try:
        import urllib3.exceptions

        retriable += (urllib3.exceptions.ProtocolError, urllib3.exceptions.MaxRetryError)
    except ImportError:
        logger.trace("urllib3 is not installed; skipping urllib3-specific retriable exceptions.")
    try:
        import requests.exceptions

        retriable += (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
        )
    except ImportError:
        logger.trace("requests not available; skipping requests-specific retriable exceptions.")

    last_error = None
    load_kwargs = dict(kwargs)

    for attempt in range(max_retries):
        try:
            return load_fn(variant, **load_kwargs)
        except retriable as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(base_delay_sec * (2**attempt), max_delay_sec)
                if jitter:
                    delay *= 0.5 + random.random()
                time.sleep(delay)
            else:
                break

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"paddlenlp_from_pretrained: exhausted {max_retries} retries for '{variant}' "
        "without capturing a retriable error. This is unexpected — the loader neither "
        "succeeded nor raised a recognized retriable exception."
    )


def fetch_paddle_model(url, save_dir):
    model_name = os.path.splitext(os.path.basename(url))[0]
    file_names = ["inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel"]

    # Download the tar file
    response = requests.get(url, stream=True)
    tar_path = os.path.join(save_dir, model_name + ".tar")

    # Check if the model is already downloaded
    model_dir = os.path.join(save_dir, model_name)
    if os.path.exists(model_dir) and all(os.path.exists(os.path.join(model_dir, file)) for file in file_names):
        print(f"Model already downloaded at {model_dir}. Skipping download.")

    else:
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=save_dir)

        os.remove(tar_path)

        # Verify if all required files are present in the model directory
        if not all(os.path.exists(os.path.join(model_dir, file)) for file in file_names):
            raise FileNotFoundError(f"Some required model files are missing in {model_dir}.")

    model_path = os.path.join(model_dir, "inference")
    model = paddle.jit.load(model_path)

    return model


def reset_seeds():
    random.seed(0)
    paddle.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)


def install_yolox_if_missing(
    version: str = "0.3.0",
    package_name: str = "yolox",
) -> bool:
    """Install yolox==<version> without dependencies if not already installed.

    Reason to install yolox=0.3.0 through subprocess :
    requirements of yolox=0.3.0 can be found here https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/requirements.txt
    onnx==1.8.1 and onnxruntime==1.8.0 are required by yolox which are incompatible with our package versions
    Dependencies required by yolox for pytorch implemetation are already present in pybuda and packages related to onnx is not needed
    pip install yolox==0.3.0 --no-deps can be used to install a package without installing its dependencies through terminal
    But in pybuda packages were installed through requirements.txt file not though terminal.
    unfortunately there is no way to include --no-deps in  requirements.txt file.
    for this reason , yolox==0.3.0 is intalled through subprocess.

    Returns True if yolox is present (already or after installation).
    Raises subprocess.CalledProcessError on pip failure or ImportError if import fails after install.
    """
    # 1) Quick check: is package already installed and version matches?
    try:
        installed_ver: Optional[str] = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        installed_ver = None

    if installed_ver is not None:
        if installed_ver == version:
            logger.info(f"{package_name}=={version} already installed. Skipping installation.")
            return True
        else:
            logger.info(
                f"{package_name} is installed (version={installed_ver}) but requested version is {version}. Proceeding to install requested version."
            )

    # 2) Edge-case: module importable but metadata missing
    spec = importlib.util.find_spec(package_name)
    if spec is not None and installed_ver is None:
        logger.info(
            f"{package_name} appears importable but metadata not found. Will attempt to (re)install {package_name}=={version}."
        )

    logger.info(f"Installing {package_name}=={version} (no deps, no build isolation)...")
    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"{package_name}=={version}",
        "--no-deps",
        "--no-build-isolation",
    ]

    # 3) Run pip and capture output for debugging if it fails
    proc = subprocess.run(pip_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.error(f"pip install failed (rc={proc.returncode}). stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        # raise so caller can see failure
        raise subprocess.CalledProcessError(proc.returncode, pip_cmd, output=proc.stdout, stderr=proc.stderr)

    # 4) Verify import after install
    importlib.invalidate_caches()
    try:
        importlib.import_module(package_name)
    except Exception as e:
        logger.error(f"Import of {package_name} failed after installation: {e}")
        raise

    # 5) final version sanity check
    try:
        installed_ver_after = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        installed_ver_after = None

    if installed_ver_after != version:
        logger.warning(f"Installed {package_name} version is {installed_ver_after} (requested {version}).")

    logger.info(f"{package_name}=={installed_ver_after} is installed and importable.")
    return True
