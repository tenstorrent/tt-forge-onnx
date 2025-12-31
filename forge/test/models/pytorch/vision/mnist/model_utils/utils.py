# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger

from test.models.pytorch.vision.mnist.model_utils.model import MnistModel


def load_model():
    model = MnistModel()
    model.eval()
    return model


def load_input():
    """
    Load MNIST test input data.

    If the dataset files are corrupted, this function will attempt to clean
    and re-download them automatically.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # Try to load the dataset, with error handling for corrupted files
    try:
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    except (RuntimeError, ValueError) as e:
        # Check if error is related to corrupted data files
        if "shape" in str(e).lower() or "invalid" in str(e).lower() or "size" in str(e).lower():
            logger.warning(f"MNIST dataset appears to be corrupted: {e}\n" f"Attempting to clean and re-download...")

            # Remove corrupted MNIST data directory
            mnist_data_dir = "./data/MNIST"
            if os.path.exists(mnist_data_dir):
                try:
                    shutil.rmtree(mnist_data_dir)
                    logger.info(f"Removed corrupted MNIST data directory: {mnist_data_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to remove corrupted data directory: {cleanup_error}")
                    raise RuntimeError(
                        f"MNIST dataset is corrupted and could not be cleaned. "
                        f"Please manually delete '{mnist_data_dir}' and re-run the test."
                    ) from e

            # Retry loading with download=True to re-download clean data
            test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        else:
            # Re-raise if it's a different error
            raise

    dataloader = DataLoader(test_dataset, batch_size=1)
    test_input, _ = next(iter(dataloader))
    return [test_input]
