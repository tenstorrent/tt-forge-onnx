# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cache cleanup utilities for tt-metal-cache directory.

This module provides functionality to clean up the tt-metal-cache directory
after test execution.
"""

import os
import shutil
from typing import Optional
from loguru import logger


def find_cache_directory() -> Optional[str]:
    """
    Find the tt-metal-cache directory that exists.
    
    Returns:
        Path to the cache directory, or None if none exists
    """
    cache_paths = [
        "/github/home/.cache/tt-metal-cache",  # CI environment
        os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache"),  # Local environment
        "/tmp/tt-metal-cache",  # Fallback
    ]
    
    # Also check environment variable
    env_cache = os.environ.get("TT_METAL_CACHE_DIR")
    if env_cache:
        cache_paths.insert(0, env_cache)
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            return cache_path
    
    return None


def cleanup_cache(cache_path: Optional[str] = None) -> bool:
    """
    Remove the tt-metal-cache directory.
    
    Args:
        cache_path: Optional specific cache path to remove.
                   If None, will find and remove the first existing cache path.
    
    Returns:
        True if cleanup was successful, False otherwise.
    """
    if cache_path is None:
        cache_path = find_cache_directory()
    
    if cache_path is None:
        logger.debug("No tt-metal-cache directory found to clean up")
        return False
    
    try:
        shutil.rmtree(cache_path)
        logger.info(f"[CACHE_CLEANUP] Removed tt-metal cache directory: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"[CACHE_CLEANUP] Failed to remove tt-metal cache directory {cache_path}: {e}")
        return False
