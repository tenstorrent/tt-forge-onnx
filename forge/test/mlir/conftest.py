# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(autouse=True)
def close_tt_device_after_test():
    """Close all TT devices after each test so the next test starts clean."""
    yield
    try:
        from forge._C import runtime as forge_runtime
        sys = forge_runtime.experimental.TTSystem.get_system()
        if sys is not None and sys.is_initialized():
            sys.close_devices()
    except Exception:
        pass
