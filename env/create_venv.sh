# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

if [[ -z "$TTFORGE_PYTHON_VERSION" ]]; then
    echo "TTFORGE_PYTHON_VERSION environment variable is not set"
    exit 1
fi

if [[ -z "$TTFORGE_TOOLCHAIN_DIR" ]]; then
    echo "TTFORGE_TOOLCHAIN_DIR environment variable is not set"
    exit 1
fi

if [[ -z "$TTFORGE_VENV_DIR" ]]; then
    echo "TTFORGE_VENV_DIR environment variable is not set"
    exit 1
fi

if [[ -z "$CURRENT_SOURCE_DIR" ]]; then
    echo "CURRENT_SOURCE_DIR environment variable is not set"
    exit 1
fi

$TTFORGE_PYTHON_VERSION -m venv $TTFORGE_VENV_DIR
unset LD_PRELOAD
source $TTFORGE_VENV_DIR/bin/activate
python -m pip install --upgrade pip
pip3 install wheel==0.37.1
pip3 install setuptools==78.1.0
pip3 install -r "$CURRENT_SOURCE_DIR/linux_requirements.txt" -f https://download.pytorch.org/whl/cpu/torch_stable.html
