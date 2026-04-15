#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# Install forge-onnx specific build dependencies
pkg_install \
    software-properties-common \
    build-essential \
    git \
    git-lfs \
    libhwloc-dev \
    pandoc \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    linux-tools-generic \
    ninja-build \
    wget \
    ccache \
    doxygen \
    libgtest-dev \
    libgmock-dev \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev \
    jq \
    curl \
    gh \
    expect \
    lcov \
    libgl1 \
    libglx-mesa0 \
    unzip \
    xxd
