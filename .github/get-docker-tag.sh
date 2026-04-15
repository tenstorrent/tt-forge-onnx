#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash for docker image tag.

# The hash is based on the MLIR docker tag
if [ -f third_party/tt-mlir/.github/get-docker-tag.sh ]; then
    MLIR_DOCKER_TAG=$(cd third_party/tt-mlir && .github/get-docker-tag.sh)
else
    MLIR_DOCKER_TAG="default-tag"
fi

# The hash is based on the environment files (filter out files that are not tracked in git).
ENV_HASH=$(find env -type f -name "*.txt" | sort | sed 's|^\./||' | grep -Fxf <(git ls-files env/) | xargs cat | sha256sum | cut -d ' ' -f 1)

# The hash is based on the Dockerfile(s) and the shared install script.
# Including docker_install.sh ensures any change to it also busts the cached image.
DOCKERFILE_HASH=$( (cat .github/Dockerfile.base .github/Dockerfile.ci .github/Dockerfile.ird .github/docker_install.sh | sha256sum) | cut -d ' ' -f 1)

# Combine the hashes and calculate the final hash
DOCKER_TAG=$( (echo $MLIR_DOCKER_TAG; echo $ENV_HASH; echo $DOCKERFILE_HASH) | sha256sum | cut -d ' ' -f 1)
echo dt-$DOCKER_TAG
