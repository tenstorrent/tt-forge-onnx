#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Enable BuildKit for parallel multi-stage builds, better layer caching,
# and support for advanced Dockerfile features (e.g. --mount=type=cache).
export DOCKER_BUILDKIT=1

REPO=tenstorrent/tt-forge-onnx
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-forge-onnx-base-ubuntu-24-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-forge-onnx-ci-ubuntu-24-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-onnx-base-ird-ubuntu-24-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-onnx-ird-ubuntu-24-04

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3

    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            ${from_image:+--build-arg FROM_IMAGE=$from_image} \
            -t $image_name:$DOCKER_TAG \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
