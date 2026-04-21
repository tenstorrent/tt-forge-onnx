#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# todo(vvukoman): No need for <project> input, it is always tt-forge-onnx
# todo(vvukoman): Rename all forge_fe references to forge_onnx
if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments are required."
    echo "Usage: $0 <project> <docker-commit-tag>"
    exit 1
fi

PROJECT=$1
COMMIT_TAG=$2

IMAGE_NAME=ghcr.io/tenstorrent/$PROJECT-slim

echo "Building image $IMAGE_NAME:$COMMIT_TAG"
docker build \
  --progress=plain \
  -t $IMAGE_NAME:$COMMIT_TAG \
  -f .github/Dockerfile.release-ubuntu .

echo "Pushing image $IMAGE_NAME:$COMMIT_TAG"
docker push $IMAGE_NAME:$COMMIT_TAG

echo "Image built and pushed successfully"
echo $IMAGE_NAME
echo $IMAGE_NAME:$COMMIT_TAG
