#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 1 ]; then
    echo "Error: Exactly 1 argument is required."
    echo "Usage: $0 <docker-commit-tag>"
    exit 1
fi

PROJECT="tt-forge-onnx"
COMMIT_TAG=$1

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
