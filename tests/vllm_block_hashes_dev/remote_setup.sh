#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Prepares the remote test host (10.239.23.121, two L20) for run_validation.sh: its own
# vllm-iaxl-dev image is vLLM 0.23, so this pulls vllm/vllm-openai:v0.29.0 and builds the dev
# image from docker/Dockerfile.dev the way start.sh does (copying the local image is not an
# option: docker load needs about twice its 25 GB on a root disk with 54 GB free). Then it starts
# the container with this repo, /media/model-space and _data mounted, and builds iaxl in it
# (pip install -e .). Run it in tmux on that host: the container keeps a shell open.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_IMAGE=vllm/vllm-openai:v0.29.0
IMAGE=vllm-iaxl-dev-qiuyu-vllm-block-hashes-dev:1.5.1
CONTAINER_NAME=iaxl.vllm.qiuyu-vllm-block-hashes-dev
cd "$REPO_DIR"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker pull "$BASE_IMAGE"
    docker build -f docker/Dockerfile.dev -t "$IMAGE" . \
        --build-arg "BASE=$BASE_IMAGE" \
        --build-arg DEVICE=cuda \
        --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy
fi
df -h /

docker run \
    --runtime nvidia --gpus all \
    -e IAXL_VERSION=1.5.1 -e DEVICE=cuda \
    -e http_proxy -e https_proxy -e no_proxy \
    -e HF_HOME=/_data/hf_home \
    -v "$REPO_DIR/_data:/_data" \
    -v /media/model-space:/media/model-space:ro \
    -v "$REPO_DIR:$REPO_DIR" -w "$REPO_DIR" \
    --privileged \
    --pid host \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" > /dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2> /dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        exec bash'
