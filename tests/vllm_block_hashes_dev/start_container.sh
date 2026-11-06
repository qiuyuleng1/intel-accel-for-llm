#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Dev container for run_validation.sh. iaxl.vllm.qiuyu-python-overhead-dev only mounts
# Qwen3-32B, and a running container cannot get another mount, so this one is started from a
# snapshot of it (vLLM 0.29 with this repo installed editable, so no rebuild) with all of
# /media/model-space mounted. Same docker run options as start.sh, without --rm so that the
# container survives its terminal. Run it in tmux: it keeps a shell open.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_CONTAINER=iaxl.vllm.qiuyu-python-overhead-dev
IMAGE=vllm-iaxl-dev-qiuyu-vllm-block-hashes-dev:1.5.1
CONTAINER_NAME=iaxl.vllm.qiuyu-vllm-block-hashes-dev

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker commit "$SOURCE_CONTAINER" "$IMAGE"
fi

docker run \
    --runtime nvidia --gpus all \
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
    bash -c 'git config --global --add safe.directory "$PWD"; exec bash'
