#!/bin/bash -e
# Recreate the v0.29 dev container with both perf models mounted (dev helper).
# The image vllm-iaxl-dev-qiuyu-v29 must already exist (build-v29-container.sh).
# Removes the existing iaxl.vllm.qiuyu-v29 container first.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export IAXL_DEV_DOCKER_IMAGE=vllm-iaxl-dev-qiuyu-v29
# setvars.sh mounts the directory of $MODEL read-only; the second model is added
# below.
export MODEL=/home/pese/model-space/Qwen2.5-72B-Instruct
EXTRA_MODEL=/home/pese/model-space/Qwen3-32B
source setvars.sh > /dev/null
CONTAINER_NAME=iaxl.vllm.qiuyu-v29

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    -v "$EXTRA_MODEL:$EXTRA_MODEL:ro" \
    --privileged \
    --pid host \
    --net host \
    -dit \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation > /tmp/pip_install.log 2>&1
        echo PIP_EXIT=$? >> /tmp/pip_install.log
        exec bash'
echo CONTAINER_STARTED
