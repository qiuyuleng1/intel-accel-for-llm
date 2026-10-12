#!/bin/bash -e
# Build the v0.29 dev image and start a new container (dev helper). The v0.23
# image (vllm-iaxl-dev-qiuyu) and container (iaxl.vllm.qiuyu) are left untouched.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
export IAXL_DEV_DOCKER_IMAGE=vllm-iaxl-dev-qiuyu-v29
source setvars.sh > /dev/null
CONTAINER_NAME=iaxl.vllm.qiuyu-v29
echo "base=$IAXL_BASE_DOCKER_IMAGE dev=$IAXL_DEV_DOCKER_IMAGE container=$CONTAINER_NAME"

docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
    --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy

# Same as start.sh, but detached (-dit) instead of interactive (-it).
docker run \
    "${DOCKER_RUN_ARGS[@]}" \
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
