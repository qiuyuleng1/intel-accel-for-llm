#!/bin/bash -e
# usage: start-1026-container.sh   (dev helper; run in tmux, the shell is interactive)
# Start the 1026 dev container the way 1026/start.sh does (same image, name and run
# arguments from 1026/setvars.sh, `pip install -e .` on entry), plus a mount of this
# 1008 repo, where the test helpers of the 1026 PR branch live, and with MODEL set to
# Qwen3-32B so that setvars.sh mounts the model directory. Qwen3.5-27B (hybrid) is
# mounted read-only as well.
REPO=/home/pese/qiuyu/intel-accel-for-llm-1026
DEV=/home/pese/qiuyu/intel-accel-for-llm-1008
HYBRID_MODEL=/media/model-space/Qwen3.5-27B
cd "$REPO"
export MODEL=/media/model-space/Qwen3-32B
source setvars.sh
docker image inspect "$IAXL_DEV_DOCKER_IMAGE" > /dev/null

docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    -v "$DEV:$DEV" \
    -v "$HYBRID_MODEL:$HYBRID_MODEL:ro" \
    --rm \
    --privileged \
    --pid host \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        exec bash'
