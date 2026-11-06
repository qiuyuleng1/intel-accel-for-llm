#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copies the local dev image to the remote test host through a temporary registry, because the
# remote pulls from Docker Hub through its proxy at under 1 MB/s while this host reaches it at
# ~26 MB/s, and `docker load` there would need about twice the 25 GB image on a root disk with
# ~46 GB free. Steps: start registry:2 on 127.0.0.1:5555 here -> push the image to it -> SSH
# reverse tunnel so that the remote's 127.0.0.1:5000 reaches it (Docker allows plain HTTP for
# localhost registries) -> stop the slow pull in the remote tmux session vbh-ctr -> pull and tag
# on the remote -> rerun remote_setup.sh there (the image now exists, so it only starts the
# container and builds iaxl) -> close the tunnel, remove the registry. Run it here, in tmux.
set -euo pipefail

REMOTE=ubuntu@10.239.23.121
REMOTE_REPO_DIR=/media/workspace/qiuyu/intel-accel-for-llm-1026-vbh
IMAGE=vllm-iaxl-dev-qiuyu-vllm-block-hashes-dev:1.5.1
REGISTRY_CONTAINER=qiuyu-vbh-registry
LOCAL_PORT=5555
REMOTE_PORT=5000
TUNNEL_PID=""

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

cleanup() {
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2> /dev/null || true
    fi
    docker rm -f "$REGISTRY_CONTAINER" > /dev/null 2>&1 || true
    docker volume rm "$REGISTRY_CONTAINER" > /dev/null 2>&1 || true
    docker rmi "localhost:$LOCAL_PORT/$IMAGE" > /dev/null 2>&1 || true
}
trap cleanup EXIT

log "starting the registry on 127.0.0.1:$LOCAL_PORT"
docker pull registry:2
docker run -d --name "$REGISTRY_CONTAINER" -p "127.0.0.1:$LOCAL_PORT:5000" \
    -v "$REGISTRY_CONTAINER:/var/lib/registry" registry:2

log "pushing $IMAGE"
docker tag "$IMAGE" "localhost:$LOCAL_PORT/$IMAGE"
docker push "localhost:$LOCAL_PORT/$IMAGE"

log "opening the tunnel: remote 127.0.0.1:$REMOTE_PORT -> local 127.0.0.1:$LOCAL_PORT"
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -R "127.0.0.1:$REMOTE_PORT:127.0.0.1:$LOCAL_PORT" "$REMOTE" &
TUNNEL_PID=$!
sleep 5
kill -0 "$TUNNEL_PID"

log "stopping the Docker Hub pull in the remote tmux session vbh-ctr"
ssh -o BatchMode=yes "$REMOTE" "tmux send-keys -t vbh-ctr C-c"
sleep 10

log "pulling on the remote"
ssh -o BatchMode=yes "$REMOTE" "docker pull localhost:$REMOTE_PORT/$IMAGE \
    && docker tag localhost:$REMOTE_PORT/$IMAGE $IMAGE \
    && docker rmi localhost:$REMOTE_PORT/$IMAGE && df -h / | tail -1"

log "rerunning remote_setup.sh in vbh-ctr (starts the container, builds iaxl)"
ssh -o BatchMode=yes "$REMOTE" \
    "tmux send-keys -t vbh-ctr 'cd $REMOTE_REPO_DIR && bash tests/vllm_block_hashes_dev/remote_setup.sh' Enter"
log "DONE"
