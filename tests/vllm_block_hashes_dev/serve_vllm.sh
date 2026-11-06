#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Starts or stops the vLLM server of run_validation.sh (same container, port and settings) on the
# host it runs on (the remote test host, or this host with start_container.sh's container), for
# the gsm8k passes of gsm8k_two_passes.sh. GPUS (default 0,1) picks the two GPUs.
#   serve_vllm.sh start <model path> <vllm log>   waits for /health, then prints SERVE_PID=<pid>
#   serve_vllm.sh stop                            stops it and waits until GPUS are free
set -uo pipefail
# The host exports an HTTP proxy; requests to our own vLLM must not go through it.
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
CONTAINER_NAME=iaxl.vllm.qiuyu-vllm-block-hashes-dev
GPUS=${GPUS:-0,1}
PORT=8020
CONTROLLER_PORT=18720
WORKER_BASE_PORT=18820

# Same as run_validation.sh.
SERVE_ENV=(
    TP_SIZE=2 CUDA_VISIBLE_DEVICES="$GPUS" PORT="$PORT"
    IAXL_API_CONTROLLER_PORT="$CONTROLLER_PORT" IAXL_API_WORKER_BASE_PORT="$WORKER_BASE_PORT"
    IAXL_DDR_POOL_SIZE_GB=32
    IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0
)
SERVE_ARGS=(
    --kv-transfer-config "'{\"kv_connector\":\"KVShrinkConnector\",\"kv_connector_module_path\":\"kvshrink.kvshrink_connector\",\"kv_role\":\"kv_both\"}'"
    --trust-remote-code -tp 2 --gpu-memory-utilization 0.95 --max-model-len 32765
    --port "$PORT" --no-enable-prefix-caching
)

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

gpu_memory_used_mib() {
    nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1
}

find_serve_pid() {
    ps -eo pid,args | awk -v port="--port $PORT" \
        '$2 ~ /python/ && $3 ~ /\/vllm$/ && $4 == "serve" && index($0, port) {print $1}'
}

start() {
    local model_path=$1 vllm_log=$2 serve_pid=""
    if [[ -n "$(find_serve_pid)" ]]; then
        log "ERROR: a vLLM already listens on port $PORT (pid $(find_serve_pid))"
        return 1
    fi
    if (( $(gpu_memory_used_mib) >= 1000 )); then
        log "ERROR: GPU $GPUS in use"
        return 1
    fi
    log "starting vLLM $model_path on GPU $GPUS, log $vllm_log"
    mkdir -p "$(dirname "$vllm_log")"
    : > "$vllm_log"
    docker exec -w "$REPO_DIR" -d "$CONTAINER_NAME" bash -c "exec > $vllm_log 2>&1; \
        export MODEL=$model_path ${SERVE_ENV[*]}; \
        source setvars.sh; \
        export LD_PRELOAD=\"\$IAXL_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
        exec vllm serve $model_path ${SERVE_ARGS[*]}"
    for _ in $(seq 1 60); do
        serve_pid=$(find_serve_pid)
        [[ -n "$serve_pid" ]] && break
        sleep 1
    done
    if [[ -z "$serve_pid" ]]; then
        log "ERROR: vllm serve did not start; see $vllm_log"
        return 1
    fi
    for _ in $(seq 1 240); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null; then
            log "vLLM ready"
            echo "SERVE_PID=$serve_pid"
            return 0
        fi
        if ! ps -p "$serve_pid" > /dev/null; then
            log "ERROR: vLLM exited during startup; see $vllm_log"
            return 1
        fi
        sleep 5
    done
    log "ERROR: vLLM not ready after 20 minutes"
    return 1
}

stop() {
    local serve_pid container_id
    serve_pid=$(find_serve_pid)
    if [[ -n "$serve_pid" ]]; then
        log "stopping vLLM (pid $serve_pid)"
        docker exec "$CONTAINER_NAME" kill -TERM "$serve_pid" || true
        for _ in $(seq 1 60); do
            ps -p "$serve_pid" > /dev/null || break
            sleep 2
        done
    fi
    for _ in $(seq 1 30); do
        (( $(gpu_memory_used_mib) < 1000 )) && break
        sleep 2
    done
    # Only kill GPU processes that run in our container.
    container_id=$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")
    if (( $(gpu_memory_used_mib) >= 1000 )); then
        for pid in $(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader); do
            if grep -q "$container_id" "/proc/$pid/cgroup" 2> /dev/null; then
                log "killing leftover GPU process $pid of our container"
                docker exec "$CONTAINER_NAME" kill -9 "$pid" || true
            fi
        done
    fi
    log "GPU $GPUS memory used (max): $(gpu_memory_used_mib) MiB"
}

case "${1:-}" in
    start) start "$2" "$3" ;;
    stop) stop ;;
    *) echo "usage: $0 start <model path> <vllm log> | stop" >&2; exit 1 ;;
esac
