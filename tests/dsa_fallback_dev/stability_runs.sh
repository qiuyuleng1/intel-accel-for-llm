#!/bin/bash
# DSA vs CUDA batch stability on 6000d (doc/design/dsa-auto-fallback.zh-CN.md): settings A-D interleaved, GPU0 + wq0.0.
# Usage: RESULT_DIR=<dir> bash tests/dsa_fallback_dev/stability_runs.sh   (DRY_RUN=1 only prints the commands)
# Needs the container started by ./start.sh in another tmux window, logging to $RESULT_DIR/start.log.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
ROUNDS=${ROUNDS:-10}
SETTINGS=${SETTINGS:-"A B C D"}
DRY_RUN=${DRY_RUN:-0}
CONTAINER_READY_TIMEOUT_S=${CONTAINER_READY_TIMEOUT_S:-1800}
GPU_BUSY_MIB=500
FRAG_SIZES=256K,384K,512K,640K,768K,896K,1M,1152K,1280K,1408K,1536K,1664K,1792K,1920K,2M
# 15 fragment sizes x 2 directions x 2 methods
EXPECTED_RESULT_LINES=60

declare -A TOTAL_GIB=([A]=0.0625 [B]=0.0625 [C]=0.125 [D]=0.125)
declare -A WARMUP=([A]=1 [B]=2 [C]=1 [D]=2)
declare -A ITERATIONS=([A]=3 [B]=10 [C]=3 [D]=10)

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*" | tee -a "$RESULT_DIR/progress.log"
}

wait_container_ready() {
    local waited_s=0 start_log="$RESULT_DIR/start.log" container_start_line=""
    log "waiting for container $CONTAINER + pip install -e . (watching $start_log)"
    while true; do
        # The typed command line "echo START_SH_EXIT=$?" is in start.log too, so require a digit.
        if grep -a -q -E "START_SH_EXIT=[0-9]+" "$start_log" 2>/dev/null; then
            log "ERROR: start.sh exited, see $start_log"
            return 1
        fi
        # docker build output also has "Successfully installed", so only read lines after the container is up.
        if [[ -z "$container_start_line" && "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == true ]]; then
            container_start_line=$(wc -l <"$start_log")
            log "container $CONTAINER running, waiting for pip install -e . (start.log line $container_start_line)"
        fi
        if [[ -n "$container_start_line" ]]; then
            if tail -n +"$container_start_line" "$start_log" | grep -a -q -E "error: subprocess-exited-with-error|^ERROR: (Failed|Could not)"; then
                log "ERROR: pip install -e . failed in the container, see $start_log"
                return 1
            fi
            if tail -n +"$container_start_line" "$start_log" | grep -a -q "Successfully installed"; then
                log "container $CONTAINER ready (waited ${waited_s}s)"
                return 0
            fi
        fi
        if ((waited_s >= CONTAINER_READY_TIMEOUT_S)); then
            log "ERROR: container not ready after ${CONTAINER_READY_TIMEOUT_S}s, see $start_log"
            return 1
        fi
        sleep 10
        waited_s=$((waited_s + 10))
    done
}

# Someone else on GPU0 or wq0.0 would distort the numbers, so stop instead of measuring.
require_gpu0_and_wq_free() {
    local gpu_memory_mib wq_clients
    gpu_memory_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    wq_clients=$(cat /sys/bus/dsa/devices/wq0.0/clients)
    if ((gpu_memory_mib > GPU_BUSY_MIB)); then
        log "ERROR: GPU0 busy (${gpu_memory_mib} MiB used)"
        return 1
    fi
    if [[ "$wq_clients" != 0 ]]; then
        log "ERROR: wq0.0 busy (clients=$wq_clients)"
        return 1
    fi
}

run_one() {
    local setting=$1 round=$2
    local run_log="$RESULT_DIR/$setting/run$round.log"
    local command=(
        docker exec -w "$REPO_DIR" "$CONTAINER" bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
        --methods batch,iaxl --direction both --frag-sizes "$FRAG_SIZES"
        --total-gib "${TOTAL_GIB[$setting]}" --warmup "${WARMUP[$setting]}" --iterations "${ITERATIONS[$setting]}"
        --output-dir "$RESULT_DIR" --plot ''
    )
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: ${command[*]} > $run_log 2>&1"
        return 0
    fi
    mkdir -p "$RESULT_DIR/$setting"
    if ! "${command[@]}" >"$run_log" 2>&1; then
        log "ERROR: benchmark exited non-zero, see $run_log"
        return 1
    fi
    if grep -q " FAIL$" "$run_log"; then
        log "ERROR: copy check FAIL, see $run_log"
        return 1
    fi
    local result_lines
    result_lines=$(grep -c -E " (PASS|FAIL)$" "$run_log")
    if ((result_lines != EXPECTED_RESULT_LINES)); then
        log "ERROR: $result_lines result lines (expected $EXPECTED_RESULT_LINES), see $run_log"
        return 1
    fi
}

mkdir -p "$RESULT_DIR"
log "start: settings=$SETTINGS rounds=$ROUNDS result_dir=$RESULT_DIR"
if [[ "$DRY_RUN" != 1 ]]; then
    wait_container_ready || exit 1
    {
        echo "git: $(git -C "$REPO_DIR" rev-parse HEAD) branch $(git -C "$REPO_DIR" branch --show-current)"
        echo "image: $(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
        md5sum "$REPO_DIR"/benchmark/tensor_xfer/tensor_xfer_benchmark.{sh,py} "$REPO_DIR/tests/dsa_fallback_dev/stability_runs.sh"
    } >"$RESULT_DIR/env.txt"
fi

setting_count=$(wc -w <<<"$SETTINGS")
total_runs=$((ROUNDS * setting_count))
run_index=0
first_run_started=$(date +%s)
for round in $(seq 1 "$ROUNDS"); do
    for setting in $SETTINGS; do
        run_index=$((run_index + 1))
        if [[ "$DRY_RUN" != 1 ]]; then
            require_gpu0_and_wq_free || exit 1
        fi
        run_started=$(date +%s)
        run_one "$setting" "$round" || exit 1
        now=$(date +%s)
        seconds_left=$(((now - first_run_started) * (total_runs - run_index) / run_index))
        log "run $run_index/$total_runs done: setting $setting round $round ($((now - run_started))s), about $((seconds_left / 60))m$((seconds_left % 60))s left"
    done
done

if [[ "$DRY_RUN" != 1 ]]; then
    python3 "$REPO_DIR/tests/dsa_fallback_dev/stability_tables.py" "$RESULT_DIR" >"$RESULT_DIR/tables.md"
fi
log "ALL DONE. Tables: $RESULT_DIR/tables.md"
