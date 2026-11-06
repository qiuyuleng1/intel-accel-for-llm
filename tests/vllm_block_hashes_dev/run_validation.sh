#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Validates kvshrink taking its DDR keys from vLLM's block hashes, on Qwen2.5-32B-Instruct and on
# the GDN hybrid Qwen3.8-27B, TP=2 on the two L20 of the remote test host (10.239.23.121), with
# compression, QAT, IAA and DSA off. Runs on that host, in tmux; vLLM runs in the container of
# remote_setup.sh. Per model:
#   A: start vLLM -> check_hits.py first (a fresh ~6000-token prompt: one miss, then 3 hits)
#      -> persist the DDR to disk -> stop vLLM
#   B: start vLLM again with the same settings -> check_hits.py restart (the same prompt must hit
#      what A persisted) -> stop vLLM
#   C: start vLLM with --prefix-caching-hash-algo xxhash and without PYTHONHASHSEED -> it must
#      exit at startup with the connector's ValueError
# Hit outputs are judged by their answer to the prompt's question, not byte identity (user,
# 11-06); the texts are in the reports for a human look.
# Results: results/<time>_<code md5>/<model>/, vLLM logs written there live; summary in run.log.
#   ./run_validation.sh                                          # both models
#   MODEL_PATHS=/media/model-space/Qwen3.8-27B ./run_validation.sh  # one model
set -uo pipefail
# Ctrl-C also kills the tee below; ignoring SIGPIPE keeps the cleanup (stopping vLLM) alive.
trap '' PIPE
# The host exports an HTTP proxy; requests to our own vLLM must not go through it.
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
CONTAINER_NAME=iaxl.vllm.qiuyu-vllm-block-hashes-dev
MODEL_PATHS=${MODEL_PATHS:-"/media/model-space/Qwen/Qwen2.5-32B-Instruct /media/model-space/Qwen3.8-27B"}
# Both L20; setvars.sh binds each rank to CPUs of its GPU's NUMA node.
GPUS=0,1
# Not used by other servers on this host (host network).
PORT=8020
CONTROLLER_PORT=18720
WORKER_BASE_PORT=18820
HITS=3
# ~6000-token prompt (user, 11-06): 300 records of the sensor log (check_hits.py).
RECORDS=300

SERVE_ENV=(
    TP_SIZE=2 CUDA_VISIBLE_DEVICES="$GPUS" PORT="$PORT"
    IAXL_API_CONTROLLER_PORT="$CONTROLLER_PORT" IAXL_API_WORKER_BASE_PORT="$WORKER_BASE_PORT"
    # The default (80% of host memory) is far more than this test needs.
    IAXL_DDR_POOL_SIZE_GB=32
    # No accelerator (user, 11-06): blocks are stored raw, no QAT, IAA or DSA.
    IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0
)
# examples/kvshrink-vllm-serve.sh, except --gpu-memory-utilization: 0.8 of a 46 GB L20 leaves
# Qwen2.5-32B (32.8 GB of weights per GPU at TP=2) about 4 GB for KV (user, 11-06: use 0.95).
SERVE_ARGS=(
    --kv-transfer-config "'{\"kv_connector\":\"KVShrinkConnector\",\"kv_connector_module_path\":\"kvshrink.kvshrink_connector\",\"kv_role\":\"kv_both\"}'"
    --trust-remote-code -tp 2 --gpu-memory-utilization 0.95 --max-model-len 32765
    --port "$PORT" --no-enable-prefix-caching
)

CODE_MD5=$(cd "$REPO_DIR" && cat $(git ls-files 'kvshrink/*.py' 'iaxl/*.py') \
    "$DEV_DIR/run_validation.sh" "$DEV_DIR/check_hits.py" | md5sum | cut -c1-8)
RESULTS_DIR="$DEV_DIR/results/$(date +%Y%m%d_%H%M%S)_$CODE_MD5"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

SERVE_PID=""
declare -A VERDICTS=()

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

gpu_memory_used_mib() {
    # The larger of the two GPUs.
    nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1
}

find_serve_pid() {
    # Only the python process running the vllm script: before its exec, the bash wrapper (and
    # the subshells setvars.sh forks) carry the same "vllm serve ... --port" text.
    ps -eo pid,args | awk -v port="--port $PORT" \
        '$2 ~ /python/ && $3 ~ /\/vllm$/ && $4 == "serve" && index($0, port) {print $1}'
}

stop_vllm() {
    # Also called after vLLM exited by itself (SERVE_PID empty): its workers may still hold the GPUs.
    if [[ -n "$SERVE_PID" ]]; then
        log "stopping vLLM (pid $SERVE_PID)"
        docker exec "$CONTAINER_NAME" kill -TERM "$SERVE_PID" || true
        for _ in $(seq 1 60); do
            ps -p "$SERVE_PID" > /dev/null || break
            sleep 2
        done
        SERVE_PID=""
    fi
    for _ in $(seq 1 30); do
        (( $(gpu_memory_used_mib) < 1000 )) && break
        sleep 2
    done
    # The host is shared: only kill GPU processes that run in our container.
    if (( $(gpu_memory_used_mib) >= 1000 )); then
        for pid in $(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader); do
            if grep -q "$CONTAINER_ID" "/proc/$pid/cgroup" 2> /dev/null; then
                log "killing leftover GPU process $pid of our container"
                docker exec "$CONTAINER_NAME" kill -9 "$pid" || true
            fi
        done
    fi
    log "GPU $GPUS memory used (max): $(gpu_memory_used_mib) MiB"
}

cleanup() {
    if [[ -n "$SERVE_PID" ]]; then
        stop_vllm || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_vllm() {
    # start_vllm <vllm log> <shell commands run before vllm> <extra vllm serve arguments>
    local vllm_log=$1 setup=$2 extra_args=$3
    log "starting vLLM $MODEL_NAME on GPU $GPUS, log $vllm_log"
    # Created by the host user, so the log stays writable after the run.
    : > "$vllm_log"
    docker exec -w "$REPO_DIR" -d "$CONTAINER_NAME" bash -c "exec > $vllm_log 2>&1; \
        export MODEL=$MODEL_PATH ${SERVE_ENV[*]}; \
        source setvars.sh; \
        export LD_PRELOAD=\"\$IAXL_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
        $setup \
        exec vllm serve $MODEL_PATH ${SERVE_ARGS[*]} $extra_args"
    for _ in $(seq 1 60); do
        SERVE_PID=$(find_serve_pid)
        [[ -n "$SERVE_PID" ]] && return 0
        sleep 1
    done
    log "ERROR: vllm serve did not start; see $vllm_log"
    return 1
}

wait_until_ready() {
    # Returns 0 once /health answers, 1 if vLLM exits first or does not answer in 20 minutes.
    for _ in $(seq 1 240); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null; then
            log "vLLM ready (pid $SERVE_PID)"
            return 0
        fi
        if ! ps -p "$SERVE_PID" > /dev/null; then
            log "vLLM exited during startup"
            SERVE_PID=""
            return 1
        fi
        sleep 5
    done
    log "ERROR: vLLM not ready after 20 minutes"
    return 1
}

run_check() {
    # run_check <phase> <out dir> <vllm log> [extra check_hits.py arguments]
    local phase=$1 out=$2 vllm_log=$3
    shift 3
    no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 python3 "$DEV_DIR/check_hits.py" \
        "$phase" --api "http://localhost:$PORT/v1/completions" --model "$MODEL_PATH" \
        --vllm-log "$vllm_log" --out "$out" --compare answer --records "$RECORDS" \
        --hits "$HITS" "$@" > "$out.log" 2>&1
    local status=$?
    grep '^SUMMARY' "$out.log" || tail -5 "$out.log"
    return $status
}

validate_model() {
    local model_dir=$RESULTS_DIR/$MODEL_NAME
    mkdir -p "$model_dir"

    log "=== $MODEL_NAME A: miss then hits ==="
    if ! start_vllm "$model_dir/vllm_a.log" "" "" || ! wait_until_ready; then
        VERDICTS["$MODEL_NAME:A"]="FAIL (vLLM did not start)"
        stop_vllm
        return
    fi
    grep -aE "KVShrinkConnector\(|KVShrink patch|KVShrink CPU rank" \
        "$model_dir/vllm_a.log" | sed -E 's/^.*\] //' | sort -u | head -12
    if run_check first "$model_dir/first" "$model_dir/vllm_a.log"; then
        VERDICTS["$MODEL_NAME:A"]=PASS
    else
        VERDICTS["$MODEL_NAME:A"]=FAIL
    fi
    log "persisting the DDR to disk"
    curl -s -X POST "http://localhost:$CONTROLLER_PORT/v1/cache/persist" \
        -H 'Content-Type: application/json' -d '{"count":999999}' > "$model_dir/persist.json"
    cat "$model_dir/persist.json"
    echo
    stop_vllm

    log "=== $MODEL_NAME B: restart, the same prompt must hit ==="
    if ! start_vllm "$model_dir/vllm_b.log" "" "" || ! wait_until_ready; then
        VERDICTS["$MODEL_NAME:B"]="FAIL (vLLM did not start)"
        stop_vllm
    else
        if run_check restart "$model_dir/restart" "$model_dir/vllm_b.log" \
            --first-dir "$model_dir/first"; then
            VERDICTS["$MODEL_NAME:B"]=PASS
        else
            VERDICTS["$MODEL_NAME:B"]=FAIL
        fi
        stop_vllm
    fi

    log "=== $MODEL_NAME C: xxhash without PYTHONHASHSEED must not start ==="
    if ! start_vllm "$model_dir/vllm_c.log" "unset PYTHONHASHSEED;" \
        "--prefix-caching-hash-algo xxhash"; then
        VERDICTS["$MODEL_NAME:C"]="FAIL (vllm serve did not launch)"
        stop_vllm
        return
    fi
    if wait_until_ready; then
        VERDICTS["$MODEL_NAME:C"]="FAIL (vLLM started)"
    elif grep -aq "requires PYTHONHASHSEED" "$model_dir/vllm_c.log"; then
        grep -a "requires PYTHONHASHSEED" "$model_dir/vllm_c.log" | sed -E 's/^.*\] //' | head -1
        VERDICTS["$MODEL_NAME:C"]=PASS
    else
        VERDICTS["$MODEL_NAME:C"]="FAIL (exited without the connector's error)"
    fi
    stop_vllm
}

log "branch $(git -C "$REPO_DIR" branch --show-current), code md5 $CODE_MD5, results $RESULTS_DIR"
git -C "$REPO_DIR" --no-pager diff --stat
# remote_setup.sh builds the image and then iaxl inside the container (pip install -e .); wait
# up to 3 hours for both.
CONTAINER_ID=""
for _ in $(seq 1 360); do
    CONTAINER_ID=$(docker inspect --format '{{if .State.Running}}{{.Id}}{{end}}' "$CONTAINER_NAME" 2> /dev/null)
    if [[ -n "$CONTAINER_ID" ]] && docker exec -w "$REPO_DIR" "$CONTAINER_NAME" \
        python3 -c "import iaxl.torch_ext, kvshrink.vllm_patch" > /dev/null 2>&1; then
        break
    fi
    CONTAINER_ID=""
    sleep 30
done
if [[ -z "$CONTAINER_ID" ]]; then
    log "ERROR: container $CONTAINER_NAME not running or iaxl not importable after 3 hours"
    exit 1
fi
log "container $CONTAINER_NAME ready"
if (( $(gpu_memory_used_mib) >= 1000 )); then
    log "ERROR: GPU $GPUS in use:"
    nvidia-smi -i "$GPUS" --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
    exit 1
fi
if [[ -n "$(find_serve_pid)" ]]; then
    log "ERROR: a vLLM already listens on port $PORT (pid $(find_serve_pid))"
    exit 1
fi

for MODEL_PATH in $MODEL_PATHS; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    validate_model
done

log "=== summary ==="
for MODEL_PATH in $MODEL_PATHS; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    for phase in A B C; do
        echo "$MODEL_NAME $phase: ${VERDICTS["$MODEL_NAME:$phase"]:-not run}"
    done
done
log "ALL_DONE $RESULTS_DIR"
