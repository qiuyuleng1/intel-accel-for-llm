#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# gsm8k accuracy with kvshrink taking its DDR keys from vLLM's block hashes: per model, one vLLM
# on the remote test host (serve_vllm.sh: TP=2 on the two L20, no accelerator, same settings
# as run_validation.sh) and the same gsm8k run twice against it. Pass 1 computes the prompts
# and stores their KV in DDR; pass 2 sends the same prompts, which load it. Then
# gsm8k_compare.py compares the two passes' accuracy, per-question correctness and outputs.
# gsm8k settings are those of run-gsm8k.sh in KVCacheClip/internal/daocloud/test/accuracy-gsm8k
# (evalscope, 200 questions, 4-shot, 50 in parallel, max_tokens 2048, temperature 0); Qwen3.8
# additionally gets enable_thinking=false (user, 11-06) and 26 shots: its kvshrink block is 784
# tokens and 4-shot prompts (652-762 tokens) fill none, so nothing would be stored or loaded;
# 26 shots (~5500 tokens) as in LMCache_v0.5.6/qiuyu_exp/qwen38_hybrid/run_gsm8k.sh. evalscope
# runs on this host and reaches vLLM through an SSH tunnel, because the remote host has no
# evalscope and installs through a slow proxy. SERVER_HOST=local runs vLLM on this host instead,
# in the container of start_container.sh on GPUS. Run it on this host, in tmux.
#   ./gsm8k_two_passes.sh                                            # both models, remote host
#   MODEL_PATHS=/media/model-space/Qwen3.8-27B ./gsm8k_two_passes.sh  # one model
#   SERVER_HOST=local GPUS=1,2 MODEL_PATHS=/media/model-space/Qwen3.8-27B ./gsm8k_two_passes.sh
set -uo pipefail
trap '' PIPE
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
REMOTE=ubuntu@10.239.23.121
REMOTE_DEV_DIR=/media/workspace/qiuyu/intel-accel-for-llm-1026-vbh/tests/vllm_block_hashes_dev
SERVER_HOST=${SERVER_HOST:-remote}
GPUS=${GPUS:-0,1}
MODEL_PATHS=${MODEL_PATHS:-"/media/model-space/Qwen/Qwen2.5-32B-Instruct /media/model-space/Qwen3.8-27B"}
GSM8K_DIR=/home/pese/qiuyu/Al-dsv4/KVCacheClip/internal/daocloud/test/accuracy-gsm8k
DATASET_DIR=$GSM8K_DIR/evalscope/datasets
EVALSCOPE=/home/pese/.local/bin/evalscope
# run-gsm8k.sh defaults.
LIMIT=200
PARALLEL=50
# serve_vllm.sh's port; on the remote host evalscope reaches it through LOCAL_PORT here.
SERVER_PORT=8020
LOCAL_PORT=18020
# serve_vllm.sh's kvshrink management port.
CONTROLLER_PORT=18720
# A pass takes about 2 minutes; evalscope waits forever for a request that never returns
# (11-06: one request of 200 stuck in pass 1), so cap it.
PASS_TIMEOUT_S=1800

CODE_MD5=$(cd "$REPO_DIR" && cat $(git ls-files 'kvshrink/*.py' 'iaxl/*.py') \
    "$DEV_DIR/gsm8k_two_passes.sh" "$DEV_DIR/serve_vllm.sh" "$DEV_DIR/gsm8k_compare.py" \
    | md5sum | cut -c1-8)
RUN_NAME="gsm8k-$(date +%Y%m%d_%H%M%S)_$CODE_MD5"
RESULTS_DIR="$DEV_DIR/results/$RUN_NAME"
if [[ "$SERVER_HOST" == local ]]; then
    SERVER_DEV_DIR=$DEV_DIR
    API_PORT=$SERVER_PORT
else
    SERVER_DEV_DIR=$REMOTE_DEV_DIR
    API_PORT=$LOCAL_PORT
fi
SERVER_RESULTS_DIR="$SERVER_DEV_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

TUNNEL_PID=""
SERVER_UP=0

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

on_server() {
    # Runs a shell command on the host that runs vLLM.
    if [[ "$SERVER_HOST" == local ]]; then
        bash -c "$1"
    else
        ssh -o BatchMode=yes "$REMOTE" "$1"
    fi
}

cleanup() {
    if (( SERVER_UP )); then
        on_server "GPUS=$GPUS bash $SERVER_DEV_DIR/serve_vllm.sh stop" || true
    fi
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2> /dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

run_pass() {
    # run_pass <pass name> <generation config>: one gsm8k run, the command of run-gsm8k.sh.
    local pass=$1 generation_config=$2
    local pass_dir=$MODEL_DIR/$pass
    local log_start log_end
    log_start=$(on_server "stat -c %s $SERVER_VLLM_LOG")
    log "$MODEL_NAME $pass: gsm8k, $LIMIT questions, $PARALLEL in parallel"
    timeout "$PASS_TIMEOUT_S" "$EVALSCOPE" eval \
        --model "$MODEL_PATH" \
        --api-url "http://127.0.0.1:$API_PORT/v1" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets gsm8k \
        --dataset-dir "$DATASET_DIR" \
        --limit "$LIMIT" \
        --eval-batch-size "$PARALLEL" \
        --work-dir "$pass_dir" \
        "${DATASET_ARGS[@]}" \
        --generation-config "$generation_config" > "$pass_dir.evalscope.log" 2>&1
    local status=$?
    sleep 2
    log_end=$(on_server "stat -c %s $SERVER_VLLM_LOG")
    # Byte offsets, not line counts: the log contains '\r' progress output.
    on_server "tail -c +$((log_start + 1)) $SERVER_VLLM_LOG | head -c $((log_end - log_start))" \
        | grep -a "get_num_new_matched_tokens" > "$pass_dir/connector_log.txt"
    log "$MODEL_NAME $pass: evalscope exit $status, $(wc -l < "$pass_dir/connector_log.txt") connector lines," \
        "score $(grep -ohE '"score": [0-9.]+' "$pass_dir"/*/reports/*/gsm8k.json | head -1)"
    if (( status != 0 )); then
        # Exit 124: the pass hit PASS_TIMEOUT_S. Keep what vLLM and kvshrink still hold.
        on_server "curl -s --noproxy '*' http://localhost:$SERVER_PORT/metrics" \
            > "$pass_dir/vllm_metrics_at_failure.txt"
        on_server "curl -s --noproxy '*' http://localhost:$CONTROLLER_PORT/v1/cache/status" \
            > "$pass_dir/cache_status_at_failure.json"
    fi
    return $status
}

log "branch $(git -C "$REPO_DIR" branch --show-current), code md5 $CODE_MD5, results $RESULTS_DIR"
log "vLLM on $SERVER_HOST host, GPU $GPUS, logs in $SERVER_RESULTS_DIR"
if [[ "$SERVER_HOST" != local ]]; then
    rsync -a -e "ssh -o BatchMode=yes" "$DEV_DIR"/*.sh "$DEV_DIR"/*.py "$REMOTE:$REMOTE_DEV_DIR/"
    if ! on_server "md5sum $REMOTE_DEV_DIR/../../kvshrink/kvshrink_connector.py" \
        | grep -q "$(md5sum < "$REPO_DIR/kvshrink/kvshrink_connector.py" | cut -c1-32)"; then
        log "ERROR: the remote kvshrink_connector.py differs from this host's"
        exit 1
    fi
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$SERVER_PORT" "$REMOTE" &
    TUNNEL_PID=$!
    sleep 3
    kill -0 "$TUNNEL_PID" || exit 1
fi

for MODEL_PATH in $MODEL_PATHS; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    MODEL_DIR=$RESULTS_DIR/$MODEL_NAME
    SERVER_VLLM_LOG=$SERVER_RESULTS_DIR/$MODEL_NAME/vllm.log
    mkdir -p "$MODEL_DIR/pass1" "$MODEL_DIR/pass2"
    GENERATION_CONFIG='{"max_tokens": 2048, "temperature": 0.0}'
    DATASET_ARGS=()
    if [[ "$MODEL_NAME" == Qwen3.8-27B ]]; then
        # chat_template_kwargs must go in extra_body: evalscope 1.9.1 does not forward it as a
        # top-level generation config key.
        GENERATION_CONFIG='{"max_tokens": 2048, "temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
        DATASET_ARGS=(--dataset-args '{"gsm8k": {"few_shot_num": 26}}')
    fi

    log "=== $MODEL_NAME ==="
    # Set before starting, so that an interrupt during the start also stops the server.
    SERVER_UP=1
    if ! on_server "GPUS=$GPUS bash $SERVER_DEV_DIR/serve_vllm.sh start $MODEL_PATH $SERVER_VLLM_LOG"; then
        log "ERROR: vLLM did not start for $MODEL_NAME"
        on_server "GPUS=$GPUS bash $SERVER_DEV_DIR/serve_vllm.sh stop"
        SERVER_UP=0
        continue
    fi
    BLOCK_SIZE=$(on_server "grep -aoE 'KVShrinkConnector\(SCHEDULER\): block_size=[0-9]+' $SERVER_VLLM_LOG" \
        | head -1 | grep -oE '[0-9]+$')
    log "$MODEL_NAME: kvshrink block size $BLOCK_SIZE"
    PASSES_OK=0
    if run_pass pass1 "$GENERATION_CONFIG" && run_pass pass2 "$GENERATION_CONFIG"; then
        PASSES_OK=1
    fi
    on_server "GPUS=$GPUS bash $SERVER_DEV_DIR/serve_vllm.sh stop"
    SERVER_UP=0
    if (( PASSES_OK )); then
        python3 "$DEV_DIR/gsm8k_compare.py" "$MODEL_DIR" "$BLOCK_SIZE"
    else
        log "ERROR: $MODEL_NAME: a pass failed, no comparison"
    fi
done

log "ALL_DONE $RESULTS_DIR"
