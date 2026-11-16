#!/bin/bash
# Qwen3.8-27B gsm8k twice on one kvshrink vLLM (pass 2 loads pass 1's KV); settings of the b300 repo's
# gsm8k_two_passes_b300.sh, LIMIT questions (default 200).
# From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 -e LIMIT=30 iaxl.vllm-kimi-k3-qiuyu \
#       bash -lc 'bash tests/bench_dev/qwen38_gsm8k_two_passes.sh'
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

# The Kimi-K3 container mounts only Kimi-K3's model dir; QWEN_MODEL can point to hardlinks under /_data.
export MODEL=${QWEN_MODEL:-/home/intel/model-space/Qwen3.8-27B}
export TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6
export VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'
export KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'
export KVSHRINK_IAA_DEVICES='0|1|2|3'
export MAX_MODEL_LEN=256128 PORT=8000
export IAXL_DSA_GD_ENABLE=1 IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1
export IAXL_CPU_ZIP_ENABLE=0 IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0
export IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 IAXL_SCRATCH_POOL_SIZE_GB=8

DATASET_DIR=$DEV_DIR/gsm8k-datasets
LIMIT=${LIMIT:-200}
PARALLEL=50
GENERATION_CONFIG='{"max_tokens": 2048, "temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
DATASET_ARGS='{"gsm8k": {"few_shot_num": 26}}'
SERVE_READY_TIMEOUT_S=1800
PASS_TIMEOUT_S=1800

CODE_MD5=$(cat kvshrink/*.py "$DEV_DIR/qwen38_gsm8k_two_passes.sh" "$DEV_DIR/gsm8k_compare.py" \
    | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/qwen38-gsm8k-$(date +%Y%m%d_%H%M%S)_${CODE_MD5}_limit$LIMIT
MODEL_DIR=$RESULTS_DIR/$(basename "$MODEL")
SERVE_LOG=$MODEL_DIR/serve.log
mkdir -p "$MODEL_DIR/pass1" "$MODEL_DIR/pass2"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

SERVE_PID=""

stop_serve() {
    if [[ -z "$SERVE_PID" ]]; then
        return 0
    fi
    kill -TERM -- "-$SERVE_PID" 2> /dev/null
    local waited=0
    while (( waited < 180 )) && kill -0 -- "-$SERVE_PID" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PID" 2> /dev/null
    SERVE_PID=""
    log "serve stopped after ${waited}s"
}
trap stop_serve EXIT
trap 'exit 130' INT TERM

run_pass() {
    # run_pass <pass name>: one gsm8k run, then the connector log lines it produced.
    local pass=$1
    local pass_dir=$MODEL_DIR/$pass
    local log_start log_end status
    log_start=$(stat -c %s "$SERVE_LOG")
    log "$pass: gsm8k, $LIMIT questions, $PARALLEL in parallel"
    timeout "$PASS_TIMEOUT_S" evalscope eval \
        --model "$MODEL" \
        --api-url "http://127.0.0.1:$PORT/v1" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets gsm8k \
        --dataset-dir "$DATASET_DIR" \
        --limit "$LIMIT" \
        --eval-batch-size "$PARALLEL" \
        --work-dir "$pass_dir" \
        --dataset-args "$DATASET_ARGS" \
        --generation-config "$GENERATION_CONFIG" > "$pass_dir.evalscope.log" 2>&1
    status=$?
    sleep 2
    log_end=$(stat -c %s "$SERVE_LOG")
    # Byte offsets, not line counts: the serve log contains '\r' progress output.
    tail -c +$((log_start + 1)) "$SERVE_LOG" | head -c $((log_end - log_start)) \
        | tr '\r' '\n' | grep -a "get_num_new_matched_tokens" > "$pass_dir/connector_log.txt"
    log "$pass: evalscope exit $status, $(wc -l < "$pass_dir/connector_log.txt") connector lines," \
        "score $(grep -ohE '"score": [0-9.]+' "$pass_dir"/*/reports/*/gsm8k.json | head -1)"
    return $status
}

log "code md5 $CODE_MD5, results $RESULTS_DIR"
log "starting vLLM $MODEL on GPU $CUDA_VISIBLE_DEVICES, log $SERVE_LOG"
# Own process group, so stop_serve cannot touch any other vLLM.
setsid bash examples/kvshrink-vllm-serve.sh > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2> /dev/null; then
        log "ERROR: serve exited before becoming ready"
        exit 1
    fi
    if (( waited >= SERVE_READY_TIMEOUT_S )); then
        log "ERROR: serve not ready within ${SERVE_READY_TIMEOUT_S}s"
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
BLOCK_SIZE=$(tr '\r' '\n' < "$SERVE_LOG" | grep -aoE "Setting attention block size to [0-9]+" \
    | tail -1 | awk '{print $NF}')
log "serve ready after ${waited}s, block size $BLOCK_SIZE"

PASSES_OK=0
if run_pass pass1 && run_pass pass2; then
    PASSES_OK=1
fi
stop_serve
# The hook line must not appear for Qwen: the connector patches only a loaded Kimi-K3 MLA module.
for pattern in "Traceback" " ERROR " "dsa batch failed" "IAXL check failed" \
    "Mamba align patch registered" "Kimi-K3 MLA KV transfer hook registered"; do
    log "serve log \"$pattern\": $(tr '\r' '\n' < "$SERVE_LOG" | grep -ac "$pattern")"
done
if (( PASSES_OK )); then
    python3 "$DEV_DIR/gsm8k_compare.py" "$MODEL_DIR" "$BLOCK_SIZE"
else
    log "ERROR: a pass failed, no comparison"
fi
log "ALL_DONE $RESULTS_DIR"
