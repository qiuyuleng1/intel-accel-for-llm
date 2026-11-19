#!/bin/bash
# kvshrink + Rust frontend compatibility smoke test on b300. Starts
# examples/kvshrink-vllm-serve.sh as test-qwen38-27B-rust.sh's kvshrink-zip-dsa
# setting does (TP=4 on GPU0/2/4/6, FLASH_ATTN, QAT + IAA compression, DSA,
# VLLM_USE_RUST_FRONTEND=1, chat template without generation tags), sends one
# ~4000-token completion twice (rust_kvshrink_smoke_client.py), prints the
# connector's get_num_new_matched_tokens lines, and stops vLLM.
# Results: tests/bench_dev/results/rust-kvshrink-smoke-<time>/ (serve.log,
# run.log, request.json, response_*.json, connector_hits.txt).
#
# Runs INSIDE the KVShrink container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-b300 iaxl.vllm-fddeef0-qiuyu \
#       bash -lc 'bash tests/bench_dev/rust_kvshrink_smoke.sh'
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

export MODEL=/home/intel/model-space/Qwen3.8-27B
export TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6
export VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'
export KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0' KVSHRINK_IAA_DEVICES='0|1|2|3'
export IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0 IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=1000
export MAX_MODEL_LEN=256128 ATTENTION_BACKEND=FLASH_ATTN PORT=8000
export VLLM_USE_RUST_FRONTEND=1
export VLLM_SERVE_EXTRA_ARGS="--chat-template $DEV_DIR/qwen38-27B-chat-template-no-generation.jinja"
SERVE_READY_TIMEOUT_S=600

RESULTS_DIR=$DEV_DIR/results/rust-kvshrink-smoke-$(date +%Y%m%d_%H%M%S)
SERVE_LOG=$RESULTS_DIR/serve.log
mkdir -p "$RESULTS_DIR"
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

if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    log "ERROR: something already serves on port $PORT"
    exit 1
fi
log "results $RESULTS_DIR; starting vLLM (Rust frontend, kvshrink, FLASH_ATTN)"
# Own process group, so stop_serve cannot touch any other vLLM.
setsid bash examples/kvshrink-vllm-serve.sh > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2> /dev/null; then
        log "ERROR: serve exited before becoming ready; last lines:"
        tr '\r' '\n' < "$SERVE_LOG" | grep -av "^$" | tail -30
        exit 1
    fi
    if (( waited >= SERVE_READY_TIMEOUT_S )); then
        log "ERROR: serve not ready within ${SERVE_READY_TIMEOUT_S}s"
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
log "serve ready after ${waited}s"
tr '\r' '\n' < "$SERVE_LOG" | grep -aE "Launching Rust frontend|Setting attention block size|Using [A-Z_]+ attention backend" \
    | sed -E 's/--args-json.*//' | sort -u | cut -c1-200

python3 "$DEV_DIR/rust_kvshrink_smoke_client.py" "$RESULTS_DIR" "$PORT"
sleep 2
tr '\r' '\n' < "$SERVE_LOG" | grep -a "get_num_new_matched_tokens" > "$RESULTS_DIR/connector_hits.txt"
log "connector lines (externally-cached tokens per scheduler query):"
sed -E 's/^.*get_num_new_matched_tokens/get_num_new_matched_tokens/' "$RESULTS_DIR/connector_hits.txt" | cut -c1-200
log "ERROR lines in serve.log: $(tr '\r' '\n' < "$SERVE_LOG" | grep -ac ' ERROR ')"
