#!/bin/bash
# Kimi-K3 (/home/intel/model-space/Kimi-K3, 1.5 TB bf16, KDA + MLA hybrid) with kvshrink
# on b300: TP=8 on all 8 GPUs, DSA / QAT / IAA / CPU compression all off and KV stored
# uncompressed, DDR pool 200 GB, MAX_MODEL_LEN 32768, --gpu-memory-utilization 0.9
# (appended after the serve script's 0.8; vLLM takes the last one). Everything else is
# examples/kvshrink-vllm-serve.sh. Then kimi_k3_smoke_client.py: 2 short questions and
# a 60-shot gsm8k prompt sent twice plus two more questions on the same examples, each
# answer checked; the connector log shows each request's kvshrink hit.
#
# Runs INSIDE the KVShrink container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-b300 iaxl.vllm-fddeef0-qiuyu \
#       bash -lc 'bash tests/bench_dev/kimi_k3_smoke.sh'
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

export MODEL=/home/intel/model-space/Kimi-K3
export TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_MODEL_LEN=32768 PORT=8000
export IAXL_DSA_GD_ENABLE=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=0
export IAXL_KV_COMPRESSION=0
export IAXL_DDR_POOL_TOTAL_SIZE_GB=200
export VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.9"
# Loading 1.5 TB of weights from disk.
SERVE_READY_TIMEOUT_S=5400

CODE_MD5=$(cat kvshrink/*.py "$DEV_DIR/kimi_k3_smoke.sh" "$DEV_DIR/kimi_k3_smoke_client.py" \
    | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/kimi-k3-$(date +%Y%m%d_%H%M%S)_$CODE_MD5
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
    while (( waited < 300 )) && kill -0 -- "-$SERVE_PID" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PID" 2> /dev/null
    SERVE_PID=""
    log "serve stopped after ${waited}s"
}
trap stop_serve EXIT
trap 'exit 130' INT TERM

log "code md5 $CODE_MD5, results $RESULTS_DIR"
log "starting vLLM $MODEL, TP $TP_SIZE, log $SERVE_LOG"
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
    sleep 10
    waited=$((waited + 10))
done
log "serve ready after ${waited}s"
tr '\r' '\n' < "$SERVE_LOG" | grep -aE "KVShrinkConnector\(SCHEDULER\)|Setting attention block size|GPU KV cache size|max_num_batched_tokens=" \
    | sed -E 's/^\(([A-Za-z_0-9 =]+)[^)]*\) //' | sort -u

if ! python3 "$DEV_DIR/kimi_k3_smoke_client.py" run --port "$PORT" --model "$MODEL" \
    --dataset-dir "$DEV_DIR/gsm8k-datasets" --out "$RESULTS_DIR/requests.jsonl"; then
    log "ERROR: client failed"
fi
sleep 2
tr '\r' '\n' < "$SERVE_LOG" | grep -a "get_num_new_matched_tokens" > "$RESULTS_DIR/connector_log.txt"
stop_serve
for pattern in "Traceback" " ERROR " "IAXL check failed" "cache key missing after retry"; do
    log "serve log \"$pattern\": $(tr '\r' '\n' < "$SERVE_LOG" | grep -ac "$pattern")"
done
if [[ -f "$RESULTS_DIR/requests.jsonl" ]]; then
    python3 "$DEV_DIR/kimi_k3_smoke_client.py" report --requests "$RESULTS_DIR/requests.jsonl" \
        --connector-log "$RESULTS_DIR/connector_log.txt" | tee "$RESULTS_DIR/report.txt"
fi
log "ALL_DONE $RESULTS_DIR"
