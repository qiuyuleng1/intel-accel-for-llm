#!/bin/bash
# Kimi-K3 (/home/intel/model-space/Kimi-K3, 2.8T MXFP4, KDA + MLA hybrid) with kvshrink on
# b300: TP=8 on all 8 GPUs, DSA / QAT / IAA / CPU compression all off and KV stored
# uncompressed, DDR pool 200 GB, MAX_MODEL_LEN 32768. vLLM options are those of
# examples/kvshrink-vllm-serve.sh plus --gpu-memory-utilization 0.9 (after the script's 0.8;
# vLLM takes the last one); KDA uses vLLM's default FlashKDA prefill, whose in-prefill
# checkpoints kvshrink saves.
# Async loads wait for every layer before the request runs: Kimi-K3's MLA forward never
# calls wait_for_layer_load, so a request must not start before its attention KV is in.
# Then kimi_k3_smoke_client.py: 2 short questions, a 60-shot gsm8k prompt sent twice plus
# two more questions on the same examples, and a 120-shot prompt (longer than one
# 16384-token chunk) sent twice plus one more question; each answer is checked and the
# connector log shows each request's kvshrink hit.
# vLLM is left running after the test (loading Kimi-K3 takes 20+ min); the next run reuses it,
# so its kvshrink code and vLLM options stay those of the run that started it.
#
# Runs INSIDE this clone's container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 iaxl.vllm-kimi-k3-qiuyu \
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
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=-1
export VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.9"
# Loading 1.5 TB of weights from disk took 1600 s on 11-12.
SERVE_READY_TIMEOUT_S=5400

CODE_MD5=$(cat kvshrink/*.py "$DEV_DIR/kimi_k3_smoke.sh" "$DEV_DIR/kimi_k3_smoke_client.py" \
    | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/kimi-k3-$(date +%Y%m%d_%H%M%S)_$CODE_MD5
# "<process group> <serve log>" of the vLLM left running by an earlier run.
SERVE_STATE=$DEV_DIR/results/kimi-k3-serve.state
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

SERVE_PGID=""
stop_serve() {
    kill -TERM -- "-$SERVE_PGID" 2> /dev/null
    local waited=0
    while (( waited < 300 )) && kill -0 -- "-$SERVE_PGID" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PGID" 2> /dev/null
    rm -f "$SERVE_STATE"
    log "serve stopped after ${waited}s"
}
trap 'exit 130' INT TERM

log "code md5 $CODE_MD5, results $RESULTS_DIR"
if [[ -f "$SERVE_STATE" ]]; then
    read -r SERVE_PGID SERVE_LOG < "$SERVE_STATE"
    if ! kill -0 -- "-$SERVE_PGID" 2> /dev/null; then
        SERVE_PGID=""
    fi
fi
if [[ -n "$SERVE_PGID" ]]; then
    log "reusing running vLLM (process group $SERVE_PGID), log $SERVE_LOG"
    # Only the part of the serve log written by this run is checked below.
    SERVE_LOG_START=$(stat -c %s "$SERVE_LOG")
else
    SERVE_LOG=$RESULTS_DIR/serve.log
    SERVE_LOG_START=0
    log "starting vLLM $MODEL, TP $TP_SIZE, log $SERVE_LOG"
    # Own process group, so stop_serve cannot touch any other vLLM.
    setsid bash examples/kvshrink-vllm-serve.sh > "$SERVE_LOG" 2>&1 &
    SERVE_PGID=$!
    echo "$SERVE_PGID $SERVE_LOG" > "$SERVE_STATE"
fi
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 -- "-$SERVE_PGID" 2> /dev/null; then
        log "ERROR: serve exited before becoming ready; last lines:"
        tr '\r' '\n' < "$SERVE_LOG" | grep -av "^$" | tail -30
        rm -f "$SERVE_STATE"
        exit 1
    fi
    if (( waited >= SERVE_READY_TIMEOUT_S )); then
        log "ERROR: serve not ready within ${SERVE_READY_TIMEOUT_S}s"
        stop_serve
        exit 1
    fi
    sleep 10
    waited=$((waited + 10))
done
log "serve ready after ${waited}s"
tr '\r' '\n' < "$SERVE_LOG" | grep -aE "KVShrinkConnector\(SCHEDULER\)|Registered .* KV cache layers|KDA prefill backend|Setting attention block size|GPU KV cache size|max_num_batched_tokens=" \
    | sed -E 's/^\(([A-Za-z_0-9 =]+)[^)]*\) //' | sort -u

if ! python3 "$DEV_DIR/kimi_k3_smoke_client.py" run --port "$PORT" --model "$MODEL" \
    --dataset-dir "$DEV_DIR/gsm8k-datasets" --out "$RESULTS_DIR/requests.jsonl"; then
    log "ERROR: client failed"
fi
# The kvstore monitor logs DDR pool usage every 10 s.
sleep 12
# Byte offset, not line count: the serve log contains '\r'.
serve_log_of_this_run() {
    tail -c "+$((SERVE_LOG_START + 1))" "$SERVE_LOG" | tr '\r' '\n'
}
serve_log_of_this_run | grep -a "\[monitor\]" | tail -1
serve_log_of_this_run | grep -a "get_num_new_matched_tokens" > "$RESULTS_DIR/connector_log.txt"
for pattern in "Traceback" " ERROR " "IAXL check failed" "cache key missing after retry"; do
    log "serve log \"$pattern\": $(serve_log_of_this_run | grep -ac "$pattern")"
done
if [[ -f "$RESULTS_DIR/requests.jsonl" ]]; then
    block_size=$(tr '\r' '\n' < "$SERVE_LOG" | grep -aoE "Setting attention block size to [0-9]+" \
        | tail -1 | awk '{print $NF}')
    python3 "$DEV_DIR/kimi_k3_smoke_client.py" report --requests "$RESULTS_DIR/requests.jsonl" \
        --connector-log "$RESULTS_DIR/connector_log.txt" --block-size "$block_size" \
        | tee "$RESULTS_DIR/report.txt"
fi
log "vLLM left running (process group $SERVE_PGID, log $SERVE_LOG); to stop it from the host:"
log "  docker exec iaxl.vllm-kimi-k3-qiuyu kill -TERM -- -$SERVE_PGID"
log "ALL_DONE $RESULTS_DIR"
