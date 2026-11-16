#!/bin/bash
# Kimi-K3 with LMCache 0.5.7 (image qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7, vLLM 0.29.0),
# in the same layout as kvshrink's DP2 TP4 run: DP2 TP4 + EP, --max-num-seqs 128,
# --gpu-memory-utilization 0.9, --max-model-len -1, vLLM prefix caching off (the image's
# lmcache_hybrid_patch then switches the KDA layers to mamba_cache_mode='align').
# LMCache MP server: --chunk-size 1536 (the vLLM block size at TP4), --separate-object-groups,
# L1 100 GB. Then, on the same vLLM:
#   1. kimi_k3_smoke_client.py, every request once per DP rank; the hit is vLLM's per-request
#      cached_tokens (--enable-prompt-tokens-details);
#   2. kimi_k3_capacity.sh cases 32k x 6 and 128k x 1 per DP rank (12 and 2 in total).
# vLLM and the LMCache server are left running.
#
# Runs INSIDE the LMCache image container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 lmcache.vllm-kimi-k3-qiuyu \
#       bash -lc 'bash tests/bench_dev/kimi_k3_lmcache.sh'
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

MODEL=/home/intel/model-space/Kimi-K3
DP_SIZE=2
TP_SIZE=4
PORT=8000
LMCACHE_PORT=5560
LMCACHE_HTTP_PORT=8160
LMCACHE_CHUNK_SIZE=1536
LMCACHE_L1_SIZE_GB=100
SERVE_READY_TIMEOUT_S=5400
# vLLM's default 600 s ran out during CUDA graph capture (11-13: weights took 193 s, KV profiling 108 s).
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# The image has no `datasets`; it is installed for the smoke client only (pip --target).
SMOKE_PYTHONPATH=/opt/smoke-deps
# "<vLLM process group> <serve log>", the format kimi_k3_capacity.sh reads.
SERVE_STATE=$DEV_DIR/results/kimi-k3-lmcache-serve.state

CODE_MD5=$(cat "$DEV_DIR/kimi_k3_lmcache.sh" "$DEV_DIR/kimi_k3_smoke_client.py" \
    "$DEV_DIR/kimi_k3_capacity.sh" | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/kimi-k3-lmcache-$(date +%Y%m%d_%H%M%S)_$CODE_MD5
SERVE_LOG=$RESULTS_DIR/serve.log
LMCACHE_SERVER_LOG=$RESULTS_DIR/lmcache_server.log
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

# Zombies left by a dead process still answer kill -0, so look for a live one.
group_alive() {
    ps -o stat= -g "$1" 2> /dev/null | grep -qv '^Z'
}

if [[ -f "$SERVE_STATE" ]]; then
    log "ERROR: $SERVE_STATE exists; an LMCache vLLM may still be running"
    exit 1
fi
log "code md5 $CODE_MD5, results $RESULTS_DIR"
log "$(python3 -c 'from importlib.metadata import version; print("vllm", version("vllm"), "lmcache", version("lmcache"))')"

# Own process group, so it can be stopped on its own later.
setsid python3 -c 'import sys; from lmcache.cli.main import main; sys.argv[0] = "lmcache"; sys.exit(main())' \
    server --host 127.0.0.1 --port "$LMCACHE_PORT" \
    --http-host 127.0.0.1 --http-port "$LMCACHE_HTTP_PORT" \
    --chunk-size "$LMCACHE_CHUNK_SIZE" --separate-object-groups \
    --l1-size-gb "$LMCACHE_L1_SIZE_GB" --eviction-policy LRU \
    > "$LMCACHE_SERVER_LOG" 2>&1 &
server_pgid=$!
waited=0
until curl -sf -o /dev/null "http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck"; do
    if ! kill -0 "$server_pgid" 2> /dev/null; then
        log "ERROR: lmcache server exited before becoming ready; last lines:"
        tail -20 "$LMCACHE_SERVER_LOG"
        exit 1
    fi
    if (( waited >= 600 )); then
        log "ERROR: lmcache server not ready within 600s"
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
log "lmcache server ready after ${waited}s (process group $server_pgid, log $LMCACHE_SERVER_LOG)"

kv_transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":'"$LMCACHE_PORT"'}}'
log "starting vLLM $MODEL, DP $DP_SIZE TP $TP_SIZE, log $SERVE_LOG"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid vllm serve "$MODEL" \
    --kv-transfer-config "$kv_transfer_config" \
    --no-enable-prefix-caching \
    --enable-prompt-tokens-details \
    --trust-remote-code \
    -tp "$TP_SIZE" --data-parallel-size "$DP_SIZE" --enable-expert-parallel \
    --gpu-memory-utilization 0.9 --max-num-seqs 128 --max-model-len -1 \
    --port "$PORT" > "$SERVE_LOG" 2>&1 &
vllm_pgid=$!
echo "$vllm_pgid $SERVE_LOG" > "$SERVE_STATE"
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    if ! group_alive "$vllm_pgid"; then
        log "ERROR: vLLM exited before becoming ready; last lines:"
        tr '\r' '\n' < "$SERVE_LOG" | grep -av "^$" | tail -30
        rm -f "$SERVE_STATE"
        kill -TERM -- "-$server_pgid" 2> /dev/null
        log "lmcache server stopped (vLLM did not start)"
        exit 1
    fi
    if (( waited >= SERVE_READY_TIMEOUT_S )); then
        log "ERROR: vLLM not ready within ${SERVE_READY_TIMEOUT_S}s (left running for inspection)"
        exit 1
    fi
    sleep 10
    waited=$((waited + 10))
done
log "vLLM ready after ${waited}s"
tr '\r' '\n' < "$SERVE_LOG" | grep -aE "LMCache hybrid patch|Setting attention block size|GPU KV cache size|Auto-fit max_model_len|Model loading took|Resolved LMCache MP geometry" \
    | sed -E 's/^\(([A-Za-z_0-9 =]+)[^)]*\) //' | sort | uniq -c

log "=== smoke test, every request once per DP rank"
if ! PYTHONPATH=$SMOKE_PYTHONPATH python3 "$DEV_DIR/kimi_k3_smoke_client.py" run --port "$PORT" --model "$MODEL" \
    --dataset-dir "$DEV_DIR/gsm8k-datasets" --dp-size "$DP_SIZE" --out "$RESULTS_DIR/requests.jsonl"; then
    log "ERROR: smoke client failed"
fi
: > "$RESULTS_DIR/no_connector_log.txt"
PYTHONPATH=$SMOKE_PYTHONPATH python3 "$DEV_DIR/kimi_k3_smoke_client.py" report --requests "$RESULTS_DIR/requests.jsonl" \
    --connector-log "$RESULTS_DIR/no_connector_log.txt" --block-size "$LMCACHE_CHUNK_SIZE" \
    | tee "$RESULTS_DIR/report.txt"

log "=== capacity: 32k x 6 and 128k x 1 per DP rank"
SERVE_STATE=$SERVE_STATE SKIP_SMOKE=1 CASES="32000:6 128000:1" bash "$DEV_DIR/kimi_k3_capacity.sh"

for pattern in "Traceback" " ERROR " "OutOfMemoryError"; do
    log "serve log \"$pattern\": $(tr '\r' '\n' < "$SERVE_LOG" | grep -ac "$pattern")"
    log "lmcache server log \"$pattern\": $(grep -ac "$pattern" "$LMCACHE_SERVER_LOG")"
done
log "vLLM (process group $vllm_pgid) and lmcache server (process group $server_pgid) left running"
log "ALL_DONE $RESULTS_DIR"
