#!/bin/bash -e
# LMCache MP mode for kvshrink-benchmark.sh: a standalone `lmcache server` plus
# `vllm serve` with LMCacheMPConnector. The vLLM options match
# kvshrink-vllm-serve.sh except the connector: prefix caching off, -tp TP_SIZE,
# --data-parallel-size DP_SIZE when DP_SIZE > 1, then VLLM_SERVE_EXTRA_ARGS.
# With prefix caching off the image's lmcache_hybrid_patch (port of PR #41)
# switches the Kimi-K3 KDA layers to mamba_cache_mode='align'.
#
# Runs in qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7 (vLLM 0.29.0 +
# LMCache 0.5.7), not in the KVShrink container, so setvars.sh is not sourced.
# Both processes share this script's process group; stopping the group stops
# the LMCache server as well.
: "${MODEL:?set MODEL}"
: "${TP_SIZE:?set TP_SIZE}"
: "${LMCACHE_CHUNK_SIZE:?set LMCACHE_CHUNK_SIZE to the vLLM block size}"
: "${LMCACHE_L1_SIZE_GB:?set LMCACHE_L1_SIZE_GB}"
: "${LMCACHE_SERVER_LOG:?set LMCACHE_SERVER_LOG}"
LMCACHE_PORT=${LMCACHE_PORT:-5560}
LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT:-8160}
LMCACHE_READY_TIMEOUT=600

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi
read -r -a extra_args <<<"${VLLM_SERVE_EXTRA_ARGS:-}"

python3 -c 'import sys, setproctitle; setproctitle.setproctitle("LMCache::server"); from lmcache.cli.main import main; sys.argv[0] = "lmcache"; sys.exit(main())' \
    server --host 127.0.0.1 --port "$LMCACHE_PORT" \
    --http-host 127.0.0.1 --http-port "$LMCACHE_HTTP_PORT" \
    --chunk-size "$LMCACHE_CHUNK_SIZE" --separate-object-groups \
    --l1-size-gb "$LMCACHE_L1_SIZE_GB" --eviction-policy LRU \
    > "$LMCACHE_SERVER_LOG" 2>&1 &
server_pid=$!

waited=0
until curl -sf -o /dev/null "http://127.0.0.1:$LMCACHE_HTTP_PORT/healthcheck"; do
    if ! kill -0 "$server_pid" 2> /dev/null; then
        echo "ERROR: lmcache server exited before becoming ready (see $LMCACHE_SERVER_LOG)" >&2
        exit 1
    fi
    if (( waited >= LMCACHE_READY_TIMEOUT )); then
        echo "ERROR: lmcache server not ready within ${LMCACHE_READY_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
echo "lmcache server ready after ${waited}s (log: $LMCACHE_SERVER_LOG)"

kv_transfer_config='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":'"$LMCACHE_PORT"'}}'

vllm serve "$MODEL" \
    --kv-transfer-config "$kv_transfer_config" \
    --no-enable-prefix-caching \
    --trust-remote-code \
    "${parallel_args[@]}" \
    --gpu-memory-utilization 0.8 \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    "${extra_args[@]}"
