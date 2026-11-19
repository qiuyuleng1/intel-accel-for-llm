#!/bin/bash -e
# LMCache MP mode for kvshrink-benchmark.sh: a standalone `lmcache server` plus
# `vllm serve` with LMCacheMPConnector. The vLLM flags match
# kvshrink-vllm-serve.sh except the connector. Prefix caching stays off: the
# image's lmcache_hybrid_patch (port of PR #41) keeps mamba_cache_mode='align'
# for hybrid models such as Qwen3.8.
#
# Runs in qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7 (vLLM 0.29.0 +
# LMCache 0.5.7), not in the KVShrink container, so setvars.sh is not sourced.
# Both processes share this script's process group; stopping the group stops
# the LMCache server as well.
# BASELINE=prefix_cache runs plain vLLM of this image with its GPU prefix cache
# and no LMCache server: the reference for lmcache on the same vLLM version.
: "${MODEL:?set MODEL}"
: "${TP_SIZE:?set TP_SIZE}"
: "${LMCACHE_CHUNK_SIZE:?set LMCACHE_CHUNK_SIZE to the vLLM block size}"
: "${LMCACHE_L1_SIZE_GB:?set LMCACHE_L1_SIZE_GB}"
: "${LMCACHE_SERVER_LOG:?set LMCACHE_SERVER_LOG}"
LMCACHE_PORT=${LMCACHE_PORT:-5560}
LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT:-8160}
LMCACHE_READY_TIMEOUT=300

# Unset BLOCK_SIZE keeps vLLM's default, as in kvshrink-vllm-serve.sh.
block_size_args=()
if [ -n "${BLOCK_SIZE:-}" ]; then
    block_size_args=(--block-size "$BLOCK_SIZE")
fi
# TORCH_PROFILER_DIR enables vLLM's torch profiler, as in kvshrink-vllm-serve.sh.
profiler_args=()
if [ -n "${TORCH_PROFILER_DIR:-}" ]; then
    profiler_args=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TORCH_PROFILER_DIR\", \"torch_profiler_with_stack\": false, \"max_iterations\": ${PROFILER_MAX_ITERATIONS:-20}}")
fi
# Unset ATTENTION_BACKEND keeps vLLM's choice, as in kvshrink-vllm-serve.sh.
attention_args=()
if [ -n "${ATTENTION_BACKEND:-}" ]; then
    attention_args=(--attention-backend "$ATTENTION_BACKEND")
fi
# More `vllm serve` options, as in kvshrink-vllm-serve.sh: split on whitespace.
read -r -a extra_args <<<"${VLLM_SERVE_EXTRA_ARGS:-}"

if [ "${BASELINE:-}" = "prefix_cache" ]; then
    exec vllm serve "$MODEL" \
        --enable-prefix-caching \
        --trust-remote-code \
        -tp "$TP_SIZE" \
        --gpu-memory-utilization 0.8 \
        "${block_size_args[@]}" \
        "${profiler_args[@]}" \
        "${attention_args[@]}" \
        --max-model-len "${MAX_MODEL_LEN:-32765}" \
        --port "${PORT:-8000}" \
        "${extra_args[@]}"
fi

# The `lmcache` command is a Python script, so ps would show the server as
# /usr/bin/python3.12; name it like vLLM's VLLM::Worker processes instead.
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
    -tp "$TP_SIZE" \
    --gpu-memory-utilization 0.8 \
    "${block_size_args[@]}" \
    "${profiler_args[@]}" \
    "${attention_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --port "${PORT:-8000}" \
    "${extra_args[@]}"
