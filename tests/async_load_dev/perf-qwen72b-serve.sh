#!/bin/bash -e
# usage: perf-qwen72b-serve.sh <scheme|sync> <serve_log>   (dev helper)
# Start our port-8010 vLLM for the Qwen2.5-72B perf test inside container $C:
# TP=8 on GPU0-7 (ranks 0-3 on NUMA0, 4-7 on NUMA1), KV compression with QAT + IAA
# (CPU zip off), DSA+GDRCopy transfers, async-load map from setvars.sh. With all 8
# GPUs in use, setvars.sh auto-detection (tools/auto_config.sh) picks each rank's
# CPU / QAT / DSA from its GPU's NUMA node; DSA must be configured first with
# tools/setup_dsa_cnt.sh (README). The startup log prints the chosen devices.
# Debug logs, native verbose logs, profiling and the TaskQueue trace stay off.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=${C:-iaxl.vllm.qiuyu-v29}
SCHEME=$1
# vLLM stdout/stderr goes straight here, so it survives a crash or a killed run.
SERVE_LOG=$2
if [ -z "$SCHEME" ] || [ -z "$SERVE_LOG" ]; then
    echo "usage: $0 <scheme|sync> <serve_log>" >&2
    exit 1
fi
MODEL=${MODEL:-/home/pese/model-space/Qwen2.5-72B-Instruct}
# IAXL_DDR_POOL_SIZE_GB is per rank (one KVStore per TP worker): 8 x 28 = 224 GB.
# Together with the 192 GB scratch pool this pins 416 of the host's 503 GB; the
# rest is for the OS, vLLM processes and QAT's contiguous 2 MB pages (the host is
# not shared during these runs).
DDR_POOL_GB_PER_RANK=${DDR_POOL_GB_PER_RANK:-28}
# Pinned staging pool, also per rank: one 32k-token load of 72B at TP=8 takes
# 80 layers x 2000 blocks x 8 KiB = 1.22 GiB, so 16 loads in flight plus the
# stores need ~20+ GiB per rank. 8 x 24 = 192 GB.
SCRATCH_POOL_GB_PER_RANK=${SCRATCH_POOL_GB_PER_RANK:-24}
# "sync": async load disabled, every DDR hit loads synchronously and no
# KVSHRINK_ASYNC_LOAD_SCHEME is passed; anything else is an async scheme name.
if [ "$SCHEME" = "sync" ]; then
    LOAD_ENV="KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0"
else
    LOAD_ENV="KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=1 KVSHRINK_ASYNC_LOAD_SCHEME=$SCHEME"
    # DYNAMIC_MAP / ASYNC_LOAD_LAYERS (set by perf-qwen72b-bench.sh): the
    # async-load layer map, or a fixed layer count run with
    # KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0; at most one of them. Both
    # unset keep the setvars.sh default.
    if [ -n "$DYNAMIC_MAP" ] && [ -n "$ASYNC_LOAD_LAYERS" ]; then
        echo "DYNAMIC_MAP and ASYNC_LOAD_LAYERS are exclusive" >&2
        exit 1
    fi
    if [ -n "$DYNAMIC_MAP" ]; then
        LOAD_ENV="$LOAD_ENV KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=$DYNAMIC_MAP"
    fi
    if [ -n "$ASYNC_LOAD_LAYERS" ]; then
        LOAD_ENV="$LOAD_ENV KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=$ASYNC_LOAD_LAYERS"
    fi
fi
SERVE_ENV="MODEL=$MODEL TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    IAXL_IAA_DEVICES=1 \
    IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 \
    IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1 IAXL_DDR_POOL_SIZE_GB=$DDR_POOL_GB_PER_RANK \
    IAXL_SCRATCH_POOL_SIZE_GB=$SCRATCH_POOL_GB_PER_RANK \
    PORT=8010 IAXL_API_CONTROLLER_PORT=18710 IAXL_API_WORKER_BASE_PORT=18810 \
    IAXL_DEBUG=0 IAXL_DEBUG_LOG=0 IAXL_PROFILE_MODE=disabled \
    KVSHRINK_ASYNC_LOAD_EVENT_LOG=${ASYNC_LOAD_EVENT_LOG:-0} \
    $LOAD_ENV"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] stop port-8010 vLLM, then in $C: export $SERVE_ENV; exec ./examples/kvshrink-vllm-serve.sh > $SERVE_LOG"
    exit 0
fi

# Stop our previous port-8010 server(s) and any `vllm bench serve` client of it:
# an interrupted bench leaves its client running in the container, and the
# server's graceful shutdown then waits for the client's connections forever.
# Then wait for the server's tee to exit (see restart-vllm.sh for why).
P=$(ps -eo pid,args | grep -E "[v]llm (serve|bench serve) .*--port 8010" | awk '{print $1}')
if [ -n "$P" ]; then
    docker exec $C kill -TERM $P
fi
for i in $(seq 1 90); do
    OLD=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
    OLD_TEE=$(docker exec $C bash -c "for T in \$(pgrep -f '^tee log.kvshrink-vllm\$'); do [ \"\$(readlink /proc/\$T/cwd)\" = '$REPO' ] && echo \$T; done" | tr '\n' ' ')
    [ -z "$OLD" ] && [ -z "${OLD_TEE// /}" ] && break
    sleep 2
done

docker exec -w "$REPO" -d $C bash -c "export $SERVE_ENV; \
    exec ./examples/kvshrink-vllm-serve.sh > '$SERVE_LOG' 2>&1"
for i in $(seq 1 240); do curl -sf http://localhost:8010/health > /dev/null && break; sleep 5; done
curl -s -o /dev/null -w 'health=%{http_code}\n' http://localhost:8010/health
