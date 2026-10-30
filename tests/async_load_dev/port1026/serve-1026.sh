#!/bin/bash -e
# usage: serve-1026.sh <vllm_log> [VAR=value ...]   (dev helper, run on the host)
# (Re)start vLLM on port 8010 in the 1026 container with the 1026 PR branch: Qwen3-32B,
# TP=2 on the first two GPUs (setvars.sh picks each rank's CPUs / QAT from their NUMA
# node), settings from 1026/setvars.sh, plus the VAR=value overrides. Same arguments as
# 1026/examples/kvshrink-vllm-serve.sh, except that the connector module is
# traced_kvshrink_connector (the PR connector plus one log line per raise) and the
# log goes to <vllm_log>. Waits until /health answers.
# MODEL=<path> picks another model (default Qwen3-32B); KV_CONNECTOR=0 starts plain vLLM
# without the KV connector (baseline).
REPO=/home/pese/qiuyu/intel-accel-for-llm-1026
HERE=$(cd "$(dirname "$0")" && pwd)
C=${C:-iaxl.vllm.qiuyu-1026}
VLLM_LOG=$1
shift
if [ -z "$VLLM_LOG" ]; then
    echo "usage: $0 <vllm_log> [VAR=value ...]" >&2
    exit 1
fi
# The server runs with the 1026 repo as working directory, so a relative path would point there.
VLLM_LOG=$(realpath -m "$VLLM_LOG")
MODEL=${MODEL:-/media/model-space/Qwen3-32B}
KV_ARGS="--kv-transfer-config '{\"kv_connector\":\"KVShrinkConnector\",\"kv_connector_module_path\":\"traced_kvshrink_connector\",\"kv_role\":\"kv_both\"}'"
if [ "${KV_CONNECTOR:-1}" = 0 ]; then
    KV_ARGS=""
fi

# Stop our previous port-8010 server (by PID, never by pattern inside the container).
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
if [ -n "$P" ]; then
    docker exec $C kill -TERM $P
fi
for i in $(seq 1 60); do
    P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
    [ -z "$P" ] && break
    sleep 2
done

# QAT's usdm driver allocates 4 MB physically contiguous slot buffers when the first
# compression starts; with the page cache filling the host, that failed on 10-26
# ("page allocation failure: order:10"). Same reclaim as perf-qwen72b-bench.sh: drop
# clean page cache and compact both NUMA nodes (root in the container).
docker exec $C bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /sys/devices/system/node/node0/compact; echo 1 > /sys/devices/system/node/node1/compact"
echo "reclaimed host memory: $(free -g | awk '/^Mem:/ {print "free=" $4 "G"}')" \
     "order-10 blocks: $(grep Normal /proc/buddyinfo | awk '{print $NF}' | tr '\n' ' ')"

touch "$VLLM_LOG"
docker exec -w "$REPO" -d $C bash -c "exec > $VLLM_LOG 2>&1; \
    export MODEL=$MODEL TP_SIZE=2 PORT=8010 \
    IAXL_API_CONTROLLER_PORT=18710 IAXL_API_WORKER_BASE_PORT=18810 $*; \
    source setvars.sh; \
    export LD_PRELOAD=\"\$IAXL_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
    export PYTHONPATH=$HERE\${PYTHONPATH:+:\$PYTHONPATH}; \
    exec vllm serve $MODEL $KV_ARGS \
        --trust-remote-code -tp 2 --gpu-memory-utilization 0.8 --max-model-len 32765 \
        --port 8010 --no-enable-prefix-caching"
for i in $(seq 1 120); do
    curl -sf http://localhost:8010/health > /dev/null && break
    # Fail fast instead of waiting out the loop when vLLM exits during startup.
    if [ "$i" -gt 2 ] && ! ps -eo args | grep -qE "[v]llm serve .*--port 8010"; then
        echo "vLLM exited during startup; last error lines of $VLLM_LOG:" >&2
        grep -aE "Error|error" "$VLLM_LOG" | sed -E 's/^.*\] //' | tail -3 >&2
        exit 1
    fi
    sleep 5
done
curl -s -o /dev/null -w 'health=%{http_code}\n' http://localhost:8010/health
grep -aE "KVSHRINK_VLLM_KV_ASYNC_LOAD|VLLM_CPU_OMP_THREADS_BIND|KVSHRINK_QAT|TP_SIZE|Registered|KVShrinkConnector\(|KVShrink patch|block size|Maximum concurrency|KV cache size" \
    "$VLLM_LOG" | sed -E 's/^.*\] //' | head -20
curl -sf http://localhost:8010/health > /dev/null
