#!/bin/bash -e
# usage: perf-qwen72b-serve.sh <scheme> <serve_log>   (dev helper)
# Start our port-8010 vLLM for the Qwen2.5-72B perf test inside container $C:
# TP=4 on GPU4-7 (NUMA1), KV compression with QAT + IAA (CPU zip off), DSA+GDRCopy
# transfers, async-load map from setvars.sh. setvars.sh auto-detection assigns
# CPU / QAT / DSA by the first TP_SIZE GPUs of nvidia-smi (GPU0-3, NUMA0) and does
# not honor CUDA_VISIBLE_DEVICES, so all three are pinned to NUMA1 here.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
C=${C:-iaxl.vllm.qiuyu-v29}
SCHEME=$1
# vLLM stdout/stderr goes straight here, so it survives a crash or a killed run.
SERVE_LOG=$2
if [ -z "$SCHEME" ] || [ -z "$SERVE_LOG" ]; then
    echo "usage: $0 <scheme> <serve_log>" >&2
    exit 1
fi
MODEL=${MODEL:-/home/pese/model-space/Qwen2.5-72B-Instruct}
# One NUMA1 core range (+ hyperthreads) per rank.
CPU_BIND="64-79,192-207|80-95,208-223|96-111,224-239|112-127,240-255"
# Indices into the PCI-sorted QAT list: 4-7 are 0000:81/86/8b/90 (NUMA1).
QAT_DEVICES="4|5|6|7"
# Enabled by dsa-numa1.sh.
DSA_WQS="wq8.0|wq10.0|wq12.0|wq14.0"
# IAXL_DDR_POOL_SIZE_GB is per rank (one KVStore per TP worker): 4 x 25 = 100 GB.
DDR_POOL_GB_PER_RANK=${DDR_POOL_GB_PER_RANK:-25}
# Pinned staging pool, also per rank: one 8k-token load or store of 72B takes
# 80 layers x 500 blocks x 16 KiB = 0.61 GiB, so 12 loads + 12 stores in flight
# need ~14.6 GiB. 4 x 32 GB (for 32k inputs) exhausted host memory: QAT init then
# hung in direct reclaim. 4 x 16 = 64 GB.
SCRATCH_POOL_GB_PER_RANK=${SCRATCH_POOL_GB_PER_RANK:-16}
SERVE_ENV="MODEL=$MODEL TP_SIZE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 \
    VLLM_CPU_OMP_THREADS_BIND='$CPU_BIND' KVSHRINK_QAT_DEVICES='$QAT_DEVICES' \
    KVSHRINK_DSA_DEVICES='$DSA_WQS' IAXL_IAA_DEVICES=1 \
    IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 \
    IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1 IAXL_DDR_POOL_SIZE_GB=$DDR_POOL_GB_PER_RANK \
    IAXL_SCRATCH_POOL_SIZE_GB=$SCRATCH_POOL_GB_PER_RANK \
    PORT=8010 IAXL_API_CONTROLLER_PORT=18710 IAXL_API_WORKER_BASE_PORT=18810 \
    KVSHRINK_ASYNC_LOAD_SCHEME=$SCHEME"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] stop port-8010 vLLM, then in $C: export $SERVE_ENV; exec ./examples/kvshrink-vllm-serve.sh > $SERVE_LOG"
    exit 0
fi

# Stop our previous port-8010 server and wait for its tee to exit (see
# restart-vllm.sh for why).
P=$(ps -eo pid,args | grep -E "[v]llm serve .*--port 8010" | awk '{print $1}')
if [ -n "$P" ]; then
    docker exec $C kill -TERM "$P"
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
