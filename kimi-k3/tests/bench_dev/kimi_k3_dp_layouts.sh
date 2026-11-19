#!/bin/bash
# Kimi-K3 + kvshrink in three DP + EP layouts, one vLLM launch each, in order:
#   DP8 TP1, DP4 TP2, DP2 TP4 (all with --enable-expert-parallel, --gpu-memory-utilization 0.9).
# Every layout runs with DSA / QAT / IAA and compression on (the 2 GPUs of a NUMA node share its
# devices), DDR pool 1200 GB in total, scratch pool 8 GB per rank, and MAX_MODEL_LEN=-1 so vLLM
# fits the longest request the KV cache can hold.
# Per layout: kimi_k3_smoke.sh starts vLLM (a failed launch is recorded and the next layout runs)
# and sends every smoke request to each DP rank; kimi_k3_gsm8k.sh runs on the same vLLM; then
# vLLM is stopped only if another layout follows (the last layout's vLLM stays up). summary.txt lists each layout's result dirs and the serve-log lines on
# memory, block size, KV cache size, max model length and accelerator binding.
#
# Runs INSIDE this clone's container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 iaxl.vllm-kimi-k3-qiuyu \
#       bash -lc 'bash tests/bench_dev/kimi_k3_dp_layouts.sh'
# LAYOUTS="4:2 2:4" (DP:TP pairs) runs a subset.
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

LAYOUTS=${LAYOUTS:-"8:1 4:2 2:4"}
export IAXL_DSA_GD_ENABLE=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_KV_COMPRESSION=1
export IAXL_DDR_POOL_TOTAL_SIZE_GB=1200 IAXL_SCRATCH_POOL_SIZE_GB=8
export MAX_MODEL_LEN=-1
# vLLM's memory profiling first allocates min(max_num_seqs, 1024) KV blocks; DP block sizes are
# large (1536 at TP4), so the default 1024 blocks ran out of memory.
export VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.9 --enable-expert-parallel --max-num-seqs 128"
SERVE_STATE=$DEV_DIR/results/kimi-k3-serve.state

CODE_MD5=$(cat kvshrink/*.py iaxl/utils/affinity.py tools/auto_config.sh setvars.sh \
    "$DEV_DIR"/kimi_k3_*.sh "$DEV_DIR"/kimi_k3_*.py | md5sum | cut -c1-8)
RUN_DIR=$DEV_DIR/results/kimi-k3-dp-layouts-$(date +%Y%m%d_%H%M%S)_$CODE_MD5
SUMMARY=$RUN_DIR/summary.txt
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

stop_vllm() {
    local pgid serve_log waited=0
    if [[ ! -f "$SERVE_STATE" ]]; then
        return 0
    fi
    read -r pgid serve_log < "$SERVE_STATE"
    kill -TERM -- "-$pgid" 2> /dev/null
    while (( waited < 300 )) && kill -0 -- "-$pgid" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$pgid" 2> /dev/null
    rm -f "$SERVE_STATE"
    log "vLLM stopped after ${waited}s"
}

# The next launch needs the GPU memory of the stopped vLLM back.
wait_gpus_free() {
    local waited=0 used
    while (( waited < 300 )); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        if (( used < 2000 )); then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "WARNING: a GPU still uses ${used} MiB after ${waited}s"
}

newest_dir() {
    ls -dt "$DEV_DIR"/results/$1 2> /dev/null | head -1
}

if [[ -f "$SERVE_STATE" ]]; then
    log "ERROR: $SERVE_STATE exists; stop the running vLLM first"
    exit 1
fi
log "code md5 $CODE_MD5, layouts [$LAYOUTS], results $RUN_DIR"
last_layout=${LAYOUTS##* }
for layout in $LAYOUTS; do
    dp_size=${layout%%:*}
    tp_size=${layout##*:}
    log "=== DP$dp_size TP$tp_size: launch"
    DP_SIZE=$dp_size TP_SIZE=$tp_size bash "$DEV_DIR/kimi_k3_smoke.sh"
    smoke_status=$?
    smoke_dir=$(newest_dir "kimi-k3-2*")
    gsm8k_dir="-"
    if (( smoke_status == 0 )); then
        bash "$DEV_DIR/kimi_k3_gsm8k.sh"
        gsm8k_dir=$(newest_dir "kimi-k3-gsm8k-*")
    fi
    # Only make room for the next layout: the last layout's vLLM stays up (a launch takes long).
    if [[ "$layout" != "$last_layout" ]]; then
        stop_vllm
    fi
    {
        echo "=== DP$dp_size TP$tp_size: smoke exit $smoke_status"
        echo "smoke: $smoke_dir"
        echo "gsm8k: $gsm8k_dir"
        tr '\r' '\n' < "$smoke_dir/serve.log" | grep -aE \
            "Model loading took|GPU KV cache size|Maximum concurrency|Setting attention block size|max_model_len|Bound rank|out of memory|OutOfMemory|Error|ERROR" \
            | sed -E 's/^\(([A-Za-z_0-9 =]+)[^)]*\) //' | sort | uniq -c | sort -rn | head -60
    } >> "$SUMMARY"
    log "=== DP$dp_size TP$tp_size: smoke exit $smoke_status, smoke $smoke_dir, gsm8k $gsm8k_dir"
    if [[ "$layout" != "$last_layout" ]]; then
        wait_gpus_free
    fi
done
log "vLLM of the last layout left running; state in $SERVE_STATE"
log "ALL_DONE $RUN_DIR"
