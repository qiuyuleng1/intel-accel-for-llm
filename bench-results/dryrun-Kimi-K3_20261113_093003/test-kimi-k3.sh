#!/bin/bash
# Kimi-K3 on b300 (8 x B300): kvshrink, vllm-prefix, vllm-wo-prefix and lmcache in two layouts:
#   dp2tp4  DP2 TP4 with expert parallel (--enable-expert-parallel), the layout found on 11-13
#   tp8     TP8
# Input 32000 tokens at concurrency 12 and 128000 tokens at concurrency 2 (the 11-13 capacity
# test: DP2 TP4 ran 6 + 6 and 1 + 1 per DP rank without preemption), output 128, hit rates
# 80 95 (vllm-wo-prefix: 80 only), concurrency * 10 requests per run (vllm-wo-prefix: * 2).
# With DP the 5 warm-ups of every run go to each DP rank (kvshrink-benchmark.sh).
#
# vLLM options in both layouts: --gpu-memory-utilization 0.9, --max-num-seqs 128 (needed by
# DP2 TP4: the default 1024 ran out of memory during profiling), MAX_MODEL_LEN=131200
# (128000 + 128 output tokens fit), plus --enable-expert-parallel in dp2tp4. KVShrink: QAT +
# IAA compression and DSA transfers, the 2 GPUs of a NUMA node share its devices (setvars.sh;
# needs the DSA set to 2 WQs per device, tools/setup_dsa_cnt.sh 2). KVShrink DDR pool and
# LMCache CPU cache 600 GB each (~16-36 GB per rank expected for this matrix, so nothing is
# evicted, and LMCache pins its cache faster than 1200 GB); scratch pool 8 GB per rank.
# LMCache chunk size = vLLM block size: 1536 at TP4, 768 at TP8.
#
# Runs on the HOST. Every (layout, config) is one kvshrink-benchmark.sh call:
#   kvshrink, vllm-prefix, vllm-wo-prefix: docker exec into the KVShrink container.
#   lmcache: docker exec into lmcache.vllm-kimi-k3-qiuyu (a container of the LMCache image
#       created with --init and kept, so the serve left running survives the call).
# Order: dp2tp4 (kvshrink, vllm-prefix, vllm-wo-prefix, lmcache), then tp8 (same). A serve is
# stopped only to start the next one; the last one (tp8 lmcache) is left running.
# A failed config is reported and the next one still runs.
#
# Results: bench-results/Kimi-K3_<time>/<layout>/ (summary.csv, run.log, per-config serve logs).
# Print the plan without starting anything: DRY_RUN=1 bash test-kimi-k3.sh
# Subsets: LAYOUTS="tp8" CONFIGS="kvshrink lmcache" bash test-kimi-k3.sh; HIT_RATES=95 ...
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Kimi-K3
LAYOUTS=${LAYOUTS:-"dp2tp4 tp8"}
CONFIGS=${CONFIGS:-"kvshrink vllm-prefix vllm-wo-prefix lmcache"}
HIT_RATES=${HIT_RATES:-"80 95"}
LEN_CONCURRENCY=${LEN_CONCURRENCY:-"32000:12 128000:2"}
CACHE_SIZE_GB=${CACHE_SIZE_GB:-600}
DRY_RUN=${DRY_RUN:-0}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=\${CONTAINER_NAME:-\(.*\)}$/\1/p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
KVSHRINK_START_ENV=""
LMCACHE_CONTAINER=lmcache.vllm-kimi-k3-qiuyu
PORT=8000
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=bench-results/dryrun-Kimi-K3_$(date +%Y%m%d_%H%M%S)
else
    RESULTS_ROOT=bench-results/Kimi-K3_$(date +%Y%m%d_%H%M%S)
fi

# "<DP_SIZE> <TP_SIZE> <LMCACHE_CHUNK_SIZE> <extra vLLM args>" of a layout.
layout_settings() {
    case "$1" in
        dp2tp4) echo "2 4 1536 --gpu-memory-utilization 0.9 --max-num-seqs 128 --enable-expert-parallel" ;;
        tp8) echo "1 8 768 --gpu-memory-utilization 0.9 --max-num-seqs 128" ;;
        *) echo "Unknown layout: $1 (expected dp2tp4 or tp8)" >&2; return 1 ;;
    esac
}

for layout in $LAYOUTS; do
    layout_settings "$layout" > /dev/null || exit 1
done

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-kimi-k3.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

# The (layout, config) pair whose serve stays running.
last_layout=${LAYOUTS##* }
last_config=${CONFIGS##* }

# One kvshrink-benchmark.sh call for one layout and config.
run_config() {
    local layout=$1 config=$2
    local dp_size tp_size chunk_size extra_args keep_serve=0
    read -r dp_size tp_size chunk_size extra_args <<< "$(layout_settings "$layout")"
    if [ "$layout" = "$last_layout" ] && [ "$config" = "$last_config" ]; then
        keep_serve=1
    fi
    local bench_env="MODEL=$MODEL DP_SIZE=$dp_size TP_SIZE=$tp_size CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    bench_env+=" LEN_CONCURRENCY='$LEN_CONCURRENCY' HIT_RATES='$HIT_RATES' MAX_MODEL_LEN=131200"
    bench_env+=" VLLM_SERVE_EXTRA_ARGS='$extra_args' LMCACHE_CHUNK_SIZE=$chunk_size"
    bench_env+=" IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=$CACHE_SIZE_GB LMCACHE_L1_SIZE_GB=$CACHE_SIZE_GB"
    bench_env+=" KEEP_SERVE=$keep_serve DRY_RUN=$DRY_RUN"
    local bench_cmd="CONFIGS=$config RESULTS_DIR=$RESULTS_ROOT/$layout $bench_env bash kvshrink-benchmark.sh"
    echo "  $bench_cmd"
    if [ "$config" = "lmcache" ]; then
        docker exec -w "$REPO_ROOT" "$LMCACHE_CONTAINER" bash -lc "$bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Kimi-K3 test  model=$MODEL  layouts=[$LAYOUTS]  configs=[$CONFIGS]"
echo "  input:concurrency=[$LEN_CONCURRENCY] hit rates=[$HIT_RATES] cache=${CACHE_SIZE_GB} GB"
echo "  KVShrink container=$KVSHRINK_CONTAINER  LMCache container=$LMCACHE_CONTAINER"
echo "  left running at the end: $last_layout $last_config"
echo "  results -> $RESULTS_ROOT"
echo "==========================================================="

# Fail before starting anything when the host is not in performance mode.
echo "Host settings (PERFORMANCE_MODE_SETUP.md):"
if ! check_host_settings && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: host settings differ from PERFORMANCE_MODE_SETUP.md; restore them first" >&2
    exit 1
fi

if [ "$DRY_RUN" != "1" ]; then
    # 8 workers share 4 DSA devices, one dedicated WQ each.
    if [ "$(cat /sys/bus/dsa/devices/wq0.1/state 2> /dev/null)" != "enabled" ]; then
        echo "ERROR: DSA has 1 WQ per device; run tools/setup_dsa_cnt.sh 2 first" >&2
        exit 1
    fi
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "ERROR: a vLLM already serves on port $PORT; stop it first" >&2
        exit 1
    fi
    if ! ensure_kvshrink_container; then
        exit 1
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$LMCACHE_CONTAINER" 2> /dev/null)" != "true" ]; then
        echo "ERROR: $LMCACHE_CONTAINER is not running; create it with:" >&2
        echo "  docker run -d --init --name $LMCACHE_CONTAINER --gpus all --ipc=host --network host --ulimit memlock=-1:-1 -v $MODEL:$MODEL:ro -v $REPO_ROOT:$REPO_ROOT -w $REPO_ROOT --entrypoint sleep qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7 infinity" >&2
        exit 1
    fi
fi

failed=""
for layout in $LAYOUTS; do
    for config in $CONFIGS; do
        echo ""
        echo "### $layout $config ($(date +'%F %T')) -> $RESULTS_ROOT/$layout"
        run_config "$layout" "$config"
        status=$?
        if [ "$status" != "0" ]; then
            echo "ERROR: $layout $config exited with status $status" >&2
            failed+=" $layout/$config"
        fi
    done
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/<layout>/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
