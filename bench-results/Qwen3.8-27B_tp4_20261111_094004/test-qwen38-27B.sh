#!/bin/bash
# Qwen3.8-27B on b300 at TP=4 on GPU0, GPU2, GPU4 and GPU6 (one GPU per NUMA
# node): kvshrink, vllm-prefix, lmcache and vllm-wo-prefix.
# Input 32000 tokens at concurrency 32 64 128 256 300, 128000 tokens at
# concurrency 64 80 and 256000 tokens at concurrency 32 40, output 128, hit
# rates 80 95 (vllm-wo-prefix: 80 only), concurrency * 10 requests per run
# (vllm-wo-prefix: * 2).
#
# tests/bench_dev/probe-max-concurrency.sh on b300 at TP=4 measured a GPU KV
# cache of 11.3M-12.6M tokens: vLLM's maximum concurrency is 342 at 32896
# tokens, 94 at 131200 and 48 at 262144. The block size is not set: vLLM 0.29
# picks 784 for this hybrid model at TP=4 (400 at TP=8), and lmcache's chunk
# size must equal it.
#
# Runs on the HOST. Every config is one kvshrink-benchmark.sh call:
#   kvshrink, vllm-prefix, vllm-wo-prefix: docker exec into the KVShrink
#       container. If it does not exist, ./start.sh starts it in a detached tmux
#       session and this script waits until `pip install -e .` finished.
#   lmcache: a new container of the LMCache image (docker run --rm).
# Before anything starts, the host settings of PERFORMANCE_MODE_SETUP.md
# (performance mode and ACS off) are checked; before every config the page
# cache is dropped and memory compacted. A failed config is reported and the
# next one still runs.
#
# Results: bench-results/Qwen3.8-27B_tp4_<time>/ (summary.csv, run.log and
# per-config serve logs).
# Print the plan without starting any serve or container:
#   DRY_RUN=1 bash test-qwen38-27B.sh
# Only some configs (same order as listed): CONFIGS=kvshrink bash test-qwen38-27B.sh
# Other hit rates (default 80 95): HIT_RATES=100 bash test-qwen38-27B.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Qwen3.8-27B
# vllm-wo-prefix last: it is the longest and the only one that reuses nothing.
CONFIGS=${CONFIGS:-"kvshrink vllm-prefix lmcache vllm-wo-prefix"}
HIT_RATES=${HIT_RATES:-"80 95"}
DRY_RUN=${DRY_RUN:-0}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
LMCACHE_IMAGE=qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7
LMCACHE_CONTAINER=iaxl.lmcache-bench
MODEL_TAG=$(basename "$MODEL")
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-${MODEL_TAG}_tp4_$(date +%Y%m%d_%H%M%S)
else
    RESULTS_ROOT=bench-results/${MODEL_TAG}_tp4_$(date +%Y%m%d_%H%M%S)
fi
# b300 has 4 QAT, 4 DSA and 4 IAA devices, one per NUMA node, and two GPUs per
# NUMA node. Using GPU0/2/4/6 gives every rank its own NUMA node: rank r runs
# on GPU 2r and owns QAT r, DSA queue wq(2r).0, the IAA of NUMA node r and the
# 24 physical cores of NUMA node r (no hyperthread siblings). setvars.sh cannot
# find this binding itself: it binds the ranks to the first TP_SIZE GPUs
# nvidia-smi lists (GPU0-3, two per NUMA node) whatever CUDA_VISIBLE_DEVICES says.
ACCEL_ENV="TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6"
ACCEL_ENV+=" VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'"
ACCEL_ENV+=" KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'"
# IAA by NUMA node; unset, every rank would use the IAA of all four nodes.
ACCEL_ENV+=" KVSHRINK_IAA_DEVICES='0|1|2|3'"
# start.sh sources setvars.sh, which fails at its default TP_SIZE=8 (only 4
# QAT devices), so it gets the same binding. The container still sees all 8
# GPUs: start.sh always runs it with --gpus all.
KVSHRINK_START_ENV=$ACCEL_ENV
# Shared by every kvshrink-benchmark.sh call; everything else (every layer
# compressed, 5 warm-ups, the prewarm) keeps kvshrink-benchmark.sh's defaults.
# The KVShrink DDR pool (all ranks together) and the LMCache CPU cache are
# 1000 GB as on h8263, so that fewer blocks are evicted: the DDR pool is
# 250 GB per rank at TP=4. ScratchPool is 8 GB per
# rank (setvars.sh's default). Host memory is 2015 GB, four NUMA nodes of
# ~504 GB, one rank on each: 250 + 8 GB of pools per node.
LEN_CONCURRENCY=${LEN_CONCURRENCY:-"32000:32,64,128,256,300 128000:64,80 256000:32,40"}
BENCH_ENV="MODEL=$MODEL $ACCEL_ENV"
BENCH_ENV+=" LEN_CONCURRENCY='$LEN_CONCURRENCY' HIT_RATES='$HIT_RATES'"
# 256000 input + 128 output tokens. vLLM 0.29 picks block size 784 at TP=4
# (400 at TP=8), and the LMCache chunk size must equal it.
BENCH_ENV+=" MAX_MODEL_LEN=256128 LMCACHE_CHUNK_SIZE=784"
BENCH_ENV+=" IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 LMCACHE_L1_SIZE_GB=1000"
# Requests per run by concurrency, used by the smoke test.
if [ -n "${NUM_PROMPTS_BY_CONCURRENCY:-}" ]; then
    BENCH_ENV+=" NUM_PROMPTS_BY_CONCURRENCY='$NUM_PROMPTS_BY_CONCURRENCY'"
fi
BENCH_ENV+=" DRY_RUN=$DRY_RUN"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen38-27B.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

# One kvshrink-benchmark.sh call for one config.
run_config() {
    local config=$1
    local bench_cmd="CONFIGS=$config RESULTS_DIR=$RESULTS_ROOT $BENCH_ENV bash kvshrink-benchmark.sh"
    if [ "$config" = "lmcache" ]; then
        docker run --rm --name "$LMCACHE_CONTAINER" --gpus all --network host --ipc host \
            -v "$REPO_ROOT:$REPO_ROOT" -v /home/intel/model-space:/home/intel/model-space -w "$REPO_ROOT" \
            --entrypoint bash "$LMCACHE_IMAGE" -c "$bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Qwen3.8-27B test  model=$MODEL  configs=[$CONFIGS]"
echo "  $BENCH_ENV"
echo "  KVShrink container=$KVSHRINK_CONTAINER  LMCache image=$LMCACHE_IMAGE"
echo "  results -> $RESULTS_ROOT"
echo "==========================================================="

# Fail before starting anything when the host is not in performance mode.
echo "Host settings (PERFORMANCE_MODE_SETUP.md):"
if ! check_host_settings && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: host settings differ from PERFORMANCE_MODE_SETUP.md; restore them first" >&2
    exit 1
fi

# The KVShrink container is a precondition: fail now, while someone is watching.
if [ "$DRY_RUN" != "1" ] && ! ensure_kvshrink_container; then
    exit 1
fi

failed=""
for config in $CONFIGS; do
    echo ""
    echo "### $config ($(date +'%F %T')) -> $RESULTS_ROOT"
    if [ "$DRY_RUN" != "1" ]; then
        reclaim_host_memory
    fi
    run_config "$config"
    status=$?
    if [ "$status" != "0" ]; then
        echo "ERROR: $config exited with status $status" >&2
        failed+=" $config"
    fi
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
