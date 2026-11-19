#!/bin/bash
# Qwen3-32B at TP=8: kvshrink, vllm-prefix, lmcache and vllm-wo-prefix at vLLM
# block sizes 16 and 256 (LMCache chunk size = block size). Input 32000 tokens,
# output 128, concurrency 16 32 48 with 96 192 288 requests (6 per concurrency
# slot, as test-qwen25-72B.sh at 16), hit rates 80 100 (vllm-wo-prefix: 80 only).
#
# The concurrencies stay below vLLM's maximum: GPU KV cache 1.84M tokens
# (vLLM 0.29, bench-results/Qwen3-32B_tp8_asyncload_20261106_075118) fits 57
# requests of 32128 tokens.
#
# Runs on the HOST. Every config is one kvshrink-benchmark.sh call:
#   kvshrink, vllm-prefix, vllm-wo-prefix: docker exec into the KVShrink
#       container. If it does not exist, ./start.sh starts it in a detached tmux
#       session and this script waits until `pip install -e .` finished.
#   lmcache: a new container of the LMCache image (docker run --rm).
# kvshrink stores its first layer uncompressed (IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=1).
# Before anything starts, the host settings of PERFORMANCE_MODE_SETUP.md
# (performance mode and ACS off) are checked; before every config the page
# cache is dropped and memory compacted. A failed config is reported and the
# next one still runs.
#
# Results: bench-results/Qwen3-32B_tp8_<time>/bs16 and /bs256, each with its
# own summary.csv, run.log and per-config serve logs.
# Print the plan without starting any serve or container:
#   DRY_RUN=1 bash test-qwen3-32B.sh
# Only some configs (same order as listed): CONFIGS=kvshrink bash test-qwen3-32B.sh
# Other hit rates (default 80 100): HIT_RATES=95 bash test-qwen3-32B.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/media/model-space/Qwen3-32B
BLOCK_SIZES="16 256"
# vllm-wo-prefix last: it is the longest and the only one that reuses nothing.
CONFIGS=${CONFIGS:-"kvshrink vllm-prefix lmcache vllm-wo-prefix"}
HIT_RATES=${HIT_RATES:-"80 100"}
DRY_RUN=${DRY_RUN:-0}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
LMCACHE_IMAGE=qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7
LMCACHE_CONTAINER=iaxl.lmcache-bench
MODEL_TAG=$(basename "$MODEL")
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-${MODEL_TAG}_tp8_$(date +%Y%m%d_%H%M%S)
else
    RESULTS_ROOT=bench-results/${MODEL_TAG}_tp8_$(date +%Y%m%d_%H%M%S)
fi
# Shared by every kvshrink-benchmark.sh call. MAX_MODEL_LEN is the model's
# max_position_embeddings (the script's default 201128 is for Qwen3.8). The
# DDR pool, LMCache CPU cache and ScratchPool sizes are those of
# test-qwen38-27B.sh.
BENCH_ENV="MODEL=$MODEL TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_MODEL_LEN=40960"
BENCH_ENV+=" LEN_CONCURRENCY=32000:16,32,48 HIT_RATES='$HIT_RATES'"
BENCH_ENV+=" NUM_PROMPTS_BY_CONCURRENCY='16:96 32:192 48:288'"
BENCH_ENV+=" IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 LMCACHE_L1_SIZE_GB=1000"
BENCH_ENV+=" IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=1 DRY_RUN=$DRY_RUN"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen3-32B.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

# One kvshrink-benchmark.sh call for one config at one block size.
run_config() {
    local config=$1 block_size=$2 results_dir=$3
    local bench_cmd="CONFIGS=$config RESULTS_DIR=$results_dir BLOCK_SIZE=$block_size $BENCH_ENV bash kvshrink-benchmark.sh"
    if [ "$config" = "lmcache" ]; then
        docker run --rm --name "$LMCACHE_CONTAINER" --gpus all --network host --ipc host \
            -v "$REPO_ROOT:$REPO_ROOT" -v /media/model-space:/media/model-space -w "$REPO_ROOT" \
            --entrypoint bash "$LMCACHE_IMAGE" \
            -c "LMCACHE_CHUNK_SIZE=$block_size $bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Qwen3-32B test  model=$MODEL  block_sizes=[$BLOCK_SIZES]  configs=[$CONFIGS]"
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
for block_size in $BLOCK_SIZES; do
    results_dir=$RESULTS_ROOT/bs$block_size
    for config in $CONFIGS; do
        echo ""
        echo "### bs$block_size $config ($(date +'%F %T')) -> $results_dir"
        if [ "$DRY_RUN" != "1" ]; then
            reclaim_host_memory
        fi
        run_config "$config" "$block_size" "$results_dir"
        status=$?
        if [ "$status" != "0" ]; then
            echo "ERROR: bs$block_size $config exited with status $status" >&2
            failed+=" bs$block_size/$config"
        fi
    done
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/bs*/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
