#!/bin/bash
# Qwen3.8-27B TP=4 (GPU0-3) kvshrink (no compression, DSA) benchmark, run twice with kvshrink-benchmark.sh:
# IAXL_DSA_AUTO_FALLBACK_ENABLE=1, then =0. Input 32768 tokens, concurrency 32, hit rate 100 only,
# 320 requests per run, no prewarm (at hit 100 every request is the same prompt, so the 5 warm-up
# requests store it), pools as the 11-09 6000d run (DDR 1000 GB total, scratch 8 GB per rank).
# GPU0-3 = the first 4 GPUs nvidia-smi lists, which setvars.sh binds CPUs / QAT / DSA to.
# Usage: bash tests/dsa_fallback_dev/bench_fallback.sh   (DRY_RUN=1 prints the plan, starts nothing)
#        FALLBACK_ORDER="1 0 1 0 1 0" runs one benchmark per value in that order.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTAINER=${CONTAINER:-iaxl.vllm}
DRY_RUN=${DRY_RUN:-0}
# Alternating the values spreads any drift of the machine over both settings.
FALLBACK_ORDER=${FALLBACK_ORDER:-"1 0"}
STAMP=$(date -u +%Y%m%d_%H%M%S)
GPU_BUSY_MIB=500
# Each benchmark gets 20 minutes; on timeout kvshrink-benchmark.sh's TERM trap stops its vLLM.
BENCH_TIMEOUT_S=1200

run_number=0
for fallback in $FALLBACK_ORDER; do
    run_number=$((run_number + 1))
    results_dir=bench-results/Qwen3.8-27B_tp4_nozip_hit100_xfermetrics_run${run_number}_dsa-fallback${fallback}_$STAMP
    if [[ "$DRY_RUN" != 1 ]]; then
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk -F', ' -v limit=$GPU_BUSY_MIB '$2 > limit {print "GPU" $1 ":" $2 "MiB"}')
        if [[ -n "$busy" ]]; then
            echo "ERROR: GPUs busy before IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback: $busy"
            exit 1
        fi
    fi
    echo "=== IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback -> $results_dir ($(date -u '+%H:%M:%S') UTC)"
    docker exec -w "$REPO_DIR" \
        -e MODEL=/media/model-space/Qwen3.8-27B -e TP_SIZE=4 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
        -e CONFIGS=kvshrink -e LEN_CONCURRENCY=32768:32 -e HIT_RATES=100 -e PREWARM=0 \
        -e IAXL_KV_COMPRESSION=0 -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=0 \
        -e IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 -e IAXL_SCRATCH_POOL_SIZE_GB=8 \
        -e IAXL_DSA_AUTO_FALLBACK_ENABLE="$fallback" -e RESULTS_DIR="$results_dir" -e DRY_RUN="$DRY_RUN" \
        "$CONTAINER" timeout "$BENCH_TIMEOUT_S" bash kvshrink-benchmark.sh
    exit_code=$?
    echo "=== IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback exit $exit_code ($(date -u '+%H:%M:%S') UTC)"
    if ((exit_code != 0)); then
        exit "$exit_code"
    fi
done
