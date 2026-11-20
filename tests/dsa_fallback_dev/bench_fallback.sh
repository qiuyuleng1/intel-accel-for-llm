#!/bin/bash
# Qwen3.8-27B TP=8 kvshrink (QAT + IAA compression, DSA) benchmark, run twice with kvshrink-benchmark.sh:
# IAXL_DSA_AUTO_FALLBACK_ENABLE=1, then =0. Input 32768 tokens, concurrency 64, hit rate 80 and 100,
# 640 requests per run, same pools as the 11-09 6000d run (DDR 1000 GB total, scratch 8 GB per rank).
# Usage: bash tests/dsa_fallback_dev/bench_fallback.sh   (DRY_RUN=1 prints the plan, starts nothing)
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTAINER=${CONTAINER:-iaxl.vllm}
DRY_RUN=${DRY_RUN:-0}
STAMP=$(date -u +%Y%m%d_%H%M%S)
GPU_BUSY_MIB=500
# Each benchmark gets 2 hours; on timeout kvshrink-benchmark.sh's TERM trap stops its vLLM.
BENCH_TIMEOUT_S=7200

for fallback in 1 0; do
    results_dir=bench-results/Qwen3.8-27B_tp8_dsa-fallback${fallback}_$STAMP
    if [[ "$DRY_RUN" != 1 ]]; then
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v limit=$GPU_BUSY_MIB '$2 > limit {print "GPU" $1 ":" $2 "MiB"}')
        if [[ -n "$busy" ]]; then
            echo "ERROR: GPUs busy before IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback: $busy"
            exit 1
        fi
    fi
    echo "=== IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback -> $results_dir ($(date -u '+%H:%M:%S') UTC)"
    docker exec -w "$REPO_DIR" \
        -e MODEL=/media/model-space/Qwen3.8-27B -e TP_SIZE=8 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e CONFIGS=kvshrink -e LEN_CONCURRENCY=32768:64 -e HIT_RATES="80 100" \
        -e IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 -e IAXL_SCRATCH_POOL_SIZE_GB=8 \
        -e IAXL_DSA_AUTO_FALLBACK_ENABLE="$fallback" -e RESULTS_DIR="$results_dir" -e DRY_RUN="$DRY_RUN" \
        "$CONTAINER" timeout "$BENCH_TIMEOUT_S" bash kvshrink-benchmark.sh
    exit_code=$?
    echo "=== IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback exit $exit_code ($(date -u '+%H:%M:%S') UTC)"
    if ((exit_code != 0)); then
        exit "$exit_code"
    fi
done
