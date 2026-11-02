#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Runs release_bench.py in the dev container (CONTAINER_NAME in setvars.sh): no GPU, no vLLM.
# Results go to tests/omp_release_dev/results/<time>_<md5 of the bench code>_blocks<N>/, log
# written live.
#
#   BLOCKS=500 ./run_release_bench.sh      # blocks per task: 500 = 8k input, 1000 = 16k, 2000 = 32k
#   ./run_release_bench.sh --rounds 3      # extra arguments go to release_bench.py
#   CPUS=16-31 ./run_release_bench.sh      # CPUs shared by the main thread and the queue thread
#   DRY_RUN=1 ./run_release_bench.sh       # print the command only
set -euo pipefail

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$BENCH_DIR/../.." && pwd)
CONTAINER_NAME=$(sed -n 's/^CONTAINER_NAME=//p' "$REPO_DIR/setvars.sh")
BLOCKS=${BLOCKS:-500}
# A real rank's vLLM worker and IAXL threads share one CPU range; with TP=8 on this host that is
# 16 CPUs per rank (setvars.sh auto-detect: rank 1 = 16-31).
CPUS=${CPUS:-16-31}
DRY_RUN=${DRY_RUN:-0}
# The scratch pool size of the real Qwen2.5-72B runs (perf-qwen72b-bench.sh: 16 GiB per rank).
POOL_GB=16
# The real runs start vLLM through examples/kvshrink-vllm-serve.sh, which preloads
# IAXL_LD_PRELOAD (tcmalloc + libiomp5); the malloc decides what the label strings cost.
PRELOAD=$(docker exec "$CONTAINER_NAME" printenv IAXL_LD_PRELOAD)
# setvars.sh default, which the serve script sources; start.sh does not pass it into the container.
TCMALLOC_RELEASE_RATE=0

CODE_MD5=$(cat "$BENCH_DIR/release_bench.py" "$BENCH_DIR/release_bench_ext.cpp" | md5sum | cut -c1-8)
OUT_DIR="$BENCH_DIR/results/$(date +%Y%m%d_%H%M%S)_${CODE_MD5}_blocks$BLOCKS"

# Runs as the host user so that the build and result files stay editable outside the container.
CMD=(docker exec -u "$(id -u):$(id -g)"
     -e HOME="$BENCH_DIR/build"
     -e CUDA_VISIBLE_DEVICES=
     -e IAXL_SCRATCH_POOL_SIZE_GB="$POOL_GB"
     -e LD_PRELOAD="$PRELOAD"
     -e TCMALLOC_RELEASE_RATE="$TCMALLOC_RELEASE_RATE"
     -w "$REPO_DIR" "$CONTAINER_NAME"
     taskset -c "$CPUS" python3 -u "$BENCH_DIR/release_bench.py" --out-dir "$OUT_DIR"
     --blocks-per-task "$BLOCKS" "$@")

if [[ "$DRY_RUN" == 1 ]]; then
    echo "${CMD[*]}"
    exit 0
fi
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "container $CONTAINER_NAME is not running; start it with ./start.sh" >&2
    exit 1
fi
mkdir -p "$OUT_DIR" "$BENCH_DIR/build"
echo "results: $OUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUT_DIR/run.log"
