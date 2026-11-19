#!/bin/bash
# Experiment a: host copy throughput of raw (uncompressed) KV blocks with 1, 4,
# 12 and 20 OpenMP threads (tests/bench_dev/raw_copy_bench.c). kvshrink's
# no-compression settings run with 4 threads (setvars.sh caps it there), the
# compression settings with 12. Runs in the KVShrink container on rank 0's
# CPUs and memory as the benchmark binds them (cores 0-23, NUMA node 0) with
# kvshrink's LD_PRELOAD (tcmalloc + libiomp5). CPU only, no GPU.
# Results: tests/bench_dev/results/raw-copy-bench-<time>/result.txt
#
# Runs on the b300 HOST: bash tests/bench_dev/raw_copy_bench.sh
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so
OUT_DIR=tests/bench_dev/results/raw-copy-bench-$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"
cp tests/bench_dev/raw_copy_bench.c "$0" "$OUT_DIR/"

docker exec -w "$REPO_DIR" "$CONTAINER" bash -lc "
    gcc -O2 -fopenmp -o $OUT_DIR/raw_copy_bench tests/bench_dev/raw_copy_bench.c || exit 1
    for threads in 1 4 12 20; do
        # No OMP_PLACES / OMP_PROC_BIND: with libiomp5 preloaded under numactl they
        # ran every thread on one core (1, 4 and 12 threads all gave 7.7 GB/s, 11-17).
        OMP_NUM_THREADS=\$threads LD_PRELOAD=$PRELOAD \
            timeout 300 numactl --physcpubind=0-23 --membind=0 $OUT_DIR/raw_copy_bench
    done
" 2>&1 | tee "$OUT_DIR/result.txt"
echo "results: $OUT_DIR/result.txt"
