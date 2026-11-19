#!/bin/bash
# Builds and runs gdr_handle_overflow_check.c inside the iaxl container (see the .c file for
# what it checks). Output goes to tests/bench_dev/results/gdr-handle-overflow-<time>/run.log.
#
# Usage (from the repo root on the host):
#   bash tests/bench_dev/gdr_handle_overflow_check.sh [GPU] [CHURN_GIB] [STEP_MIB]
# Defaults: GPU 1, 128 GiB churn buffer, span grows by 128 MiB per call. With these values
# the cumulative pinned size crosses 2^32 pages (16 TiB) at call 513 (span 64.125 GiB).
set -euo pipefail

GPU=${1:-1}
CHURN_GIB=${2:-128}
STEP_MIB=${3:-128}
CONTAINER_NAME=${CONTAINER_NAME:-iaxl.vllm-fddeef0-qiuyu}
DSA_WQ=${DSA_WQ:-wq0.0}
TIMEOUT_S=${TIMEOUT_S:-900}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT_REL=tests/bench_dev/results/gdr-handle-overflow-$(date +%Y%m%d_%H%M%S)

docker exec -w "$REPO" "$CONTAINER_NAME" bash -c "
set -euo pipefail
mkdir -p $OUT_REL
md5sum tests/bench_dev/gdr_handle_overflow_check.c iaxl/csrc/dsa_gd/dsa_gd.c \
    iaxl/csrc/dsa_gd/dsa_memcpy.c > $OUT_REL/code.md5
gcc -O2 -g -Wall -std=gnu11 -D_GNU_SOURCE -fopenmp \
    -I iaxl/csrc/include -I /usr/local/cuda/include \
    -o $OUT_REL/gdr_handle_overflow_check \
    tests/bench_dev/gdr_handle_overflow_check.c iaxl/csrc/dsa_gd/dsa_gd.c \
    iaxl/csrc/dsa_gd/dsa_memcpy.c iaxl/csrc/common/envs.c \
    -L /usr/local/cuda/lib64 -lcudart -lgdrapi -ldl -lpthread
echo build ok
IAXL_DSA_WQS=$DSA_WQ timeout $TIMEOUT_S $OUT_REL/gdr_handle_overflow_check $GPU $CHURN_GIB $STEP_MIB \
    > $OUT_REL/run.log 2>&1 || echo \"exit code \$?\" >> $OUT_REL/run.log
echo EXIT >> $OUT_REL/run.log
"
echo "log: $REPO/$OUT_REL/run.log"
