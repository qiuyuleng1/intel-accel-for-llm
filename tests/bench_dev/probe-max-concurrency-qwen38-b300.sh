#!/bin/bash
# Runs on the b300 HOST. Reads vLLM's GPU KV capacity and "Maximum
# concurrency" for Qwen3.8-27B at TP=4 on GPU0/2/4/6, with the same binding as
# test-qwen38-27B.sh: starts the KVShrink container if needed, then runs
# probe-max-concurrency.sh in it (one KVShrink serve per input length, no
# requests). The longest input is the model's max_position_embeddings 262144
# minus the 128 output tokens.
#   bash tests/bench_dev/probe-max-concurrency-qwen38-b300.sh
# Results: tests/bench_dev/results/probe-tp4-<time>/summary.txt
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Qwen3.8-27B
INPUT_LENS=${INPUT_LENS:-"32768 131072 262016"}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
# Same binding as test-qwen38-27B.sh; see the reasons there.
ACCEL_ENV="TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6"
ACCEL_ENV+=" VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'"
ACCEL_ENV+=" KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'"
KVSHRINK_START_ENV=$ACCEL_ENV

# reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

if ! ensure_kvshrink_container; then
    exit 1
fi
reclaim_host_memory
docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc \
    "MODEL=$MODEL $ACCEL_ENV INPUT_LENS='$INPUT_LENS' bash tests/bench_dev/probe-max-concurrency.sh"
