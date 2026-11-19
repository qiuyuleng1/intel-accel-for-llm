#!/bin/bash
# Is the API server (frontend) what limits Qwen3.8-27B at hit rate 95 on b300?
# Plain vLLM with its GPU prefix cache (kvshrink-benchmark.sh config
# vllm-prefix), TP=4 on GPU0/2/4/6 as test-qwen38-27B.sh, at the two points
# where requests queued most before reaching the engine: input 32000 tokens at
# concurrency 256 and 128000 tokens at concurrency 64, hit rate 95, output 128,
# concurrency * 10 requests per run. Warm-ups and seeds are those of
# test-qwen38-27B.sh, so the python variant repeats the two runs of
# bench-results/Qwen3.8-27B_tp4_20261111_095943 vllm-prefix; only the 32000
# prewarm differs (concurrency 256 instead of 300, the largest one here).
#
# Frontend variants, each one serve start (one kvshrink-benchmark.sh call):
#   rust            VLLM_USE_RUST_FRONTEND=1: vllm-rs replaces the Python API
#                   server; it tokenizes on min(CPUs, 32) threads
#   python          vLLM's default Python API server: one tokenizer thread
#                   (--renderer-num-workers 1)
#   rust-threads8   rust with VLLM_RS_REQUEST_WORKER_THREADS=8
#   python-workers8 python with --renderer-num-workers 8
# rust runs first so that a Rust frontend that cannot serve this model shows up
# within minutes.
#
# Every variant gets --chat-template CHAT_TEMPLATE, the model's
# chat_template.jinja without its {% generation %} / {% endgeneration %} lines:
# vllm-rs refuses to start on the original ("unknown statement generation").
# The bench sends /v1/completions, which never renders a chat template, so the
# template does not change any result; it is passed to the Python variants too
# so that every variant has the same serve options.
#
# While a variant runs, tests/bench_dev/cpu_thread_sampler.py samples the CPU
# usage of every thread of the API server / vllm-rs, EngineCore, workers and
# the bench client once a second into <variant>/cpu_threads.csv.
#
# Runs on the HOST, like test-qwen38-27B.sh. Results:
# bench-results/Qwen3.8-27B_tp4_frontend_<time>/<variant>/ (summary.csv,
# run.log, vllm-prefix/serve.log, cpu_threads.csv).
# Print the plan without starting anything: DRY_RUN=1 bash test-qwen38-27B-frontend.sh
# Only some variants: VARIANTS="python rust" bash test-qwen38-27B-frontend.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Qwen3.8-27B
CHAT_TEMPLATE=$REPO_ROOT/tests/bench_dev/qwen38-27B-chat-template-no-generation.jinja
VARIANTS=${VARIANTS:-"rust python rust-threads8 python-workers8"}
DRY_RUN=${DRY_RUN:-0}
# One variant normally takes ~15 minutes; a hung bench must not block the rest.
VARIANT_TIMEOUT_S=3600
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
MODEL_TAG=$(basename "$MODEL")
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-${MODEL_TAG}_tp4_frontend_$(date +%Y%m%d_%H%M%S)
else
    RESULTS_ROOT=bench-results/${MODEL_TAG}_tp4_frontend_$(date +%Y%m%d_%H%M%S)
fi
# Same GPU / NUMA binding as test-qwen38-27B.sh (see there).
ACCEL_ENV="TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6"
ACCEL_ENV+=" VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'"
ACCEL_ENV+=" KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'"
ACCEL_ENV+=" KVSHRINK_IAA_DEVICES='0|1|2|3'"
KVSHRINK_START_ENV=$ACCEL_ENV
# Same as test-qwen38-27B.sh except the points and hit rates; the pool sizes do
# not matter for vllm-prefix (no connector) but are kept identical.
BENCH_ENV="MODEL=$MODEL $ACCEL_ENV"
BENCH_ENV+=" LEN_CONCURRENCY='32000:256 128000:64' HIT_RATES=95"
BENCH_ENV+=" MAX_MODEL_LEN=256128 LMCACHE_CHUNK_SIZE=784"
BENCH_ENV+=" IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 LMCACHE_L1_SIZE_GB=1000"
BENCH_ENV+=" DRY_RUN=$DRY_RUN"

# Environment that selects the frontend of one variant.
variant_env() {
    case "$1" in
        rust) echo "VLLM_USE_RUST_FRONTEND=1 VLLM_SERVE_EXTRA_ARGS='--chat-template $CHAT_TEMPLATE'" ;;
        python) echo "VLLM_SERVE_EXTRA_ARGS='--chat-template $CHAT_TEMPLATE'" ;;
        rust-threads8) echo "VLLM_USE_RUST_FRONTEND=1 VLLM_RS_REQUEST_WORKER_THREADS=8 VLLM_SERVE_EXTRA_ARGS='--chat-template $CHAT_TEMPLATE'" ;;
        python-workers8) echo "VLLM_SERVE_EXTRA_ARGS='--chat-template $CHAT_TEMPLATE --renderer-num-workers 8'" ;;
        *) echo "Unknown variant: $1 (expected rust, python, rust-threads8 or python-workers8)" >&2; return 1 ;;
    esac
}

for variant in $VARIANTS; do
    variant_env "$variant" > /dev/null || exit 1
done

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen38-27B-frontend.log") 2>&1
cp "${BASH_SOURCE[0]}" tests/bench_dev/cpu_thread_sampler.py "$CHAT_TEMPLATE" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

SAMPLER_PID=""
stop_sampler() {
    if [ -n "$SAMPLER_PID" ]; then
        kill -TERM "$SAMPLER_PID" 2> /dev/null
        wait "$SAMPLER_PID" 2> /dev/null
        SAMPLER_PID=""
    fi
}
trap stop_sampler EXIT

echo "==========================================================="
echo "Qwen3.8-27B frontend test  variants=[$VARIANTS]"
echo "  $BENCH_ENV"
echo "  KVShrink container=$KVSHRINK_CONTAINER  timeout per variant ${VARIANT_TIMEOUT_S}s"
echo "  results -> $RESULTS_ROOT"
echo "==========================================================="

echo "Host settings (PERFORMANCE_MODE_SETUP.md):"
if ! check_host_settings && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: host settings differ from PERFORMANCE_MODE_SETUP.md; restore them first" >&2
    exit 1
fi

if [ "$DRY_RUN" != "1" ] && ! ensure_kvshrink_container; then
    exit 1
fi

failed=""
for variant in $VARIANTS; do
    variant_dir="$RESULTS_ROOT/$variant"
    mkdir -p "$variant_dir"
    bench_cmd="CONFIGS=vllm-prefix RESULTS_DIR=$variant_dir $BENCH_ENV $(variant_env "$variant")"
    # timeout inside the container: on expiry it signals kvshrink-benchmark.sh
    # and its bench client, and kvshrink-benchmark.sh's trap stops the serve.
    bench_cmd+=" timeout $VARIANT_TIMEOUT_S bash kvshrink-benchmark.sh"
    echo ""
    echo "### $variant ($(date +'%F %T')): $(variant_env "$variant") -> $variant_dir"
    if [ "$DRY_RUN" != "1" ]; then
        reclaim_host_memory
        # Pinned to the last hyperthread, away from the vLLM OMP cores 0-95.
        taskset -c 191 python3 tests/bench_dev/cpu_thread_sampler.py "$variant_dir/cpu_threads.csv" &
        SAMPLER_PID=$!
    fi
    docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    status=$?
    stop_sampler
    if [ "$status" != "0" ]; then
        echo "ERROR: $variant exited with status $status" >&2
        failed+=" $variant"
    fi
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/<variant>/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
