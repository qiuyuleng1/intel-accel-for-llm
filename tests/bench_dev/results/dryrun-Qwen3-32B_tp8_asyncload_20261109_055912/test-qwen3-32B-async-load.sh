#!/bin/bash
# Qwen3-32B at TP=8: does KVShrink's async KV loading raise TPOT?
# Input 32768 tokens, output 128, hit rate 100 only, concurrency 16 and 48,
# concurrency * 5 requests per run (80 and 240), 5 warm-ups and the prewarm of
# kvshrink-benchmark.sh. One vLLM serve per setting:
#   default           kvshrink with the setvars.sh defaults (ENABLED=1,
#                     LAYERS_DYNAMIC=1, MAP 0-3:0,4-6:8,7-:16): with 7 or more
#                     requests in flight a hit is scheduled once its first 16
#                     of 64 layers are loaded, the other 48 load during its
#                     prefill step
#   async-off         kvshrink, KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0: synchronous
#                     load, all 64 layers load during the hit's prefill step
#   async-all-layers  kvshrink, ENABLED=1, LAYERS_DYNAMIC=0, LAYERS=-1: a hit is
#                     scheduled only after all 64 layers are loaded
#   vllm-prefix       plain vLLM with its GPU prefix cache: the reference TPOT
#   vllm-prefix-piecewise  vllm-prefix with CUDAGRAPH_MODE=PIECEWISE, the CUDA
#                     graph mode vLLM forces on kvshrink (KVShrinkConnector
#                     requires it); vllm-prefix otherwise runs FULL_AND_PIECEWISE
#                     (Nov 6: kvshrink TPOT +26-51% over vllm-prefix at every
#                     async-load setting)
#   kvshrink-gpu-prefix  kvshrink default with GPU_PREFIX_CACHING=1: after the
#                     prewarm every hit comes from vLLM's GPU prefix cache, so
#                     KVShrink runs (PIECEWISE, its per-step hooks) but loads
#                     nothing and all requests share one copy of the prompt
#                     (Nov 6: forced PIECEWISE raised vllm-prefix TPOT by 0-9%
#                     only, kvshrink stayed 25-39% above that)
#   lmcache           LMCache 0.5.6 MP mode on vLLM 0.33.0, GPU prefix cache off
#                     like kvshrink: every request gets its own GPU copy of the
#                     prompt (Nov 9 profile: that copy alone made kvshrink's
#                     full-batch decode step 72 ms instead of 49 ms at 48 requests).
#                     LMCache chunk 16 = the vLLM block size, so a hit covers as
#                     many tokens as kvshrink's.
#   vllm-prefix-v033  vllm-prefix on the same vLLM 0.33.0 image: the reference
#                     TPOT for lmcache.
#                     Both run in a new container of the LMCache image
#                     (docker run --rm), not in the KVShrink container.
# Not Qwen3.8-27B: on that hybrid model the map's 16 layers come on top of its
# 48 mamba layers, i.e. all 64, so default equals async-all-layers there.
# Hit rate 100 only: at hit rate 80 every request also prefills ~6500 tokens,
# which dominates TPOT (~700 ms vs ~120 ms on Qwen3.8-27B, Nov 5).
# Concurrency 48: vLLM fits ~57 requests of 32896 tokens at TP=8 (estimated
# from 285,061 KV tokens at TP=2 on this host), so 64 does not fit.
# ScratchPool 48 GB per rank instead of 28: a hit loads 1 GiB of KV per rank
# (64 layers * 32768 tokens * 512 bytes), and 48 can arrive at once.
# MAX_MODEL_LEN 40960 = the model's max_position_embeddings.
# VLLM_LOG_STATS_INTERVAL=1 makes vLLM log Running / Waiting / Deferred every
# second instead of every 10 s (read by tests/bench_dev/running-reqs-per-run.py).
#
# Runs on the HOST, like test-qwen38-27B.sh: checks the host settings of
# PERFORMANCE_MODE_SETUP.md, makes sure the KVShrink container is up, and drops
# the page cache before every serve. A failed setting is reported and the next
# one still runs.
# Results: bench-results/Qwen3-32B_tp8_asyncload_<time>/<setting>/
#   (summary.csv, run.log, <config>/serve.log).
#   bash test-qwen3-32B-async-load.sh
#   SETTINGS="async-off vllm-prefix" bash test-qwen3-32B-async-load.sh   # some settings only
#   DRY_RUN=1 bash test-qwen3-32B-async-load.sh                         # print the plan only
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/media/model-space/Qwen3-32B
SETTINGS=${SETTINGS:-"default async-off async-all-layers vllm-prefix"}
DRY_RUN=${DRY_RUN:-0}
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
LMCACHE_IMAGE=qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6
LMCACHE_CONTAINER=iaxl.lmcache-bench
MODEL_TAG=$(basename "$MODEL")
# RESULTS_TAG names what a run is for, e.g. steptrace.
RUN_NAME=${MODEL_TAG}_tp8_asyncload${RESULTS_TAG:+_$RESULTS_TAG}_$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-$RUN_NAME
else
    RESULTS_ROOT=bench-results/$RUN_NAME
fi
BENCH_ENV="MODEL=$MODEL TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_MODEL_LEN=40960 BLOCK_SIZE=16"
BENCH_ENV+=" LEN_CONCURRENCY='32768:16,48' HIT_RATES='100' NUM_PROMPTS_BY_CONCURRENCY='16:80 48:240'"
BENCH_ENV+=" IAXL_SCRATCH_POOL_SIZE_GB=48 IAXL_DDR_POOL_TOTAL_SIZE_GB=40"
BENCH_ENV+=" LMCACHE_CHUNK_SIZE=16 LMCACHE_L1_SIZE_GB=40"
BENCH_ENV+=" VLLM_LOG_STATS_INTERVAL=1 DRY_RUN=$DRY_RUN"
# kvshrink/step_trace.py: KVSHRINK_STEP_TRACE=1 logs one line per worker step,
# KVSHRINK_THREAD_MAP=1 logs the worker's Python thread ids.
BENCH_ENV+=" KVSHRINK_STEP_TRACE=${KVSHRINK_STEP_TRACE:-0} KVSHRINK_THREAD_MAP=${KVSHRINK_THREAD_MAP:-0}"

# kvshrink-benchmark.sh config of one setting.
setting_config() {
    case "$1" in
        default|async-off|async-all-layers|kvshrink-gpu-prefix) echo "kvshrink" ;;
        vllm-prefix|vllm-prefix-piecewise) echo "vllm-prefix" ;;
        lmcache) echo "lmcache" ;;
        vllm-prefix-v033) echo "vllm-prefix-v033" ;;
        *) echo "Unknown setting: $1 (expected default, async-off, async-all-layers, kvshrink-gpu-prefix, vllm-prefix, vllm-prefix-piecewise, lmcache or vllm-prefix-v033)" >&2; return 1 ;;
    esac
}

# KVShrink async-load variables of one setting; setvars.sh supplies the rest.
setting_env() {
    case "$1" in
        default) echo "" ;;
        async-off) echo "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0" ;;
        async-all-layers) echo "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=1 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=0 KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=-1" ;;
        vllm-prefix) echo "" ;;
        vllm-prefix-piecewise) echo "CUDAGRAPH_MODE=PIECEWISE" ;;
        kvshrink-gpu-prefix) echo "GPU_PREFIX_CACHING=1" ;;
        lmcache|vllm-prefix-v033) echo "" ;;
        *) echo "Unknown setting: $1 (expected default, async-off, async-all-layers, kvshrink-gpu-prefix, vllm-prefix, vllm-prefix-piecewise, lmcache or vllm-prefix-v033)" >&2; return 1 ;;
    esac
}

for setting in $SETTINGS; do
    setting_config "$setting" > /dev/null || exit 1
done

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen3-32B-async-load.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

echo "==========================================================="
echo "Qwen3-32B async-load test  model=$MODEL  settings=[$SETTINGS]"
echo "  $BENCH_ENV"
echo "  KVShrink container=$KVSHRINK_CONTAINER"
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
for setting in $SETTINGS; do
    results_dir=$RESULTS_ROOT/$setting
    bench_cmd="CONFIGS=$(setting_config "$setting") RESULTS_DIR=$results_dir $BENCH_ENV $(setting_env "$setting") bash kvshrink-benchmark.sh"
    echo ""
    echo "### $setting ($(date +'%F %T')): $(setting_config "$setting") [$(setting_env "$setting")] -> $results_dir"
    # A dry run only prints the plan, so it does not need the container.
    if [ "$DRY_RUN" = "1" ]; then
        bash -c "$bench_cmd"
    elif [ "$(setting_config "$setting")" = "lmcache" ] || [ "$(setting_config "$setting")" = "vllm-prefix-v033" ]; then
        reclaim_host_memory
        docker run --rm --name "$LMCACHE_CONTAINER" --gpus all --network host --ipc host \
            -v "$REPO_ROOT:$REPO_ROOT" -v /media/model-space:/media/model-space -w "$REPO_ROOT" \
            --entrypoint bash "$LMCACHE_IMAGE" -c "$bench_cmd"
    else
        reclaim_host_memory
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
    status=$?
    if [ "$status" != "0" ]; then
        echo "ERROR: $setting exited with status $status" >&2
        failed+=" $setting"
    fi
done

if [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "=== per setting: async-load values the serve used, how often each load mode was chosen (all requests incl. prewarm and warm-ups), and per run: completed/num_prompts, TTFT mean/median ms, TPOT mean/median ms, output tokens/s"
    for setting in $SETTINGS; do
        config=$(setting_config "$setting")
        serve_log=$RESULTS_ROOT/$setting/$config/serve.log
        summary=$RESULTS_ROOT/$setting/summary.csv
        echo "## $setting ($config)"
        if [ ! -f "$serve_log" ] || [ ! -f "$summary" ]; then
            echo "    no results"
            continue
        fi
        tr '\r' '\n' < "$serve_log" | grep -a -m 4 "^  KVSHRINK_VLLM_KV_ASYNC_LOAD"
        tr '\r' '\n' < "$serve_log" | grep -aoE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" | head -1
        tr '\r' '\n' < "$serve_log" | grep -aoE "enable_prefix_caching=(True|False)" | head -1
        tr '\r' '\n' < "$serve_log" | grep -aoE "async=(True|False), selected_load_layers=-?[0-9]+, async_load_layers=-?[0-9]+" | sort | uniq -c
        awk -F, 'NR > 1 {printf "    con %s hit %s: %s/%s, TTFT %s / %s, TPOT %s / %s, %s tok/s\n", $5, $6, $11, $7, $16, $17, $19, $20, $14}' "$summary"
        python3 tests/bench_dev/running-reqs-per-run.py "$RESULTS_ROOT/$setting" "$config"
    done
fi

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/*/summary.csv"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
