#!/bin/bash
# Kimi-K3 on b300 (8 x B300): kvshrink, vllm-prefix, vllm-wo-prefix and lmcache in two layouts:
#   dp2tp4  DP2 TP4 with expert parallel (--enable-expert-parallel), the layout found on 11-13
#   tp8     TP8
# Input lengths and concurrencies, output 128, hit rates 80 95 (vllm-wo-prefix: 80 only),
# concurrency * 10 requests per run (vllm-wo-prefix: * 2):
#   dp2tp4  32000 at 4, 8 and 12, 128000 at 2 (the 11-13 capacity test: 6 + 6 and 1 + 1 per DP rank ran
#           without preemption, 7 + 7 at 32000 was preempted)
#   tp8     the dp2tp4 cases plus higher ones: 32000 at 4, 8, 12 and 24, 128000 at 2 and 6, 256000 at 2.
#           The higher ones are not measured: TP8's GPU KV cache holds
#           1,117,892 tokens (2.46x DP2 TP4's two groups together), and these use ~70%, ~70%
#           and ~46% of it for the prompts, leaving room for the KDA state blocks that made
#           DP2 TP4 fit 6 instead of vLLM's 7 per rank.
# With DP the 5 warm-ups of every run go to each DP rank (kvshrink-benchmark.sh).
#
# vLLM options in both layouts: --gpu-memory-utilization 0.9, --max-num-seqs 128 (needed by
# DP2 TP4: the default 1024 ran out of memory during profiling), MAX_MODEL_LEN = longest input
# + 128 output tokens (131200 dp2tp4, 256128 tp8), plus --enable-expert-parallel in dp2tp4.
# KVShrink: QAT +
# IAA compression and DSA transfers, the 2 GPUs of a NUMA node share its devices (setvars.sh;
# needs the DSA set to 2 WQs per device, tools/setup_dsa_cnt.sh 2), 8 host OpenMP threads per
# rank (QAT 4 + IAA 4 instances, as on 11-13; a TP8 rank has 12 CPUs and setvars.sh keeps 4 of
# them for vLLM, so 8 is the most it allows). KVShrink DDR pool and
# LMCache CPU cache 600 GB each (75 GB per rank at TP8; the largest run, tp8 32000 x 24 at
# hit 80, adds ~1.6M new tokens, ~42 GB per rank uncompressed, and LMCache pins its cache
# faster than 1200 GB); scratch pool SCRATCH_POOL_SIZE_GB per rank (default 8 GB as on 11-13;
# night-kimi-then-qwen.sh in the b300 repo passes 64).
# LMCache chunk size = vLLM block size: 1536 at TP4, 768 at TP8.
# kvshrink-nozip: kvshrink without compression (QAT, IAA and CPU compression off), DSA
# transfers, and the same 8 host OpenMP threads instead of setvars.sh's cap of 4 without
# compression (on Qwen3.8-27B, 32000 x 256 at hit 95, 12 instead of 4 threads gave 28% more
# requests/s, 11-17). Results in <layout>/kvshrink-nozip/: kvshrink-benchmark.sh names its
# serve log directory and summary rows after its config, kvshrink.
#
# Runs on the HOST. Every (layout, config) is one kvshrink-benchmark.sh call:
#   kvshrink, vllm-prefix, vllm-wo-prefix: docker exec into the KVShrink container.
#   lmcache: docker exec into lmcache.vllm-kimi-k3-qiuyu (a container of the LMCache image
#       created with --init and kept, so the serve left running survives the call).
# Default (11-17): tp8, kvshrink then kvshrink-nozip. The 11-13 run was LAYOUTS="dp2tp4 tp8"
# CONFIGS="kvshrink vllm-prefix vllm-wo-prefix lmcache" (dp2tp4 first). A serve is stopped
# only to start the next one; the last (layout, config) is left running unless
# KEEP_LAST_SERVE=0 (night-kimi-then-qwen.sh in the b300 repo runs Qwen3.8-27B next).
# A failed config is reported and the next one still runs. Every config has a timeout
# (CONFIG_TIMEOUT_S, 2 h; tp8 kvshrink took 43 min on 11-13) so that a hung serve or bench
# cannot block what follows.
#
# Results: bench-results/Kimi-K3_<time>/<layout>/ (summary.csv, run.log, per-config serve logs).
# Copying results back: b300 cannot reach the 6000d host, so this script cannot push. After
# every (layout, config), i.e. every vLLM restart, it writes CONFIG_DONE/<layout>-<config>
# (exit status and time), after the last one ALL_DONE; tests/bench_dev/pull-b300-kimi-results.sh
# on the 6000d host then copies the whole run directory back.
# Print the plan without starting anything: DRY_RUN=1 bash test-kimi-k3.sh
# RUN_NAME=<name> sets the run directory name (default Kimi-K3_<time>).
# Subsets: LAYOUTS="tp8" CONFIGS="kvshrink lmcache" bash test-kimi-k3.sh; HIT_RATES=95 ...
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Kimi-K3
LAYOUTS=${LAYOUTS:-"tp8"}
CONFIGS=${CONFIGS:-"kvshrink kvshrink-nozip"}
HIT_RATES=${HIT_RATES:-"80 95"}
CACHE_SIZE_GB=${CACHE_SIZE_GB:-600}
SCRATCH_POOL_SIZE_GB=${SCRATCH_POOL_SIZE_GB:-8}
DRY_RUN=${DRY_RUN:-0}
KEEP_LAST_SERVE=${KEEP_LAST_SERVE:-1}
CONFIG_TIMEOUT_S=7200
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=\${CONTAINER_NAME:-\(.*\)}$/\1/p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
KVSHRINK_START_ENV=""
LMCACHE_CONTAINER=lmcache.vllm-kimi-k3-qiuyu
PORT=8000
RUN_NAME=${RUN_NAME:-Kimi-K3_$(date +%Y%m%d_%H%M%S)}
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=bench-results/dryrun-$RUN_NAME
else
    RESULTS_ROOT=bench-results/$RUN_NAME
fi

# "<DP_SIZE> <TP_SIZE> <LMCACHE_CHUNK_SIZE> <extra vLLM args>" of a layout.
layout_settings() {
    case "$1" in
        dp2tp4) echo "2 4 1536 --gpu-memory-utilization 0.9 --max-num-seqs 128 --enable-expert-parallel" ;;
        tp8) echo "1 8 768 --gpu-memory-utilization 0.9 --max-num-seqs 128" ;;
        *) echo "Unknown layout: $1 (expected dp2tp4 or tp8)" >&2; return 1 ;;
    esac
}

# "<MAX_MODEL_LEN> <input_len:concurrency ...>" of a layout.
layout_lengths() {
    case "$1" in
        dp2tp4) echo "131200 32000:4,8,12 128000:2" ;;
        tp8) echo "256128 32000:4,8,12,24 128000:2,6 256000:2" ;;
        *) echo "Unknown layout: $1 (expected dp2tp4 or tp8)" >&2; return 1 ;;
    esac
}

# kvshrink-benchmark.sh config of a config of this script.
benchmark_config() {
    case "$1" in
        kvshrink|kvshrink-nozip) echo "kvshrink" ;;
        vllm-prefix|vllm-wo-prefix|lmcache) echo "$1" ;;
        *) echo "Unknown config: $1" >&2; return 1 ;;
    esac
}

# kvshrink switches of a config (read by setvars.sh in the serve script); empty keeps
# kvshrink-benchmark.sh's defaults.
config_env() {
    case "$1" in
        kvshrink) echo "IAXL_OMP_THREAD_NUM=8" ;;
        kvshrink-nozip) echo "IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1 IAXL_OMP_THREAD_NUM=8" ;;
        *) echo "" ;;
    esac
}

for layout in $LAYOUTS; do
    layout_settings "$layout" > /dev/null || exit 1
done
for config in $CONFIGS; do
    benchmark_config "$config" > /dev/null || exit 1
done

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-kimi-k3.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

# The (layout, config) pair whose serve stays running.
last_layout=${LAYOUTS##* }
last_config=${CONFIGS##* }

# One kvshrink-benchmark.sh call for one layout and config.
run_config() {
    local layout=$1 config=$2 results_dir=$3
    local dp_size tp_size chunk_size extra_args max_model_len len_concurrency keep_serve=0
    read -r dp_size tp_size chunk_size extra_args <<< "$(layout_settings "$layout")"
    read -r max_model_len len_concurrency <<< "$(layout_lengths "$layout")"
    if [ "$KEEP_LAST_SERVE" = "1" ] && [ "$layout" = "$last_layout" ] && [ "$config" = "$last_config" ]; then
        keep_serve=1
    fi
    local bench_env="MODEL=$MODEL DP_SIZE=$dp_size TP_SIZE=$tp_size CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    bench_env+=" LEN_CONCURRENCY='$len_concurrency' HIT_RATES='$HIT_RATES' MAX_MODEL_LEN=$max_model_len"
    bench_env+=" VLLM_SERVE_EXTRA_ARGS='$extra_args' LMCACHE_CHUNK_SIZE=$chunk_size"
    bench_env+=" IAXL_SCRATCH_POOL_SIZE_GB=$SCRATCH_POOL_SIZE_GB IAXL_DDR_POOL_TOTAL_SIZE_GB=$CACHE_SIZE_GB LMCACHE_L1_SIZE_GB=$CACHE_SIZE_GB"
    bench_env+=" KEEP_SERVE=$keep_serve DRY_RUN=$DRY_RUN $(config_env "$config")"
    # timeout runs inside the container: on expiry it signals kvshrink-benchmark.sh, whose
    # trap stops the serve.
    local bench_cmd="CONFIGS=$(benchmark_config "$config") RESULTS_DIR=$results_dir $bench_env timeout $CONFIG_TIMEOUT_S bash kvshrink-benchmark.sh"
    echo "  $bench_cmd"
    if [ "$config" = "lmcache" ]; then
        docker exec -w "$REPO_ROOT" "$LMCACHE_CONTAINER" bash -lc "$bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Kimi-K3 test  model=$MODEL  layouts=[$LAYOUTS]  configs=[$CONFIGS]"
echo "  hit rates=[$HIT_RATES] cache=${CACHE_SIZE_GB} GB scratch=${SCRATCH_POOL_SIZE_GB} GB per rank"
for layout in $LAYOUTS; do
    echo "  $layout: MAX_MODEL_LEN input:concurrency = $(layout_lengths "$layout")"
done
echo "  KVShrink container=$KVSHRINK_CONTAINER  LMCache container=$LMCACHE_CONTAINER"
if [ "$KEEP_LAST_SERVE" = "1" ]; then
    echo "  left running at the end: $last_layout $last_config"
else
    echo "  left running at the end: none (KEEP_LAST_SERVE=0)"
fi
echo "  timeout per config ${CONFIG_TIMEOUT_S}s"
echo "  results -> $RESULTS_ROOT"
echo "==========================================================="

# Fail before starting anything when the host is not in performance mode.
echo "Host settings (PERFORMANCE_MODE_SETUP.md):"
if ! check_host_settings && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: host settings differ from PERFORMANCE_MODE_SETUP.md; restore them first" >&2
    exit 1
fi

if [ "$DRY_RUN" != "1" ]; then
    # 8 workers share 4 DSA devices, one dedicated WQ each.
    if [ "$(cat /sys/bus/dsa/devices/wq0.1/state 2> /dev/null)" != "enabled" ]; then
        echo "ERROR: DSA has 1 WQ per device; run tools/setup_dsa_cnt.sh 2 first" >&2
        exit 1
    fi
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "ERROR: a vLLM already serves on port $PORT; stop it first" >&2
        exit 1
    fi
    if ! ensure_kvshrink_container; then
        exit 1
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$LMCACHE_CONTAINER" 2> /dev/null)" != "true" ]; then
        echo "ERROR: $LMCACHE_CONTAINER is not running; create it with:" >&2
        echo "  docker run -d --init --name $LMCACHE_CONTAINER --gpus all --ipc=host --network host --ulimit memlock=-1:-1 -v $MODEL:$MODEL:ro -v $REPO_ROOT:$REPO_ROOT -w $REPO_ROOT --entrypoint sleep qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7 infinity" >&2
        exit 1
    fi
fi

failed=""
for layout in $LAYOUTS; do
    for config in $CONFIGS; do
        results_dir=$RESULTS_ROOT/$layout
        if [ "$config" = "kvshrink-nozip" ]; then
            results_dir=$RESULTS_ROOT/$layout/kvshrink-nozip
        fi
        echo ""
        echo "### $layout $config ($(date +'%F %T')) -> $results_dir"
        run_config "$layout" "$config" "$results_dir"
        status=$?
        if [ "$status" != "0" ]; then
            echo "ERROR: $layout $config exited with status $status" >&2
            failed+=" $layout/$config"
        fi
        mkdir -p "$RESULTS_ROOT/CONFIG_DONE"
        echo "status=$status end=$(date +'%F %T')" > "$RESULTS_ROOT/CONFIG_DONE/$layout-$config"
    done
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/<layout>/summary.csv"
echo "end=$(date +'%F %T') failed=[${failed# }]" > "$RESULTS_ROOT/ALL_DONE"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
