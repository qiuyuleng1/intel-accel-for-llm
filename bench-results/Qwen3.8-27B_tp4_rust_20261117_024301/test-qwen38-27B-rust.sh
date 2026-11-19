#!/bin/bash
# Qwen3.8-27B on b300 at TP=4 on GPU0/2/4/6 with the Rust frontend
# (VLLM_USE_RUST_FRONTEND=1: vllm-rs replaces vLLM's Python API server, with
# its default request thread count, min(CPUs, 32)) for every setting. Input
# 32000 tokens at concurrency 32 64 128 256, 128000 at 80 and 256000 at 40,
# output 128, hit rates 80 95, concurrency * 10 requests per run
# (vllm-wo-prefix: * 2, hit rate 80 only). Fewer concurrencies than
# test-qwen38-27B.sh (which also runs 32000 at 300, 128000 at 64 and 256000 at
# 32) so that all seven settings finish in ~15 h; same requests per run.
#
# Settings, each one vLLM start (one kvshrink-benchmark.sh call), in this order:
#   kvshrink-zip-dsa      kvshrink: QAT + IAA compression (no CPU compression),
#                         DSA transfers (kvshrink-benchmark.sh's defaults)
#   lmcache               LMCache MP mode, in the LMCache image; FLASH_ATTN
#   vllm-prefix           plain vLLM with its GPU prefix cache
#   kvshrink-nozip-dsa    kvshrink: no compression at all, DSA transfers
#   kvshrink-nozip-nodsa  kvshrink: no compression at all, CUDA copies instead
#                         of DSA
#   vllm-wo-prefix        plain vLLM without KV reuse
#   kvshrink-zip-dsa-flash-attn  kvshrink-zip-dsa under FLASH_ATTN, to compare
#                         with lmcache under the same attention backend
#   kvshrink-nozip-dsa-omp12  kvshrink-nozip-dsa with 12 host OpenMP threads,
#                         as the compression settings get, instead of
#                         setvars.sh's cap of 4 without compression; not in the
#                         default list (SETTINGS=kvshrink-nozip-dsa-omp12)
# A failed setting is reported and the next one still runs. Every setting has a
# timeout (SETTING_TIMEOUT_S) so that a hung serve or bench cannot block the
# rest.
#
# Attention backend: vLLM's default (FlashInfer, block size 784) except lmcache
# and kvshrink-zip-dsa-flash-attn, which use FLASH_ATTN (block size 896; the
# LMCache chunk size must equal it). Under FlashInfer LMCache fails to store any
# KV (0 hits): it reads the KV tensor's 784 tokens per block as 784 heads.
#
# Rust frontend notes (measured 11-16, doc/design/bench-ttft-hit95-queue.zh-CN.md):
#   - vllm-rs refuses the model's chat_template.jinja ("unknown statement
#     generation"), so every setting gets --chat-template with a copy without
#     its {% generation %} / {% endgeneration %} lines. The bench sends
#     /v1/completions, which never renders a chat template.
#   - vllm-rs tokenizes with fastokens: the bench's random-token prompts came
#     out 4% shorter than with the Python frontend (HF tokenizers). These
#     results compare with each other, not with the Python-frontend runs.
#
# Copying results back: b300 cannot reach the 6000d host, so this script cannot
# push. After every setting it writes <setting>/SETTING_DONE (exit status and
# time), after the last one ALL_DONE; tests/bench_dev/pull-b300-bench-results.sh
# on the 6000d host copies each finished setting back once.
#
# Runs on the HOST like test-qwen38-27B.sh. Results:
# bench-results/Qwen3.8-27B_tp4_rust_<time>/<setting>/ (summary.csv, run.log,
# <config>/serve.log, SETTING_DONE).
# Print the plan without starting anything: DRY_RUN=1 bash test-qwen38-27B-rust.sh
# Other settings or order: SETTINGS="vllm-prefix vllm-wo-prefix" bash test-qwen38-27B-rust.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

MODEL=/home/intel/model-space/Qwen3.8-27B
CHAT_TEMPLATE=$REPO_ROOT/tests/bench_dev/qwen38-27B-chat-template-no-generation.jinja
SETTINGS=${SETTINGS:-"kvshrink-zip-dsa lmcache vllm-prefix kvshrink-nozip-dsa kvshrink-nozip-nodsa vllm-wo-prefix kvshrink-zip-dsa-flash-attn"}
HIT_RATES=${HIT_RATES:-"80 95"}
DRY_RUN=${DRY_RUN:-0}
# One setting took 3.5-4.6 h with the Python frontend (11-11 to 11-13).
SETTING_TIMEOUT_S=25200
KVSHRINK_CONTAINER=$(sed -n 's/^CONTAINER_NAME=//p' setvars.sh)
# tmux session names cannot contain '.'.
KVSHRINK_TMUX_SESSION=${KVSHRINK_CONTAINER//./-}
LMCACHE_IMAGE=qiuyu/vllm-openai-lmcache:v0.29.0-lmcache0.5.7
LMCACHE_CONTAINER=iaxl.lmcache-bench
MODEL_TAG=$(basename "$MODEL")
RUN_NAME=${MODEL_TAG}_tp4_rust_$(date +%Y%m%d_%H%M%S)
if [ "$DRY_RUN" = "1" ]; then
    RESULTS_ROOT=tests/bench_dev/results/dryrun-$RUN_NAME
else
    RESULTS_ROOT=bench-results/$RUN_NAME
fi
# Same GPU / NUMA / accelerator binding as test-qwen38-27B.sh (see there).
ACCEL_ENV="TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,2,4,6"
ACCEL_ENV+=" VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'"
ACCEL_ENV+=" KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'"
ACCEL_ENV+=" KVSHRINK_IAA_DEVICES='0|1|2|3'"
KVSHRINK_START_ENV=$ACCEL_ENV
# 32000 at 300 shows the same as at 256; at 128000 and 256000 the higher
# concurrency (closer to vLLM's maximum of 94 and 48) is kept.
LEN_CONCURRENCY=${LEN_CONCURRENCY:-"32000:32,64,128,256 128000:80 256000:40"}
# Pools as in test-qwen38-27B.sh: 1000 GB KVShrink DDR pool (250 GB per rank)
# and LMCache CPU cache, 8 GB ScratchPool per rank.
BENCH_ENV="MODEL=$MODEL $ACCEL_ENV"
BENCH_ENV+=" LEN_CONCURRENCY='$LEN_CONCURRENCY' HIT_RATES='$HIT_RATES'"
BENCH_ENV+=" MAX_MODEL_LEN=256128"
BENCH_ENV+=" IAXL_SCRATCH_POOL_SIZE_GB=8 IAXL_DDR_POOL_TOTAL_SIZE_GB=1000 LMCACHE_L1_SIZE_GB=1000"
# Requests per run by concurrency, for a smoke test.
if [ -n "${NUM_PROMPTS_BY_CONCURRENCY:-}" ]; then
    BENCH_ENV+=" NUM_PROMPTS_BY_CONCURRENCY='$NUM_PROMPTS_BY_CONCURRENCY'"
fi
BENCH_ENV+=" VLLM_USE_RUST_FRONTEND=1 VLLM_SERVE_EXTRA_ARGS='--chat-template $CHAT_TEMPLATE'"
BENCH_ENV+=" DRY_RUN=$DRY_RUN"

# kvshrink-benchmark.sh config of one setting.
setting_config() {
    case "$1" in
        kvshrink-zip-dsa|kvshrink-nozip-dsa|kvshrink-nozip-nodsa|kvshrink-zip-dsa-flash-attn|kvshrink-nozip-dsa-omp12) echo "kvshrink" ;;
        vllm-prefix|vllm-wo-prefix|lmcache) echo "$1" ;;
        *) echo "Unknown setting: $1" >&2; return 1 ;;
    esac
}

# kvshrink switches of one setting (read by setvars.sh in the serve script).
setting_env() {
    case "$1" in
        kvshrink-zip-dsa|kvshrink-zip-dsa-flash-attn) echo "IAXL_KV_COMPRESSION=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1" ;;
        kvshrink-nozip-dsa) echo "IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1" ;;
        kvshrink-nozip-nodsa) echo "IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0" ;;
        kvshrink-nozip-dsa-omp12) echo "IAXL_KV_COMPRESSION=0 IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=0 IAXL_CPU_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=1 IAXL_OMP_THREAD_NUM=12" ;;
        *) echo "" ;;
    esac
}

# Attention backend of one setting; empty keeps vLLM's default (FlashInfer).
# vLLM 0.29 picks block size 896 at TP=4 under FLASH_ATTN (measured 11-12),
# and the LMCache chunk size must equal it.
setting_backend_env() {
    case "$1" in
        lmcache|kvshrink-zip-dsa-flash-attn) echo "ATTENTION_BACKEND=FLASH_ATTN LMCACHE_CHUNK_SIZE=896" ;;
        *) echo "" ;;
    esac
}

for setting in $SETTINGS; do
    setting_config "$setting" > /dev/null || exit 1
done

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/test-qwen38-27B-rust.log") 2>&1
cp "${BASH_SOURCE[0]}" "$CHAT_TEMPLATE" "$RESULTS_ROOT/"

# check_host_settings, reclaim_host_memory, ensure_kvshrink_container.
source tools/bench-host-setup.sh

# One kvshrink-benchmark.sh call for one setting. timeout runs inside the
# container: on expiry it signals kvshrink-benchmark.sh and its bench client,
# and kvshrink-benchmark.sh's trap stops the serve.
run_setting() {
    local setting=$1 setting_dir=$2
    local bench_cmd="CONFIGS=$(setting_config "$setting") RESULTS_DIR=$setting_dir $BENCH_ENV $(setting_env "$setting") $(setting_backend_env "$setting")"
    bench_cmd+=" timeout $SETTING_TIMEOUT_S bash kvshrink-benchmark.sh"
    if [ "$setting" = "lmcache" ]; then
        docker run --rm --name "$LMCACHE_CONTAINER" --gpus all --network host --ipc host \
            -v "$REPO_ROOT:$REPO_ROOT" -v /home/intel/model-space:/home/intel/model-space -w "$REPO_ROOT" \
            --entrypoint bash "$LMCACHE_IMAGE" -c "$bench_cmd"
    else
        docker exec -w "$REPO_ROOT" "$KVSHRINK_CONTAINER" bash -lc "$bench_cmd"
    fi
}

echo "==========================================================="
echo "Qwen3.8-27B Rust-frontend test  settings=[$SETTINGS]"
echo "  $BENCH_ENV"
echo "  KVShrink container=$KVSHRINK_CONTAINER  LMCache image=$LMCACHE_IMAGE"
echo "  timeout per setting ${SETTING_TIMEOUT_S}s"
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
    setting_dir="$RESULTS_ROOT/$setting"
    mkdir -p "$setting_dir"
    echo ""
    echo "### $setting ($(date +'%F %T')): config $(setting_config "$setting") $(setting_env "$setting") $(setting_backend_env "$setting") -> $setting_dir"
    if [ "$DRY_RUN" != "1" ]; then
        reclaim_host_memory
    fi
    run_setting "$setting" "$setting_dir"
    status=$?
    if [ "$status" != "0" ]; then
        echo "ERROR: $setting exited with status $status" >&2
        failed+=" $setting"
    fi
    echo "status=$status end=$(date +'%F %T')" > "$setting_dir/SETTING_DONE"
done

echo ""
echo "Done ($(date +'%F %T')): $RESULTS_ROOT/<setting>/summary.csv"
echo "end=$(date +'%F %T') failed=[${failed# }]" > "$RESULTS_ROOT/ALL_DONE"
if [ -n "$failed" ]; then
    echo "FAILED:$failed" >&2
    exit 1
fi
