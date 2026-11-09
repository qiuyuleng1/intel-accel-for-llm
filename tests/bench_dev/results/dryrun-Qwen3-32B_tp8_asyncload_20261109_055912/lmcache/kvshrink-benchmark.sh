#!/bin/bash
#
# Qwen3.8-27B serving benchmark: KVShrink and LMCache against two plain-vLLM
# baselines.
#
#   kvshrink        KVShrink: QAT + IAA compression, DSA transfers, every layer
#                   compressed, vLLM prefix caching off. Hit rates 80 / 95 / 100.
#   lmcache         LMCache MP mode (standalone `lmcache server` + LMCacheMPConnector),
#                   CPU cache LMCACHE_L1_SIZE_GB, vLLM prefix caching off.
#                   Hit rates 80 / 95 / 100. Runs in the vLLM 0.33.0 + LMCache 0.5.6
#                   image (examples/lmcache-vllm-serve.sh), see below.
#   vllm-prefix     plain vLLM with its GPU prefix cache (upper reference).
#                   Hit rates 80 / 95 / 100.
#   vllm-prefix-v033  vllm-prefix on the LMCache image's vLLM 0.33.0, no LMCache
#                   server (examples/lmcache-vllm-serve.sh BASELINE=prefix_cache):
#                   the reference for lmcache on the same vLLM version.
#   vllm-wo-prefix  plain vLLM without any KV reuse. Hit rate 80 only: without
#                   reuse every hit rate does the same work. Only concurrency * 2
#                   requests and no prewarm: it prefills one request at a time
#                   whatever the concurrency, so its throughput settles within a
#                   few requests and it has no cache to warm. Its TTFT is mostly
#                   queueing and depends on the request count, so it is not
#                   comparable with the others; its throughput is.
#
# For every config it starts the config's serve script once with
# MAX_MODEL_LEN=201128 (serve log written live into the result dir). For every
# input length it then runs one prewarm (hit rate 100, largest concurrency * 2
# requests, not recorded; skipped for vllm-wo-prefix) and one `vllm bench serve`
# per concurrency and hit rate with concurrency * 10 requests (concurrency * 2
# for vllm-wo-prefix). It stops the serve after the last input length and exits
# as soon as a serve fails to start or dies.
#
# No concurrency may exceed vLLM's "Maximum concurrency" for its input length
# at that TP size (tests/bench_dev/probe-max-concurrency.sh).
#
# Runs INSIDE a container. The container exports TP_SIZE=8, so pass TP_SIZE,
# CUDA_VISIBLE_DEVICES and LEN_CONCURRENCY explicitly. kvshrink, vllm-prefix
# and vllm-wo-prefix run in the KVShrink container; lmcache and
# vllm-prefix-v033 each run alone in a container of
# qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6. Give both runs
# the same RESULTS_DIR to collect one summary.csv. TP=8:
#   CONFIGS="kvshrink vllm-prefix vllm-wo-prefix" RESULTS_DIR=bench-results/<name> \
#       TP_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#       LEN_CONCURRENCY="100000:8,16,32 200000:8,16" bash kvshrink-benchmark.sh
#   CONFIGS=lmcache RESULTS_DIR=bench-results/<name> ... (same as above)
# Print the plan without starting anything: add DRY_RUN=1.
#
# Optional overrides (unset = the defaults above): MODEL, MAX_MODEL_LEN,
# HIT_RATES ("80 100"; vllm-wo-prefix stays at 80), NUM_PROMPTS_BY_CONCURRENCY
# ("1:20 2:80 4:96": requests per run by concurrency, for every config),
# BLOCK_SIZE (passed to vLLM as --block-size by both serve scripts),
# LMCACHE_CHUNK_SIZE (must equal the vLLM block size),
# IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS (leading layers kvshrink stores uncompressed),
# IAXL_DDR_POOL_TOTAL_SIZE_GB, IAXL_SCRATCH_POOL_SIZE_GB and LMCACHE_L1_SIZE_GB.
set -o pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
: "${TP_SIZE:?set TP_SIZE}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the first TP_SIZE GPUs}"
: "${LEN_CONCURRENCY:?set LEN_CONCURRENCY as input_len:concurrencies entries, e.g. \"100000:8,16,32 200000:8,16\"}"
: "${MODEL:?MODEL must be exported by the container}"
export TP_SIZE CUDA_VISIBLE_DEVICES
export PORT=${PORT:-8000}

CONFIGS=${CONFIGS:-"kvshrink vllm-prefix vllm-wo-prefix"}
OUTPUT_LEN=128
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-201128}
PREWARM_PROMPTS_PER_CONCURRENCY=2
NUM_WARMUPS=5
BASE_SEED=10000
EVICT_THRESHOLD_MS=10
SCRATCH_WAIT_THRESHOLD_MS=10
WORKER_STATUS_URL=http://localhost:${IAXL_API_WORKER_BASE_PORT:-18800}/v1/cache/simple_status
SERVE_READY_TIMEOUT=1800
SERVE_STOP_TIMEOUT=180
SERVE_STOP_SETTLE=15
DRY_RUN=${DRY_RUN:-0}

# kvshrink feature switches, read by setvars.sh inside the serve script. The
# baselines load no KVShrink connector, so these do not affect them.
export IAXL_DSA_GD_ENABLE=1
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_IAA_ZIP_ENABLE=1
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-0}
# setvars.sh defaults the DDR pool to 80% of host RAM (~400 GB) and fills it at
# startup; that left too little memory for QAT's DMA buffers and the serve crashed.
# TP=8 with a 64 GB pool and 16 GB scratch peaked at 342.5 GB host memory.
# 80 GB = 10 GB per rank at TP=8.
export IAXL_DDR_POOL_TOTAL_SIZE_GB=${IAXL_DDR_POOL_TOTAL_SIZE_GB:-80}
# Pinned staging per rank. At 16 GB (TP=8, 100k, 32 simultaneous hit-100
# loads) each rank waited ~3 s in total against a ~27 s mean TTFT.
export IAXL_SCRATCH_POOL_SIZE_GB=${IAXL_SCRATCH_POOL_SIZE_GB:-24}

# lmcache config, read by examples/lmcache-vllm-serve.sh. The LMCache chunk size
# must equal the vLLM block size, which vLLM reports only after it starts:
# 448 tokens measured for Qwen3.8-27B at TP=8 on vLLM 0.33.0 with prefix
# caching off (tests/bench_dev/results/lmcache-probe-tp8-20261104_074103).
# The run exits if vLLM's block size differs. The CPU cache matches the
# KVShrink DDR pool size.
export LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-448}
export LMCACHE_L1_SIZE_GB=${LMCACHE_L1_SIZE_GB:-80}
LMCACHE_VERSIONS="vllm 0.33.0 lmcache 0.5.6"

# BASELINE value for examples/kvshrink-vllm-serve.sh; empty selects KVShrink.
# lmcache uses its own serve script, which ignores BASELINE.
config_baseline() {
    case "$1" in
        kvshrink) echo "" ;;
        lmcache) echo "" ;;
        vllm-prefix) echo "prefix_cache" ;;
        vllm-prefix-v033) echo "prefix_cache" ;;
        vllm-wo-prefix) echo "no_cache" ;;
        *) echo "Unknown config: $1 (expected kvshrink, lmcache, vllm-prefix, vllm-prefix-v033 or vllm-wo-prefix)" >&2; return 1 ;;
    esac
}

config_serve_script() {
    case "$1" in
        kvshrink|vllm-prefix|vllm-wo-prefix) echo "examples/kvshrink-vllm-serve.sh" ;;
        lmcache|vllm-prefix-v033) echo "examples/lmcache-vllm-serve.sh" ;;
        *) echo "Unknown config: $1 (expected kvshrink, lmcache, vllm-prefix, vllm-prefix-v033 or vllm-wo-prefix)" >&2; return 1 ;;
    esac
}

config_hit_rates() {
    case "$1" in
        kvshrink|lmcache|vllm-prefix|vllm-prefix-v033) echo "${HIT_RATES:-80 95 100}" ;;
        vllm-wo-prefix) echo "80" ;;
        *) echo "Unknown config: $1 (expected kvshrink, lmcache, vllm-prefix, vllm-prefix-v033 or vllm-wo-prefix)" >&2; return 1 ;;
    esac
}

# num_prompts = concurrency * this.
config_prompts_per_concurrency() {
    case "$1" in
        kvshrink|lmcache|vllm-prefix|vllm-prefix-v033) echo 10 ;;
        vllm-wo-prefix) echo 2 ;;
        *) echo "Unknown config: $1 (expected kvshrink, lmcache, vllm-prefix, vllm-prefix-v033 or vllm-wo-prefix)" >&2; return 1 ;;
    esac
}

# Requests per run: from NUM_PROMPTS_BY_CONCURRENCY when it is set (same for
# every config), else concurrency * config_prompts_per_concurrency.
num_prompts_for() {
    local config=$1 concurrency=$2
    if [ -z "${NUM_PROMPTS_BY_CONCURRENCY:-}" ]; then
        echo $((concurrency * $(config_prompts_per_concurrency "$config")))
        return 0
    fi
    local pair
    for pair in $NUM_PROMPTS_BY_CONCURRENCY; do
        if [ "${pair%%:*}" = "$concurrency" ]; then
            echo "${pair##*:}"
            return 0
        fi
    done
    echo "ERROR: concurrency $concurrency is not in NUM_PROMPTS_BY_CONCURRENCY=\"$NUM_PROMPTS_BY_CONCURRENCY\"" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Validate everything before starting any serve
# ---------------------------------------------------------------------------
# setvars.sh binds CPU/QAT/DSA to the first TP_SIZE GPUs listed by nvidia-smi,
# so exactly TP_SIZE GPUs must be visible.
IFS=, read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#VISIBLE_GPUS[@]}" -ne "$TP_SIZE" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has ${#VISIBLE_GPUS[@]} GPUs, TP_SIZE=$TP_SIZE" >&2
    exit 1
fi
for pair in $LEN_CONCURRENCY; do
    if ! [[ "$pair" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
        echo "ERROR: invalid LEN_CONCURRENCY entry '$pair' (expected input_len:concurrency[,concurrency...])" >&2
        exit 1
    fi
done
for config in $CONFIGS; do
    config_baseline "$config" > /dev/null || exit 1
    for pair in $LEN_CONCURRENCY; do
        concurrencies=${pair##*:}
        for concurrency in ${concurrencies//,/ }; do
            num_prompts_for "$config" "$concurrency" > /dev/null || exit 1
        done
    done
done
# lmcache and vllm-prefix-v033 run in the vLLM 0.33.0 + LMCache 0.5.6 image, the
# other configs in the KVShrink container, so one run cannot mix them.
config_count=0
lmcache_image_requested=0
for config in $CONFIGS; do
    config_count=$((config_count + 1))
    if [ "$config" = "lmcache" ] || [ "$config" = "vllm-prefix-v033" ]; then
        lmcache_image_requested=1
    fi
done
if [ "$lmcache_image_requested" = "1" ] && [ "$config_count" -gt 1 ]; then
    echo "ERROR: run CONFIGS=lmcache or CONFIGS=vllm-prefix-v033 alone, in a qiuyu/vllm-openai-lmcache:v0.33.0-lmcache0.5.6 container" >&2
    exit 1
fi
if [ "$DRY_RUN" != "1" ]; then
    if ! command -v vllm > /dev/null 2>&1; then
        echo "ERROR: vllm not found; run this inside the container" >&2
        exit 1
    fi
    if [ "$lmcache_image_requested" = "1" ]; then
        installed_versions=$(python3 -c 'from importlib.metadata import version; print("vllm", version("vllm"), "lmcache", version("lmcache"))')
        if [ "$installed_versions" != "$LMCACHE_VERSIONS" ]; then
            echo "ERROR: $CONFIGS needs $LMCACHE_VERSIONS, this container has $installed_versions" >&2
            exit 1
        fi
    fi
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "ERROR: something already serves on port $PORT; stop it first" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
MODEL_TAG=$(basename "$MODEL")
RESULTS_DIR=${RESULTS_DIR:-bench-results/${MODEL_TAG}_tp${TP_SIZE}_$(date +'%Y%m%d_%H%M%S')}
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1
SUMMARY_CSV="$RESULTS_DIR/summary.csv"

# lmcache and vllm-prefix-v033 use the released vLLM 0.33.0 + LMCache 0.5.6, so
# their runs record no repo state; they must not overwrite what the KVShrink run
# sharing this RESULTS_DIR wrote (the LMCache image has no git either).
if [ "$lmcache_image_requested" != "1" ]; then
    git rev-parse HEAD > "$RESULTS_DIR/git_commit.txt"
    git status --short > "$RESULTS_DIR/git_status.txt"
    git diff > "$RESULTS_DIR/git_diff.patch"
fi
cp "${BASH_SOURCE[0]}" "$RESULTS_DIR/"
md5sum "${BASH_SOURCE[0]}" examples/kvshrink-vllm-serve.sh examples/lmcache-vllm-serve.sh setvars.sh \
    kvshrink/*.py iaxl/kvstore/*.py iaxl/kvflow/*.py > "$RESULTS_DIR/code_md5.txt"

if [ ! -f "$SUMMARY_CSV" ]; then
    echo "config,tp_size,input_len_tokens,output_len_tokens,max_concurrency,hit_rate_pct,num_prompts,seed,block_size_tokens,max_concurrency_reported,completed,duration_s,request_throughput_req_s,output_throughput_tok_s,total_token_throughput_tok_s,mean_ttft_ms,median_ttft_ms,p95_ttft_ms,mean_tpot_ms,median_tpot_ms,p95_tpot_ms,preemptions,gpu_prefix_cache_hit_tokens,gpu_prefix_cache_query_tokens,external_prefix_cache_hit_tokens,kvshrink_requests_seen,kvshrink_hit_requests,kvshrink_mean_hit_tokens,ddr_used_gb_before,ddr_used_gb_after,ddr_capacity_gb,compression_ratio,evict_count,evict_over_${EVICT_THRESHOLD_MS}ms_count,evict_max_ms,evict_total_ms,dsa_batch_failed_count,scratch_wait_count,scratch_wait_over_${SCRATCH_WAIT_THRESHOLD_MS}ms_count,scratch_wait_max_ms,scratch_wait_total_ms" > "$SUMMARY_CSV"
fi

# ---------------------------------------------------------------------------
# Serve lifecycle
# ---------------------------------------------------------------------------
SERVE_PID=""

start_serve() {
    local serve_script=$1 serve_log=$2
    # Own process group, so stop_serve cannot touch any other vLLM.
    setsid bash "$serve_script" > "$serve_log" 2>&1 &
    SERVE_PID=$!
    echo "    serve pid=$SERVE_PID log=$serve_log"
}

serve_alive() {
    kill -0 "$SERVE_PID" 2> /dev/null
}

# Once vLLM is up the serve script's PID is not enough: it waits for
# `vllm serve | tee`, and a hung vLLM worker keeps tee's pipe open after the
# API server exited (Nov 5: 8 benchmarks ran against a dead server). /health
# returns 503 once the engine is dead and refuses connections once the API
# server is gone.
serve_healthy() {
    curl -sf -m 10 "http://localhost:$PORT/health" > /dev/null 2>&1
}

wait_for_serve() {
    local waited=0
    while (( waited < SERVE_READY_TIMEOUT )); do
        if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "    serve ready after ${waited}s"
            return 0
        fi
        if ! serve_alive; then
            echo "ERROR: serve exited before becoming ready" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "ERROR: serve not ready within ${SERVE_READY_TIMEOUT}s" >&2
    return 1
}

stop_serve() {
    if [ -z "$SERVE_PID" ]; then
        return 0
    fi
    kill -TERM -- "-$SERVE_PID" 2> /dev/null
    local waited=0
    while (( waited < SERVE_STOP_TIMEOUT )) && kill -0 -- "-$SERVE_PID" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PID" 2> /dev/null
    SERVE_PID=""
    sleep "$SERVE_STOP_SETTLE"
    echo "    serve stopped after ${waited}s + ${SERVE_STOP_SETTLE}s settle"
}
trap stop_serve EXIT INT TERM

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
# Last number of a serve-log line; the log holds '\r' progress bars, so split
# on them before matching.
serve_log_number() {
    local serve_log=$1 pattern=$2
    tr '\r' '\n' < "$serve_log" | grep -aoE "$pattern" | tail -1 | awk '{print $NF}'
}

# Sum of a vLLM Prometheus counter over all its label sets.
vllm_counter() {
    local name=$1
    curl -sf --max-time 15 "http://localhost:$PORT/metrics" \
        | awk -v name="$name" 'index($1, name "{") == 1 || $1 == name {sum += $NF} END {printf "%.0f\n", sum}'
}

# Rank 0 DDR pool: "<current_bytes> <capacity_bytes> <total_unzip_bytes> <total_zip_bytes>".
ddr_status() {
    local json
    if ! json=$(curl -sf --max-time 15 "$WORKER_STATUS_URL"); then
        echo "na na na na"
        return 0
    fi
    printf '%s' "$json" | python3 -c '
import json
import sys

status = json.load(sys.stdin)
print(status["current_bytes"], status["capacity_bytes"],
      status["total_unzip_bytes"], status["total_zip_bytes"])
'
}

bytes_to_gb() {
    if [ "$1" = "na" ]; then
        echo "na"
        return 0
    fi
    awk -v bytes="$1" 'BEGIN {printf "%.2f\n", bytes / 1024 / 1024 / 1024}'
}

# Serve-log bytes [start, end), i.e. what one bench run appended.
serve_log_slice() {
    local serve_log=$1 start=$2 end=$3
    tail -c +$((start + 1)) "$serve_log" | head -c $((end - start)) | tr '\r' '\n'
}

# "<requests_seen> <hit_requests> <mean_hit_tokens>" from the connector's
# get_num_new_matched_tokens lines on stdin. The scheduler may query a waiting
# request several times, so only each request's last value counts.
kvshrink_hit_stats() {
    awk '
        /get_num_new_matched_tokens, req-/ {
            request = $0
            sub(/.*get_num_new_matched_tokens, req-/, "", request)
            sub(/,.*/, "", request)
            tokens = $0
            sub(/.*externally-cached tokens: /, "", tokens)
            last_tokens[request] = tokens + 0
        }
        END {
            seen = 0
            hits = 0
            hit_tokens = 0
            for (request in last_tokens) {
                seen++
                if (last_tokens[request] > 0) {
                    hits++
                    hit_tokens += last_tokens[request]
                }
            }
            if (hits > 0) {
                printf "%d %d %.0f\n", seen, hits, hit_tokens / hits
            } else {
                printf "%d 0 0\n", seen
            }
        }'
}

# "<count> <over_threshold_count> <max_ms> <total_ms>" from the
# "[evict] triggered: ... in X.XXXs" lines on stdin. Evictions slower than the
# threshold are copied verbatim to over_log.
evict_stats() {
    local over_log=$1
    awk -v threshold="$EVICT_THRESHOLD_MS" -v over_log="$over_log" '
        /\[evict\] triggered:/ {
            seconds = $0
            sub(/.* in /, "", seconds)
            ms = (seconds + 0) * 1000
            count++
            total_ms += ms
            if (ms > max_ms) {
                max_ms = ms
            }
            if (ms > threshold) {
                over++
                print $0 > over_log
            }
        }
        END {printf "%d %d %.0f %.0f\n", count, over, max_ms, total_ms}'
}

# Count of "dsa batch failed" lines on stdin, which are copied verbatim to
# fail_log.
dsa_fail_stats() {
    local fail_log=$1
    awk -v fail_log="$fail_log" '
        /dsa batch failed/ {
            count++
            print $0 > fail_log
        }
        END {printf "%d\n", count}'
}

# "<count> <over_threshold_count> <max_ms> <total_ms>" from the
# "ScratchPool exhausted: waited X ms" lines on stdin, summed over all TP
# ranks. Waits longer than the threshold are copied verbatim to over_log.
scratch_wait_stats() {
    local over_log=$1
    awk -v threshold="$SCRATCH_WAIT_THRESHOLD_MS" -v over_log="$over_log" '
        /ScratchPool exhausted: waited/ {
            ms = $0
            sub(/.*waited /, "", ms)
            ms = ms + 0
            count++
            total_ms += ms
            if (ms > max_ms) {
                max_ms = ms
            }
            if (ms > threshold) {
                over++
                print $0 > over_log
            }
        }
        END {printf "%d %d %.1f %.1f\n", count, over, max_ms, total_ms}'
}

# CSV fields from a `vllm bench serve --save-result` JSON, "FAIL" for each if
# the run left no result.
bench_fields() {
    python3 - "$1" << 'PY'
import json
import sys

FIELDS = [
    "completed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
]

try:
    with open(sys.argv[1]) as f:
        result = json.load(f)
except (OSError, ValueError):
    result = {}

values = []
for field in FIELDS:
    value = result.get(field, "FAIL")
    if isinstance(value, float):
        value = round(value, 3)
    values.append(str(value))
print(",".join(values))
PY
}

# One `vllm bench serve`; prefix_len tokens are shared by every request, the
# remaining input tokens are random per request.
bench_serve() {
    local input_len=$1 hit_rate=$2 concurrency=$3 num_prompts=$4 num_warmups=$5 seed=$6
    shift 6
    local prefix_len=$((input_len * hit_rate / 100))
    local random_len=$((input_len - prefix_len))
    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --dataset-name random \
        --host localhost \
        --port "$PORT" \
        --random-input-len "$random_len" \
        --random-prefix-len "$prefix_len" \
        --random-output-len "$OUTPUT_LEN" \
        --ignore-eos \
        --percentile-metrics "ttft,tpot" \
        --metric-percentiles "50,95" \
        --seed "$seed" \
        --trust-remote-code \
        --request-rate inf \
        --num-warmups "$num_warmups" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$concurrency" \
        "$@"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "==========================================================="
echo "$MODEL_TAG benchmark  configs=[$CONFIGS]"
echo "  MODEL=$MODEL TP_SIZE=$TP_SIZE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  input_len:concurrencies=[$LEN_CONCURRENCY] output_len=$OUTPUT_LEN MAX_MODEL_LEN=$MAX_MODEL_LEN BLOCK_SIZE=${BLOCK_SIZE:-<vLLM default>}"
if [ -n "${NUM_PROMPTS_BY_CONCURRENCY:-}" ]; then
    prompts_plan="by concurrency [$NUM_PROMPTS_BY_CONCURRENCY]"
else
    prompts_plan="concurrency*10 (vllm-wo-prefix: *2)"
fi
echo "  num_prompts=$prompts_plan warmups=$NUM_WARMUPS prewarm=largest concurrency*$PREWARM_PROMPTS_PER_CONCURRENCY at hit 100 (vllm-wo-prefix: none)"
echo "  IAXL_DDR_POOL_TOTAL_SIZE_GB=$IAXL_DDR_POOL_TOTAL_SIZE_GB IAXL_SCRATCH_POOL_SIZE_GB=$IAXL_SCRATCH_POOL_SIZE_GB"
echo "  LMCACHE_CHUNK_SIZE=$LMCACHE_CHUNK_SIZE LMCACHE_L1_SIZE_GB=$LMCACHE_L1_SIZE_GB"
echo "  results -> $RESULTS_DIR"
echo "==========================================================="

for config in $CONFIGS; do
    baseline=$(config_baseline "$config")
    serve_script=$(config_serve_script "$config")
    hit_rates=$(config_hit_rates "$config")
    export BASELINE=$baseline
    serve_log="$RESULTS_DIR/$config/serve.log"
    # Read only by examples/lmcache-vllm-serve.sh.
    export LMCACHE_SERVER_LOG="$RESULTS_DIR/$config/lmcache_server.log"
    mkdir -p "$RESULTS_DIR/$config"

    echo ""
    echo "### $config serve=$serve_script BASELINE=$baseline MAX_MODEL_LEN=$MAX_MODEL_LEN hit_rates=[$hit_rates]"

    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] serve: BASELINE=$baseline MAX_MODEL_LEN=$MAX_MODEL_LEN BLOCK_SIZE=${BLOCK_SIZE:-} bash $serve_script > $serve_log"
    else
        start_serve "$serve_script" "$serve_log"
        if ! wait_for_serve; then
            exit 1
        fi
        block_size=$(serve_log_number "$serve_log" "Setting attention block size to [0-9]+")
        # vLLM logs the block size only when it picks one for a hybrid model;
        # otherwise it uses --block-size, which the serve scripts pass from BLOCK_SIZE.
        if [ -z "$block_size" ]; then
            block_size=${BLOCK_SIZE:-}
        fi
        kv_cache_tokens=$(serve_log_number "$serve_log" "GPU KV cache size: [0-9,]+" | tr -d ,)
        echo "    block_size=$block_size kv_cache_tokens=$kv_cache_tokens"
        if [ "$config" = "lmcache" ] && [ "$block_size" != "$LMCACHE_CHUNK_SIZE" ]; then
            echo "ERROR: vLLM block size $block_size != LMCACHE_CHUNK_SIZE $LMCACHE_CHUNK_SIZE; measure it and update LMCACHE_CHUNK_SIZE" >&2
            exit 1
        fi
    fi

    for pair in $LEN_CONCURRENCY; do
        input_len=${pair%%:*}
        concurrencies=${pair##*:}
        concurrencies=${concurrencies//,/ }
        max_concurrency=0
        for concurrency in $concurrencies; do
            if [ "$concurrency" -gt "$max_concurrency" ]; then
                max_concurrency=$concurrency
            fi
        done
        prewarm_prompts=$((max_concurrency * PREWARM_PROMPTS_PER_CONCURRENCY))
        runs=""
        for concurrency in $concurrencies; do
            for hit_rate in $hit_rates; do
                runs+=" $concurrency:$hit_rate"
            done
        done
        run_dir="$RESULTS_DIR/$config/in$input_len"
        mkdir -p "$run_dir"

        echo ""
        echo "## $config input_len=$input_len concurrencies=[$concurrencies]"

        if [ "$DRY_RUN" = "1" ]; then
            if [ "$config" != "vllm-wo-prefix" ]; then
                echo "    [dry-run] prewarm: hit 100, $prewarm_prompts prompts, concurrency $max_concurrency, seed $BASE_SEED"
            fi
            for run in $runs; do
                concurrency=${run%%:*}
                hit_rate=${run##*:}
                echo "    [dry-run] bench: concurrency $concurrency, hit $hit_rate, $(num_prompts_for "$config" "$concurrency") prompts, seed $((BASE_SEED + concurrency + hit_rate))"
            done
            continue
        fi

        # vLLM's "Maximum concurrency" is GPU KV cache tokens / tokens per request.
        max_concurrency_reported=$(awk -v kv="$kv_cache_tokens" -v len=$((input_len + OUTPUT_LEN)) \
            'BEGIN {if (kv > 0) printf "%.2f\n", kv / len}')
        echo "    max_concurrency for $((input_len + OUTPUT_LEN)) tokens per request: $max_concurrency_reported"
        if [ -z "$max_concurrency_reported" ] || [ "$max_concurrency" -gt "${max_concurrency_reported%%.*}" ]; then
            echo "    WARNING: concurrency $max_concurrency exceeds vLLM's maximum ($max_concurrency_reported)"
        fi

        # vllm-wo-prefix has no cache to warm.
        if [ "$config" != "vllm-wo-prefix" ]; then
            echo "    prewarm: hit 100, $prewarm_prompts prompts, concurrency $max_concurrency"
            bench_serve "$input_len" 100 "$max_concurrency" "$prewarm_prompts" 0 "$BASE_SEED" \
                > "$run_dir/prewarm.log" 2>&1
            if ! serve_healthy; then
                echo "ERROR: serve died during prewarm; first errors in $serve_log:" >&2
                tr '\r' '\n' < "$serve_log" | grep -aE "ERROR|CRITICAL|check failed" | head -5 >&2
                exit 1
            fi
        fi

        for run in $runs; do
            concurrency=${run%%:*}
            hit_rate=${run##*:}
            num_prompts=$(num_prompts_for "$config" "$concurrency")
            seed=$((BASE_SEED + concurrency + hit_rate))
            tag="${config}_tp${TP_SIZE}_in${input_len}_con${concurrency}_hit${hit_rate}"
            echo "--- $tag prompts=$num_prompts seed=$seed"

            preemptions_before=$(vllm_counter "vllm:num_preemptions_total")
            prefix_hits_before=$(vllm_counter "vllm:prefix_cache_hits_total")
            prefix_queries_before=$(vllm_counter "vllm:prefix_cache_queries_total")
            external_hits_before=$(vllm_counter "vllm:external_prefix_cache_hits_total")
            ddr_before="na na na na"
            if [ "$config" = "kvshrink" ]; then
                ddr_before=$(ddr_status)
            fi
            log_start=$(stat -c %s "$serve_log")

            bench_serve "$input_len" "$hit_rate" "$concurrency" "$num_prompts" "$NUM_WARMUPS" "$seed" \
                --save-result --result-dir "$run_dir" --result-filename "$tag.json" \
                > "$run_dir/$tag.log" 2>&1

            if ! serve_healthy; then
                echo "ERROR: serve died during $tag; first errors in $serve_log:" >&2
                tr '\r' '\n' < "$serve_log" | grep -aE "ERROR|CRITICAL|check failed" | head -5 >&2
                exit 1
            fi
            log_end=$(stat -c %s "$serve_log")
            preemptions=$(( $(vllm_counter "vllm:num_preemptions_total") - preemptions_before ))
            prefix_hits=$(( $(vllm_counter "vllm:prefix_cache_hits_total") - prefix_hits_before ))
            prefix_queries=$(( $(vllm_counter "vllm:prefix_cache_queries_total") - prefix_queries_before ))
            external_hits=$(( $(vllm_counter "vllm:external_prefix_cache_hits_total") - external_hits_before ))

            kvshrink_stats="na na na"
            evict="na na na na"
            dsa_failed="na"
            scratch_wait="na na na na"
            ddr_after="na na na na"
            if [ "$config" = "kvshrink" ]; then
                kvshrink_stats=$(serve_log_slice "$serve_log" "$log_start" "$log_end" | kvshrink_hit_stats)
                evict=$(serve_log_slice "$serve_log" "$log_start" "$log_end" \
                    | evict_stats "$run_dir/${tag}_evict_over_${EVICT_THRESHOLD_MS}ms.log")
                dsa_failed=$(serve_log_slice "$serve_log" "$log_start" "$log_end" \
                    | dsa_fail_stats "$run_dir/${tag}_dsa_batch_failed.log")
                scratch_wait=$(serve_log_slice "$serve_log" "$log_start" "$log_end" \
                    | scratch_wait_stats "$run_dir/${tag}_scratch_wait_over_${SCRATCH_WAIT_THRESHOLD_MS}ms.log")
                ddr_after=$(ddr_status)
            fi
            read -r requests_seen hit_requests mean_hit_tokens <<< "$kvshrink_stats"
            read -r evict_count evict_over evict_max_ms evict_total_ms <<< "$evict"
            read -r scratch_wait_count scratch_wait_over scratch_wait_max_ms scratch_wait_total_ms <<< "$scratch_wait"
            ddr_used_before=${ddr_before%% *}
            read -r ddr_used_after ddr_capacity total_unzip total_zip <<< "$ddr_after"
            compression_ratio="na"
            if [ "$total_zip" != "na" ] && [ "$total_zip" != "0" ]; then
                compression_ratio=$(awk -v unzip="$total_unzip" -v zip="$total_zip" 'BEGIN {printf "%.3f\n", unzip / zip}')
            fi

            fields=$(bench_fields "$run_dir/$tag.json")
            echo "$config,$TP_SIZE,$input_len,$OUTPUT_LEN,$concurrency,$hit_rate,$num_prompts,$seed,$block_size,$max_concurrency_reported,$fields,$preemptions,$prefix_hits,$prefix_queries,$external_hits,$requests_seen,$hit_requests,$mean_hit_tokens,$(bytes_to_gb "$ddr_used_before"),$(bytes_to_gb "$ddr_used_after"),$(bytes_to_gb "$ddr_capacity"),$compression_ratio,$evict_count,$evict_over,$evict_max_ms,$evict_total_ms,$dsa_failed,$scratch_wait_count,$scratch_wait_over,$scratch_wait_max_ms,$scratch_wait_total_ms" >> "$SUMMARY_CSV"
            echo "    $(tail -1 "$SUMMARY_CSV")"
        done
    done

    stop_serve
done

trap - EXIT INT TERM
echo ""
echo "==========================================================="
echo "Done: $SUMMARY_CSV"
echo "==========================================================="
