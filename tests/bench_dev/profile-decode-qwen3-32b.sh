#!/bin/bash
#
# Kernel-level look at full-batch decode steps of Qwen3-32B (TP=8, 48
# requests, 32768-token prompt, hit rate 100): where does the decode step of
# kvshrink default (every request has its own copy of the prompt's KV, 98k
# distinct blocks at 48 requests, Nov 9 step trace) spend the extra ~23 ms that
# kvshrink with vLLM's GPU prefix cache (all requests share one copy, ~2.3k
# blocks) does not?
#
# In the benchmark those steps come in stretches between request arrivals.
# Here all 48 requests arrive at once and decode up to 2048 tokens together, so
# once every request is running nothing arrives or loads any more and every
# step is such a full-batch decode step. vLLM's torch profiler then records
# PROFILER_MAX_ITERATIONS of them (POST /start_profile once rank 0's step trace
# shows all 48 requests decoding and none prefilling); the run stops as soon as
# the profile is written. 2048
# output tokens because LMCache loads only ~1.1 requests/s (Nov 9 benchmark,
# chunk 16), so its 48th request starts decoding ~45 s after its first.
# Everything else matches test-qwen3-32B-async-load.sh: block 16, QAT + IAA
# compression, DSA, ScratchPool 48 GB, DDR pool 40 GB, LMCache chunk 16 and
# CPU cache 40 GB.
#
# Settings:
#   default           kvshrink, every request its own GPU copy (PIECEWISE)
#   gpu-prefix        kvshrink with vLLM's GPU prefix cache, one shared copy
#   vllm-wo-prefix    vLLM 0.29.0 without KV reuse: every request prefills the
#                     prompt itself into its own GPU copy
#   lmcache           LMCache 0.5.6 on vLLM 0.33.0, its own GPU copy per request
#   vllm-prefix-v033  vLLM 0.33.0 with its GPU prefix cache, one shared copy
# default, gpu-prefix and vllm-wo-prefix run INSIDE the KVShrink container;
# lmcache and
# vllm-prefix-v033 inside a container of the LMCache image, where KVShrink is
# not installed: there PYTHONPATH adds the repo and step_trace_plugin/, whose
# dist-info registers kvshrink/step_trace.py as a vLLM plugin.
#   SETTING=<setting> RESULTS_DIR=<dir> bash tests/bench_dev/profile-decode-qwen3-32b.sh
# Writes <dir>/serve.log (with KVSHRINK_STEP_TRACE lines: step periods and, on
# rank 0 every 20 steps, the batch's block references / distinct blocks /
# runs), prewarm.log, wave.log and <dir>/torch-profile/ (traces and one
# CUDA-time table per rank).
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

: "${SETTING:?set SETTING to default, gpu-prefix, vllm-wo-prefix, lmcache or vllm-prefix-v033}"
: "${RESULTS_DIR:?set RESULTS_DIR}"
case "$SETTING" in
    default)
        SERVE_SCRIPT=examples/kvshrink-vllm-serve.sh
        export GPU_PREFIX_CACHING=0
        ;;
    gpu-prefix)
        SERVE_SCRIPT=examples/kvshrink-vllm-serve.sh
        export GPU_PREFIX_CACHING=1
        ;;
    vllm-wo-prefix)
        SERVE_SCRIPT=examples/kvshrink-vllm-serve.sh
        export BASELINE=no_cache
        ;;
    lmcache)
        SERVE_SCRIPT=examples/lmcache-vllm-serve.sh
        export BASELINE=
        ;;
    vllm-prefix-v033)
        SERVE_SCRIPT=examples/lmcache-vllm-serve.sh
        export BASELINE=prefix_cache
        ;;
    *) echo "ERROR: unknown SETTING $SETTING (expected default, gpu-prefix, vllm-wo-prefix, lmcache or vllm-prefix-v033)" >&2; exit 1 ;;
esac
if [ "$SERVE_SCRIPT" = "examples/lmcache-vllm-serve.sh" ]; then
    export LMCACHE_CHUNK_SIZE=16
    export LMCACHE_L1_SIZE_GB=40
    export LMCACHE_SERVER_LOG=$REPO_ROOT/$RESULTS_DIR/lmcache_server.log
    export PYTHONPATH=$REPO_ROOT/tests/bench_dev/step_trace_plugin:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
fi

export MODEL=/media/model-space/Qwen3-32B
export TP_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_MODEL_LEN=40960
export BLOCK_SIZE=16
export PORT=8000
export IAXL_DSA_GD_ENABLE=1
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_IAA_ZIP_ENABLE=1
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0
export IAXL_SCRATCH_POOL_SIZE_GB=48
export IAXL_DDR_POOL_TOTAL_SIZE_GB=40
export KVSHRINK_STEP_TRACE=1
export TORCH_PROFILER_DIR=$REPO_ROOT/$RESULTS_DIR/torch-profile
export PROFILER_MAX_ITERATIONS=20
CONCURRENCY=48
INPUT_LEN=32768
OUTPUT_LEN=2048
SEED=10000
SERVE_READY_TIMEOUT=1800
PROFILE_WRITE_TIMEOUT=300

mkdir -p "$RESULTS_DIR" "$TORCH_PROFILER_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

SERVE_PID=""
stop_serve() {
    if [ -z "$SERVE_PID" ]; then
        return 0
    fi
    kill -TERM -- "-$SERVE_PID" 2> /dev/null
    local waited=0
    while (( waited < 180 )) && kill -0 -- "-$SERVE_PID" 2> /dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PID" 2> /dev/null
    SERVE_PID=""
    sleep 15
    echo "serve stopped"
}
trap stop_serve EXIT INT TERM

# Sum of a vLLM Prometheus gauge over its label sets.
vllm_gauge() {
    curl -sf --max-time 5 "http://localhost:$PORT/metrics" \
        | awk -v name="$1" 'index($1, name "{") == 1 {sum += $NF} END {printf "%.0f\n", sum}'
}

# Tokens served from vLLM's GPU prefix cache and from the KV connector so far.
log_cache_hits() {
    echo "$1: gpu_prefix_cache_hit_tokens=$(vllm_gauge vllm:prefix_cache_hits_total) external_hit_tokens=$(vllm_gauge vllm:external_prefix_cache_hits_total)"
}

# One `vllm bench serve` of hit-100 requests: every prompt is the same
# INPUT_LEN-token prefix (SEED fixes it).
bench_serve() {
    local num_prompts=$1 output_len=$2
    vllm bench serve --backend vllm --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random --host localhost --port "$PORT" \
        --random-input-len 0 --random-prefix-len "$INPUT_LEN" \
        --random-output-len "$output_len" --ignore-eos --seed "$SEED" \
        --trust-remote-code --request-rate inf --num-warmups 0 \
        --num-prompts "$num_prompts" --max-concurrency "$CONCURRENCY"
}

echo "=== profile-decode SETTING=$SETTING serve=$SERVE_SCRIPT -> $RESULTS_DIR ($(date +'%F %T'))"
setsid bash "$SERVE_SCRIPT" > "$RESULTS_DIR/serve.log" 2>&1 &
SERVE_PID=$!
waited=0
until curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2> /dev/null || (( waited >= SERVE_READY_TIMEOUT )); then
        echo "ERROR: serve not ready; first errors in $RESULTS_DIR/serve.log:" >&2
        tr '\r' '\n' < "$RESULTS_DIR/serve.log" | grep -aE "ERROR|CRITICAL|Error" | head -5 >&2
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
echo "serve ready after ${waited}s"

# Stores the prompt: in KVShrink's DDR pool, LMCache's CPU cache or vLLM's GPU
# prefix cache. The KV connectors store after the response has returned, so
# give them time before the wave asks for the prompt.
bench_serve 1 1 > "$RESULTS_DIR/prewarm.log" 2>&1
sleep 10
echo "prewarm done ($(date +'%T'))"
log_cache_hits "before the wave"

bench_serve "$CONCURRENCY" "$OUTPUT_LEN" > "$RESULTS_DIR/wave.log" 2>&1 &
BENCH_PID=$!
# Running / waiting counts are not enough: without KV reuse a request already
# runs while its 32k prompt is prefilled in 8192-token chunks. Rank 0's last
# step-trace line (the step before the current one) showing every request
# decoding and no prefill means no request loads or prefills any more.
full_decode_step="reqs=$CONCURRENCY decode_reqs=$CONCURRENCY prefill_tokens=0 "
until tail -c 200000 "$RESULTS_DIR/serve.log" | tr '\r' '\n' | grep -a "step-trace rank=0 " \
    | tail -1 | grep -q "$full_decode_step"; do
    if ! kill -0 "$BENCH_PID" 2> /dev/null; then
        echo "ERROR: the wave ended before $CONCURRENCY requests decoded together" >&2
        exit 1
    fi
    sleep 0.2
done
curl -sf -X POST "http://localhost:$PORT/start_profile" > /dev/null
echo "start_profile sent ($(date +'%T')); it stops after $PROFILER_MAX_ITERATIONS steps"
log_cache_hits "all $CONCURRENCY running"
# The profiler writes one CUDA-time table per rank when it stops
# (profiler_out_<rank>.txt on vLLM 0.29, rank<rank>.profiler_out.txt on 0.33);
# the rest of the wave is not needed.
waited=0
until [ "$(ls "$TORCH_PROFILER_DIR"/*profiler_out*.txt 2> /dev/null | wc -l)" = "$TP_SIZE" ]; do
    if (( waited >= PROFILE_WRITE_TIMEOUT )); then
        echo "ERROR: profile not written within ${PROFILE_WRITE_TIMEOUT}s" >&2
        kill "$BENCH_PID" 2> /dev/null
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
done
echo "profile written ($(date +'%T'))"
kill "$BENCH_PID" 2> /dev/null
wait "$BENCH_PID" 2> /dev/null
