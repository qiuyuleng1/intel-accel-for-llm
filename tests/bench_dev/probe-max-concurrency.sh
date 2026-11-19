#!/bin/bash
#
# Read the GPU KV capacity that vLLM reports for the KVShrink serve at each
# input length: start examples/kvshrink-vllm-serve.sh with
# MAX_MODEL_LEN = input + output, wait until it serves, copy the capacity lines
# out of its log, then stop it. Sends no requests.
#
# Runs INSIDE the container. The container already exports TP_SIZE from
# setvars.sh (8), so pass TP_SIZE explicitly:
#   TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tests/bench_dev/probe-max-concurrency.sh
# On b300 use probe-max-concurrency-qwen38-b300.sh, which passes the TP=4
# binding of test-qwen38-27B.sh (GPU0/2/4/6).
set -o pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

: "${TP_SIZE:?set TP_SIZE}"
export TP_SIZE
# setvars.sh binds CPU/QAT/DSA to the first TP_SIZE GPUs listed by nvidia-smi,
# so the visible devices must be exactly those GPUs, unless the caller sets
# VLLM_CPU_OMP_THREADS_BIND, KVSHRINK_QAT_DEVICES and KVSHRINK_DSA_DEVICES.
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
export PORT=${PORT:-8000}
INPUT_LENS=${INPUT_LENS:-"100000 200000"}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SERVE_READY_TIMEOUT=${SERVE_READY_TIMEOUT:-1800}
SERVE_STOP_TIMEOUT=${SERVE_STOP_TIMEOUT:-180}

# K1: QAT + IAA compression, DSA transfers, every layer compressed.
export IAXL_DSA_GD_ENABLE=1
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export IAXL_IAA_ZIP_ENABLE=1
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0

RESULTS_DIR=${RESULTS_DIR:-tests/bench_dev/results/probe-tp${TP_SIZE}-$(date +'%Y%m%d_%H%M%S')}
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
git rev-parse HEAD > "$RESULTS_DIR/git_commit.txt"
git diff > "$RESULTS_DIR/git_diff.patch"
md5sum examples/kvshrink-vllm-serve.sh kvshrink/*.py > "$RESULTS_DIR/code_md5.txt"

SERVE_PID=""

stop_serve() {
    if [ -z "$SERVE_PID" ]; then
        return 0
    fi
    kill -TERM -- "-$SERVE_PID" 2>/dev/null
    local waited=0
    while (( waited < SERVE_STOP_TIMEOUT )) && kill -0 -- "-$SERVE_PID" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    kill -KILL -- "-$SERVE_PID" 2>/dev/null
    SERVE_PID=""
    echo "    serve stopped after ${waited}s"
}
trap stop_serve EXIT INT TERM

# Returns 0 once the serve answers /v1/models, 1 if it exits or times out.
wait_for_serve() {
    local waited=0
    while (( waited < SERVE_READY_TIMEOUT )); do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            echo "    serve ready after ${waited}s"
            return 0
        fi
        if ! kill -0 "$SERVE_PID" 2>/dev/null; then
            echo "    ERROR: serve exited before becoming ready" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "    ERROR: serve not ready within ${SERVE_READY_TIMEOUT}s" >&2
    return 1
}

for input_len in $INPUT_LENS; do
    export MAX_MODEL_LEN=$((input_len + OUTPUT_LEN))
    serve_log="$RESULTS_DIR/serve_in${input_len}.log"
    echo "=== TP_SIZE=$TP_SIZE GPUs=$CUDA_VISIBLE_DEVICES input_len=$input_len MAX_MODEL_LEN=$MAX_MODEL_LEN -> $serve_log"

    # Own process group, so stop_serve cannot touch any other vLLM.
    setsid bash examples/kvshrink-vllm-serve.sh > "$serve_log" 2>&1 &
    SERVE_PID=$!

    if ! wait_for_serve; then
        stop_serve
        exit 1
    fi

    {
        echo "=== input_len=$input_len MAX_MODEL_LEN=$MAX_MODEL_LEN TP_SIZE=$TP_SIZE GPUs=$CUDA_VISIBLE_DEVICES"
        # The log contains '\r' progress bars; split on them before matching.
        tr '\r' '\n' < "$serve_log" \
            | grep -aoE "Model loading took .*|Setting attention block size to [0-9]+ tokens|KVShrinkConnector\(WORKER\): block_size=.*|max_num_batched_tokens=[0-9]+|Available KV cache memory: .*|GPU KV cache size: .*|Maximum concurrency for .*" \
            | sort | uniq -c
    } | tee -a "$SUMMARY"

    stop_serve
done

echo "=== done: $SUMMARY"
