#!/bin/bash
# gsm8k correctness of Kimi-K3 with kvshrink, on the vLLM that kimi_k3_smoke.sh left
# running (its options; this script never starts or stops vLLM). For each few-shot count
# in SHOTS, gsm8k runs twice (pass1, pass2) over the same LIMIT questions. The few-shot
# examples are a prompt prefix shared by all questions and set the input length:
#   26 shots:  ~4.5k-token prompts, inside one 16384-token prefill chunk
#   140 shots: ~24k-token prompts, longer than one chunk, so requests also hit through
#              mamba states saved where a chunk stopped at the shared-prefix boundary
# evalscope, 50 in parallel, temperature 0, default chat template, max_tokens 4096
# (Kimi-K3 thinks before answering; prompt + 4096 stays under max_model_len 32768).
# Per pass it keeps the evalscope output, vLLM's /metrics before and after, and the
# connector log lines; kimi_k3_gsm8k_report.py then reports accuracy, hits and garbled
# outputs.
#
# Runs INSIDE this clone's container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 iaxl.vllm-kimi-k3-qiuyu \
#       bash -lc 'bash tests/bench_dev/kimi_k3_gsm8k.sh'
# LIMIT / SHOTS to change the run, DRY_RUN=1 prints the plan.
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)

# Must match kimi_k3_smoke.sh, which starts the serve.
PORT=8000
MODEL=/home/intel/model-space/Kimi-K3
SERVE_STATE=$DEV_DIR/results/kimi-k3-serve.state
SHOTS=${SHOTS:-"26 140"}
# 30 questions: one wave of PARALLEL requests per pass, so all 4 passes fit in ~10 min.
LIMIT=${LIMIT:-30}
PARALLEL=50
GENERATION_CONFIG='{"max_tokens": 4096, "temperature": 0.0}'
DATASET_DIR=$DEV_DIR/gsm8k-datasets
DRY_RUN=${DRY_RUN:-0}
# The serve may still be loading the weights (1600 s on 11-12).
SERVE_READY_TIMEOUT_S=5400
# evalscope waits forever for a request that never returns, so cap a pass.
PASS_TIMEOUT_S=600
CODE_MD5=$(cat "$REPO_DIR"/kvshrink/*.py "$DEV_DIR/kimi_k3_gsm8k.sh" \
    "$DEV_DIR/kimi_k3_gsm8k_report.py" | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/kimi-k3-gsm8k-$(date +%Y%m%d_%H%M%S)_${CODE_MD5}_limit$LIMIT

if [[ "$DRY_RUN" == 1 ]]; then
    echo "results: $RESULTS_DIR"
    for shots in $SHOTS; do
        echo "gsm8k $shots shots: pass1, pass2; $LIMIT questions, $PARALLEL in parallel"
    done
    exit 0
fi

if [[ ! -f "$SERVE_STATE" ]]; then
    echo "ERROR: no vLLM left running ($SERVE_STATE missing); start one with kimi_k3_smoke.sh"
    exit 1
fi
read -r SERVE_PGID SERVE_LOG < "$SERVE_STATE"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

# run_pass <shots> <pass>: one gsm8k run, with /metrics and the connector lines it
# produced. The run order is kept in run_order.txt for the report.
run_pass() {
    local shots=$1 pass=$2
    local pass_dir=$RESULTS_DIR/kvshrink/shots$shots/$pass
    local log_start log_end status
    mkdir -p "$pass_dir"
    echo "kvshrink shots$shots $pass" >> "$RESULTS_DIR/run_order.txt"
    curl -s "http://127.0.0.1:$PORT/metrics" > "$pass_dir/metrics_before.txt"
    log_start=$(stat -c %s "$SERVE_LOG")
    log "gsm8k $shots shots $pass: $LIMIT questions, $PARALLEL in parallel"
    timeout "$PASS_TIMEOUT_S" evalscope eval \
        --model "$MODEL" \
        --api-url "http://127.0.0.1:$PORT/v1" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets gsm8k \
        --dataset-dir "$DATASET_DIR" \
        --limit "$LIMIT" \
        --eval-batch-size "$PARALLEL" \
        --work-dir "$pass_dir" \
        --dataset-args "{\"gsm8k\": {\"few_shot_num\": $shots}}" \
        --generation-config "$GENERATION_CONFIG" > "$pass_dir/evalscope.log" 2>&1
    status=$?
    sleep 2
    curl -s "http://127.0.0.1:$PORT/metrics" > "$pass_dir/metrics_after.txt"
    log_end=$(stat -c %s "$SERVE_LOG")
    # Byte offsets, not line counts: the serve log contains '\r' progress output.
    tail -c +$((log_start + 1)) "$SERVE_LOG" | head -c $((log_end - log_start)) \
        | tr '\r' '\n' | grep -a "get_num_new_matched_tokens" > "$pass_dir/connector_log.txt"
    log "$shots shots $pass: evalscope exit $status," \
        "$(wc -l < "$pass_dir/connector_log.txt") connector lines," \
        "score $(grep -ohE '"score": [0-9.]+' "$pass_dir"/*/reports/*/gsm8k.json 2> /dev/null | head -1)"
    return $status
}

log "code md5 $CODE_MD5, results $RESULTS_DIR"
log "vLLM process group $SERVE_PGID, log $SERVE_LOG"
waited=0
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 -- "-$SERVE_PGID" 2> /dev/null; then
        log "ERROR: vLLM process group $SERVE_PGID is gone"
        exit 1
    fi
    if (( waited >= SERVE_READY_TIMEOUT_S )); then
        log "ERROR: serve not ready within ${SERVE_READY_TIMEOUT_S}s"
        exit 1
    fi
    sleep 10
    waited=$((waited + 10))
done
log "serve ready (waited ${waited}s)"
SERVE_LOG_START=$(stat -c %s "$SERVE_LOG")

failed=0
for shots in $SHOTS; do
    for pass in pass1 pass2; do
        if ! run_pass "$shots" "$pass"; then
            failed=1
        fi
    done
done
for pattern in "Traceback" " ERROR " "IAXL check failed" "cache key missing after retry" \
    "cache miss on unzip"; do
    log "serve log of this run \"$pattern\":" \
        "$(tail -c "+$((SERVE_LOG_START + 1))" "$SERVE_LOG" | tr '\r' '\n' | grep -ac "$pattern")"
done
if (( failed )); then
    log "ERROR: a pass failed (see evalscope.log); the report covers the passes that finished"
fi
python3 "$DEV_DIR/kimi_k3_gsm8k_report.py" "$RESULTS_DIR" | tee "$RESULTS_DIR/report.txt"
log "vLLM left running (process group $SERVE_PGID)"
log "ALL_DONE $RESULTS_DIR"
