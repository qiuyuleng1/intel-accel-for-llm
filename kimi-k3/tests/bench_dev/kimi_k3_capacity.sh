#!/bin/bash
# Kimi-K3 + kvshrink, DP2 TP4 + EP: does the test matrix fit? Launches (or reuses) the vLLM of
# kimi_k3_dp_layouts.sh's DP2 TP4 layout through kimi_k3_smoke.sh (smoke test included), then
# runs, per case, one `vllm bench serve` per DP rank at the same time, pinned with
# X-data-parallel-rank so each DP group gets exactly its share:
#   32k:  7 requests per DP rank, 7 in parallel (14 in total)
#   128k: 1 request per DP rank (2 in total)
# CASES="<input_len>:<per rank> ..." picks other cases; SKIP_SMOKE=1 reuses the running vLLM
# without the smoke test.
# Random prompts, 128 output tokens (ignore_eos). Per case it reports each rank's completed /
# failed requests and TTFT, the peak running / waiting requests and KV cache usage per engine
# (/metrics sampled every 2 s) and vLLM's preemption count. vLLM is never stopped.
#
# Runs INSIDE this clone's container. From the b300 host, in tmux:
#   docker exec -w /home/intel/qiuyu/intel-accel-for-llm-kimi-k3 iaxl.vllm-kimi-k3-qiuyu \
#       bash -lc 'bash tests/bench_dev/kimi_k3_capacity.sh'
set -uo pipefail
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
cd "$REPO_DIR"

# Same options as the DP2 TP4 layout of kimi_k3_dp_layouts.sh.
export DP_SIZE=2 TP_SIZE=4
export IAXL_DSA_GD_ENABLE=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_KV_COMPRESSION=1
export IAXL_DDR_POOL_TOTAL_SIZE_GB=1200 IAXL_SCRATCH_POOL_SIZE_GB=8
export MAX_MODEL_LEN=-1
export VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.9 --enable-expert-parallel --max-num-seqs 128"
MODEL=/home/intel/model-space/Kimi-K3
PORT=8000
CASES=${CASES:-"32000:7 128000:1"}
OUTPUT_LEN=128
SERVE_STATE=${SERVE_STATE:-$DEV_DIR/results/kimi-k3-serve.state}

CODE_MD5=$(cat kvshrink/*.py "$DEV_DIR/kimi_k3_capacity.sh" | md5sum | cut -c1-8)
RESULTS_DIR=$DEV_DIR/results/kimi-k3-capacity-$(date +%Y%m%d_%H%M%S)_$CODE_MD5
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

preemptions() {
    curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^vllm:num_preemptions_total/ {sum += $NF} END {print sum + 0}'
}

# Every 2 s until killed: running / waiting requests and KV cache usage per engine.
sample_gauges() {
    local out=$1
    while true; do
        curl -s "http://127.0.0.1:$PORT/metrics" \
            | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)\{' >> "$out"
        sleep 2
    done
}

log "code md5 $CODE_MD5, results $RESULTS_DIR"
if [[ "${SKIP_SMOKE:-0}" == 1 ]]; then
    if [[ ! -f "$SERVE_STATE" ]]; then
        log "ERROR: SKIP_SMOKE=1 needs a running vLLM ($SERVE_STATE missing)"
        exit 1
    fi
elif ! bash "$DEV_DIR/kimi_k3_smoke.sh"; then
    log "ERROR: smoke test (or launch) failed"
    exit 1
fi
read -r serve_pgid serve_log < "$SERVE_STATE"
log "vLLM process group $serve_pgid, log $serve_log"

for case in $CASES; do
    input_len=${case%%:*}
    per_rank=${case##*:}
    case_dir=$RESULTS_DIR/in${input_len}_x$per_rank
    mkdir -p "$case_dir"
    preempt_before=$(preemptions)
    log_start=$(stat -c %s "$serve_log")
    log "=== input $input_len: $per_rank request(s) per DP rank, $DP_SIZE ranks at once"
    sample_gauges "$case_dir/gauges.txt" &
    sampler_pid=$!
    pids=()
    for ((rank = 0; rank < DP_SIZE; rank++)); do
        vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
            --trust-remote-code --dataset-name random --random-input-len "$input_len" \
            --random-output-len "$OUTPUT_LEN" --ignore-eos --num-prompts "$per_rank" \
            --max-concurrency "$per_rank" --seed "$((input_len + rank))" \
            --header "X-data-parallel-rank=$rank" \
            --save-result --save-detailed --result-dir "$case_dir" --result-filename "dp$rank.json" \
            > "$case_dir/bench_dp$rank.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    sleep 12
    kill "$sampler_pid" 2> /dev/null
    preempt_after=$(preemptions)
    tail -c "+$((log_start + 1))" "$serve_log" | tr '\r' '\n' > "$case_dir/serve_part.log"
    for ((rank = 0; rank < DP_SIZE; rank++)); do
        python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(f'  DP{sys.argv[2]}: completed {r[\"completed\"]}, failed {r.get(\"failed\", 0)}, mean TTFT {r[\"mean_ttft_ms\"]:.0f} ms, max TTFT {max(r[\"ttfts\"]) * 1000:.0f} ms')
" "$case_dir/dp$rank.json" "$rank" || log "  DP$rank: no result (see $case_dir/bench_dp$rank.log)"
    done
    # Lines look like: vllm:num_requests_running{engine="0",model_name="..."} 7.0
    awk '
        {
            name = $1; sub(/\{.*/, "", name)
            engine = $1; sub(/.*engine="/, "", engine); sub(/".*/, "", engine)
            key = name " engine " engine
            if (!(key in peak) || $NF + 0 > peak[key]) peak[key] = $NF + 0
        }
        END { for (key in peak) printf "  max %s: %s\n", key, peak[key] }' "$case_dir/gauges.txt" | sort
    log "  preemptions during the case: $((preempt_after - preempt_before))"
    for pattern in "Traceback" " ERROR " "IAXL check failed" "cache key missing after retry"; do
        log "  serve log \"$pattern\": $(grep -ac "$pattern" "$case_dir/serve_part.log")"
    done
done
log "vLLM left running (process group $serve_pgid)"
log "ALL_DONE $RESULTS_DIR"
