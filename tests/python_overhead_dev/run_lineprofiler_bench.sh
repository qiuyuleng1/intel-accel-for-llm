#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# line_profiler report of kvshrink's Python code for one workload (32k input at concurrency 1, or 2k
# input at concurrency 16), all requests hit.
# Runs on the host, in tmux; vLLM and the benchmark client run in the dev container
# (CONTAINER_NAME in setvars.sh). vLLM loads LineProfiledKVShrinkConnector
# (lineprofiled_kvshrink_connector.py): kvshrink unchanged, line_profiler on during its calls.
#   start vLLM -> warm up -> store: the measured bench once, so that its prompt is in DDR
#   -> the same bench with line_profiler on -> stop vLLM -> line_profile_report.txt
# Results go to tests/python_overhead_dev/results/line_profiler/<time>_<code md5>_<workload>_hit/,
# the vLLM log written there live.
#
#   ./run_lineprofiler_bench.sh                    # 32k input, concurrency 1, 10 requests
#   WORKLOAD=2k_c16 ./run_lineprofiler_bench.sh    # 2k input, concurrency 16, 160 requests
#   SMOKE=1 ./run_lineprofiler_bench.sh            # 2 x concurrency requests, results/line_profiler/smoke/
set -euo pipefail

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$BENCH_DIR/../.." && pwd)
CONTAINER_NAME=$(sed -n 's/^CONTAINER_NAME=//p' "$REPO_DIR/setvars.sh")

WORKLOAD=${WORKLOAD:-32k_c1}
case "$WORKLOAD" in
    32k_c1) INPUT_LEN=32768; CONCURRENCY=1; NUM_PROMPTS=10 ;;
    2k_c16) INPUT_LEN=2048; CONCURRENCY=16; NUM_PROMPTS=160 ;;
    *) echo "WORKLOAD must be 32k_c1 or 2k_c16" >&2; exit 1 ;;
esac
RESULTS_DIR="$BENCH_DIR/results/line_profiler"
if [[ "${SMOKE:-0}" == 1 ]]; then
    NUM_PROMPTS=$((2 * CONCURRENCY))
    RESULTS_DIR="$BENCH_DIR/results/line_profiler/smoke"
fi

MODEL=/media/model-space/Qwen3-32B
OUTPUT_LEN=128
# One server config for both workloads: room for one 32k request plus its output.
MAX_MODEL_LEN=32896
# examples/kvshrink-vllm-serve.sh uses 0.8, which leaves no room for one 32k request at TP=1.
GPU_MEMORY_UTILIZATION=0.9
GPU=5
# GPU 4-7 sit on NUMA node 1, whose physical cores are 64-127 (192-255 are their hyperthreads).
CPU_BIND=64-127
# Not the ports of other users' servers on this host (host network).
PORT=8020
CONTROLLER_PORT=18720
WORKER_BASE_PORT=18820
WARMUP_PROMPTS=$((2 * CONCURRENCY))
WARMUP_SEED=1
# The store bench and the measured bench use one seed, so they send the same prompt.
HIT_SEED=4

SERVE_ENV=(
    MODEL="$MODEL" TP_SIZE=1 CUDA_VISIBLE_DEVICES="$GPU" VLLM_CPU_OMP_THREADS_BIND="$CPU_BIND"
    PORT="$PORT" IAXL_API_CONTROLLER_PORT="$CONTROLLER_PORT" IAXL_API_WORKER_BASE_PORT="$WORKER_BASE_PORT"
    # The default 8 GB holds exactly the layers of one 32k request; when it runs out, KVFlow
    # busy-waits in Python.
    IAXL_SCRATCH_POOL_SIZE_GB=16
    # The default (80% of host memory) is far more than this shared host has free.
    IAXL_DDR_POOL_SIZE_GB=64
    # Blocks are stored raw, so kv_zip never starts QAT: the QAT user-space slots are shared with
    # other jobs on this host, and a vLLM that cannot get one dies at its first compression.
    IAXL_KV_COMPRESSION=0
    IAXL_DSA_GD_ENABLE=0
)
SERVE_ARGS=(
    --kv-transfer-config "'{\"kv_connector\":\"LineProfiledKVShrinkConnector\",\"kv_connector_module_path\":\"lineprofiled_kvshrink_connector\",\"kv_role\":\"kv_both\"}'"
    --trust-remote-code -tp 1 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --port "$PORT" --no-enable-prefix-caching
)
BENCH_ARGS=(
    --backend vllm --model "$MODEL" --tokenizer "$MODEL" --dataset-name random
    --host localhost --port "$PORT" --random-output-len "$OUTPUT_LEN" --ignore-eos
    --request-rate inf --max-concurrency "$CONCURRENCY" --trust-remote-code
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95 --save-result
)
# Warm-up prompts are new random tokens; the store and measured benches send one shared prompt of
# INPUT_LEN tokens, so every measured request hits what the store bench saved.
MISS_PROMPT_ARGS=(--random-prefix-len 0 --random-input-len "$INPUT_LEN")
HIT_PROMPT_ARGS=(--random-prefix-len "$INPUT_LEN" --random-input-len 0)

CODE_MD5=$(cd "$REPO_DIR" && cat $(git ls-files 'kvshrink/*.py' 'iaxl/*.py') \
    "$BENCH_DIR/run_lineprofiler_bench.sh" "$BENCH_DIR/lineprofiled_kvshrink_connector.py" \
    | md5sum | cut -c1-8)
OUT_DIR="$RESULTS_DIR/$(date +%Y%m%d_%H%M%S)_${CODE_MD5}_${WORKLOAD}_hit"

SERVE_PID=""
ENGINE_CORE_PID=""
CONTROL_SEQ=0

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

in_container() {
    docker exec -w "$REPO_DIR" "$CONTAINER_NAME" "$@"
}

gpu_memory_used_mib() {
    nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits
}

stop_vllm() {
    [[ -z "$SERVE_PID" ]] && return 0
    log "stopping vLLM (pid $SERVE_PID)"
    in_container kill -TERM "$SERVE_PID" || true
    for _ in $(seq 1 60); do
        ps -p "$SERVE_PID" > /dev/null || break
        sleep 2
    done
    for _ in $(seq 1 30); do
        (( $(gpu_memory_used_mib) < 1000 )) && break
        sleep 2
    done
    # The GPU was free before this run, so whatever still holds it is a leftover of our server.
    if (( $(gpu_memory_used_mib) >= 1000 )); then
        for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader); do
            log "killing leftover GPU $GPU process $pid"
            in_container kill -9 "$pid" || true
        done
    fi
    SERVE_PID=""
    log "GPU $GPU memory used: $(gpu_memory_used_mib) MiB"
}

cleanup() {
    stop_vllm || true
    # Files written in the container belong to root; give them to the host user.
    docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" "$OUT_DIR" 2> /dev/null || true
}

start_vllm() {
    log "starting vLLM on GPU $GPU, log $OUT_DIR/vllm.log"
    docker exec -w "$REPO_DIR" -d "$CONTAINER_NAME" bash -c "exec > $OUT_DIR/vllm.log 2>&1; \
        export ${SERVE_ENV[*]}; \
        export PROFILE_CONTROL_DIR=$OUT_DIR; \
        export PYTHONPATH=$BENCH_DIR\${PYTHONPATH:+:\$PYTHONPATH}; \
        source setvars.sh; \
        export LD_PRELOAD=\"\$IAXL_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
        exec vllm serve $MODEL ${SERVE_ARGS[*]}"
    # Only the python process running the vllm script: before its exec, the bash wrapper (and the
    # subshells setvars.sh forks) carry the same "vllm serve ... --port" text in their arguments.
    for _ in $(seq 1 60); do
        SERVE_PID=$(ps -eo pid,args | awk -v port="--port $PORT" \
            '$2 ~ /python/ && $3 ~ /\/vllm$/ && $4 == "serve" && index($0, port) {print $1}')
        [[ -n "$SERVE_PID" ]] && break
        sleep 1
    done
    if [[ -z "$SERVE_PID" ]]; then
        log "ERROR: vllm serve did not start; see $OUT_DIR/vllm.log"
        return 1
    fi
    for _ in $(seq 1 240); do
        curl -sf "http://localhost:$PORT/health" > /dev/null && break
        if ! ps -p "$SERVE_PID" > /dev/null; then
            log "ERROR: vLLM exited during startup; last error lines of $OUT_DIR/vllm.log:"
            grep -aE "Error|error" "$OUT_DIR/vllm.log" | tail -5
            SERVE_PID=""
            return 1
        fi
        sleep 5
    done
    curl -sf "http://localhost:$PORT/health" > /dev/null
    ENGINE_CORE_PID=$(ps -eo pid,ppid,args | awk -v parent="$SERVE_PID" '$2 == parent && /VLLM::EngineCore/ {print $1}')
    if [[ -z "$ENGINE_CORE_PID" ]]; then
        log "ERROR: no VLLM::EngineCore child of $SERVE_PID"
        return 1
    fi
    log "vLLM ready: serve pid $SERVE_PID, EngineCore pid $ENGINE_CORE_PID"
}

run_bench() {
    # run_bench <name> <seed> <num prompts> <prompt args...>
    local name=$1 seed=$2 num_prompts=$3
    shift 3
    local start end
    log "bench $name: $num_prompts requests, seed $seed"
    start=$(stat -c %s "$OUT_DIR/vllm.log")
    in_container vllm bench serve "${BENCH_ARGS[@]}" "$@" --seed "$seed" --num-prompts "$num_prompts" \
        --result-dir "$OUT_DIR" --result-filename "bench_$name.json" 2>&1 | tee "$OUT_DIR/bench_$name.log"
    end=$(stat -c %s "$OUT_DIR/vllm.log")
    # The report slices the vLLM log by byte offset: the log has '\r' progress output.
    echo "$name $start $end" >> "$OUT_DIR/bench_offsets.txt"
    local failed
    failed=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['failed'])" \
        "$OUT_DIR/bench_$name.json")
    # vllm bench serve exits 0 even when every request failed (e.g. EngineCore died during the
    # bench), so check the result and the process instead of going on with a dead server.
    if [[ "$failed" != 0 ]] || ! ps -p "$ENGINE_CORE_PID" > /dev/null; then
        log "ERROR: bench $name had $failed failed requests or EngineCore $ENGINE_CORE_PID exited; see $OUT_DIR/vllm.log"
        return 1
    fi
}

send_control() {
    # send_control <command>; see lineprofiled_kvshrink_connector.py for the commands.
    CONTROL_SEQ=$((CONTROL_SEQ + 1))
    # Written to another file and renamed, so the connector never reads a half-written line.
    echo "$CONTROL_SEQ $1" > "$OUT_DIR/control.tmp"
    mv "$OUT_DIR/control.tmp" "$OUT_DIR/control"
}

wait_for_control_log() {
    # wait_for_control_log <seq> <text>: wait until the connector has logged "<seq> <text>".
    for _ in $(seq 1 120); do
        grep -q " $1 $2\$" "$OUT_DIR/control.log" 2> /dev/null && return 0
        sleep 1
    done
    log "ERROR: the connector did not log '$1 $2'; see $OUT_DIR/control.log"
    return 1
}

send_small_request() {
    curl -sf "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"hello\", \"max_tokens\": 1}" > /dev/null
}

external_tokens_of_bench() {
    # external_tokens_of_bench <name>: "count value" pairs of externally-cached tokens in the
    # connector log lines of that bench.
    local start end
    read -r start end < <(awk -v name="$1" '$1 == name {print $2, $3}' "$OUT_DIR/bench_offsets.txt")
    tail -c +$((start + 1)) "$OUT_DIR/vllm.log" | head -c $((end - start)) \
        | grep -ao 'externally-cached tokens: [0-9]*' | awk '{print $3}' | sort | uniq -c
}

write_report() {
    local report="$OUT_DIR/line_profile_report.txt"
    {
        echo "line_profiler 报告:$INPUT_LEN token 输入、并发 $CONCURRENCY、全部命中(Qwen3-32B,TP=1,GPU $GPU,KV 压缩关闭)"
        echo "测量 bench:$NUM_PROMPTS 个请求,每个输出 $OUTPUT_LEN token;下面的时间是这 $NUM_PROMPTS 个请求加起来的,单位 us(微秒)。"
        echo "connector 日志里这 $NUM_PROMPTS 个请求的 externally-cached tokens(次数 值):$(external_tokens_of_bench measured | tr '\n' ' ')"
        echo "Hits = 这一行执行的次数;Time = 总时间;Per Hit = 每次的时间;% Time = 占所在函数总时间的比例。"
        echo "line_profiler 每执行一行都要额外计时,循环里的行时间偏大。"
        echo "line_profiler 不分线程:测量期间 kvshrink 后台线程调到这些函数的时间也算在内(如 KVStoreLocal.simple_status)。"
        echo "代码 git $(git -C "$REPO_DIR" rev-parse --short HEAD),md5 $CODE_MD5,目录 $OUT_DIR"
        echo
        cat "$OUT_DIR/line_profile.txt"
    } > "$report"
    log "report: $report"
}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "container $CONTAINER_NAME is not running; start it with ./start.sh" >&2
    exit 1
fi
if (( $(gpu_memory_used_mib) >= 1000 )); then
    echo "GPU $GPU is in use ($(gpu_memory_used_mib) MiB)" >&2
    exit 1
fi
if ps -eo args | grep -q "[v]llm serve .*--port $PORT"; then
    echo "a vLLM server on port $PORT is already running" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
trap cleanup EXIT
# Ctrl-C in tmux must also stop vLLM: exit through the EXIT trap instead of being killed.
trap 'exit 130' INT TERM
log "results: $OUT_DIR"

start_vllm
run_bench warmup "$WARMUP_SEED" "$WARMUP_PROMPTS" "${MISS_PROMPT_ARGS[@]}"
# The number of prompts changes how the dataset draws its prefix, so the store bench has to be the
# measured bench itself; its first request saves the prompt.
run_bench store "$HIT_SEED" "$NUM_PROMPTS" "${HIT_PROMPT_ARGS[@]}"
send_control start
start_seq=$CONTROL_SEQ
run_bench measured "$HIT_SEED" "$NUM_PROMPTS" "${HIT_PROMPT_ARGS[@]}"
# The first request of the bench made the connector read the start command.
wait_for_control_log "$start_seq" started
send_control stop
stop_seq=$CONTROL_SEQ
# An idle EngineCore makes no connector calls, so one small request makes it read the stop command;
# the connector handles the command before doing this request's own work.
send_small_request
wait_for_control_log "$stop_seq" stopped
stop_vllm
write_report
log "done"
