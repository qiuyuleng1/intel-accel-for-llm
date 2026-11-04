#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Profiles kvshrink's Python code with py-spy while vLLM (TP=1, one GPU) serves a benchmark.
# Runs on the host, in tmux; vLLM, the benchmark client and py-spy run in the dev container
# (CONTAINER_NAME in setvars.sh). One invocation = one workload x one scenario:
#   start vLLM -> warm up -> [hit: run the measured bench once so that its prompt is stored]
#   -> baseline: the measured bench without py-spy -> the same bench with py-spy on EngineCore
#   -> stop vLLM -> analyze_pyspy.py writes summary.md.
# Results go to tests/python_overhead_dev/results/<time>_<code md5>_<workload>_<scenario>/, the
# vLLM log written there live.
#
#   WORKLOAD=32k_c1 SCENARIO=miss ./run_pyspy_bench.sh   # 32k input, concurrency 1, 50 requests
#   WORKLOAD=2k_c16 SCENARIO=hit ./run_pyspy_bench.sh    # 2k input, concurrency 16, 800 requests
#   SMOKE=1 WORKLOAD=2k_c16 SCENARIO=hit ./run_pyspy_bench.sh   # 4 x concurrency requests, results/smoke/
#   DRY_RUN=1 WORKLOAD=32k_c1 SCENARIO=miss ./run_pyspy_bench.sh   # print the commands only
set -euo pipefail

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$BENCH_DIR/../.." && pwd)
CONTAINER_NAME=$(sed -n 's/^CONTAINER_NAME=//p' "$REPO_DIR/setvars.sh")

WORKLOAD=${WORKLOAD:?set WORKLOAD to 32k_c1 or 2k_c16}
SCENARIO=${SCENARIO:?set SCENARIO to miss or hit}
case "$WORKLOAD" in
    32k_c1) INPUT_LEN=32768; CONCURRENCY=1; NUM_PROMPTS=50 ;;
    2k_c16) INPUT_LEN=2048; CONCURRENCY=16; NUM_PROMPTS=800 ;;
    *) echo "unknown WORKLOAD '$WORKLOAD' (32k_c1 or 2k_c16)" >&2; exit 1 ;;
esac
case "$SCENARIO" in
    miss|hit) ;;
    *) echo "unknown SCENARIO '$SCENARIO' (miss or hit)" >&2; exit 1 ;;
esac
SMOKE=${SMOKE:-0}
RESULTS_DIR="$BENCH_DIR/results"
if [[ "$SMOKE" == 1 ]]; then
    # Only checks that the whole pipeline works; too few samples for any conclusion.
    NUM_PROMPTS=$((4 * CONCURRENCY))
    RESULTS_DIR="$BENCH_DIR/results/smoke"
fi

MODEL=/media/model-space/Qwen3-32B
OUTPUT_LEN=128
# One max length for both workloads: the longest request is 32k input + output.
MAX_MODEL_LEN=$((32768 + OUTPUT_LEN))
# examples/kvshrink-vllm-serve.sh uses 0.8, which leaves no room for one 32k request at TP=1.
GPU_MEMORY_UTILIZATION=0.9
GPU=${GPU:-5}
# GPU 4-7 sit on NUMA node 1, whose physical cores are 64-127 (192-255 are their hyperthreads).
CPU_BIND=${CPU_BIND:-64-127}
# py-spy runs on NUMA node 0 so that it takes no CPU time from the measured process.
PYSPY_CPUS=${PYSPY_CPUS:-60-63}
PYSPY_RATE=${PYSPY_RATE:-100}
# Not the ports of other users' servers on this host (host network).
PORT=8020
CONTROLLER_PORT=18720
WORKER_BASE_PORT=18820
WARMUP_PROMPTS=$((2 * CONCURRENCY))
# miss: every bench gets its own seed, so its prompts are new to the DDR pool.
WARMUP_SEED=1
BASELINE_SEED=2
PYSPY_SEED=3
# hit: all benches use one seed, so they send the same prompt that the store run saved.
HIT_SEED=4
DRY_RUN=${DRY_RUN:-0}

SERVE_ENV=(
    MODEL="$MODEL" TP_SIZE=1 CUDA_VISIBLE_DEVICES="$GPU" VLLM_CPU_OMP_THREADS_BIND="$CPU_BIND"
    PORT="$PORT" IAXL_API_CONTROLLER_PORT="$CONTROLLER_PORT" IAXL_API_WORKER_BASE_PORT="$WORKER_BASE_PORT"
    # The default 8 GB holds exactly the layers of one 32k request; when it runs out, KVFlow
    # busy-waits, which py-spy would count as Python work.
    IAXL_SCRATCH_POOL_SIZE_GB=16
    # The default (80% of host memory) is far more than this shared host has free.
    IAXL_DDR_POOL_SIZE_GB=64
    # miss writes the pool full within seconds; check usage every 10 s instead of 30 s so that
    # eviction keeps host memory near the pool size.
    IAXL_DDR_POOL_EVICT_INTERVAL_S=10
    # Blocks are stored raw, so kv_zip never starts QAT: the QAT user-space slots are shared with
    # other jobs on this host, and a vLLM that cannot get one dies at its first compression.
    IAXL_KV_COMPRESSION=0
    IAXL_DSA_GD_ENABLE=0
)
SERVE_ARGS=(
    --kv-transfer-config "'{\"kv_connector\":\"KVShrinkConnector\",\"kv_connector_module_path\":\"kvshrink.kvshrink_connector\",\"kv_role\":\"kv_both\"}'"
    --trust-remote-code -tp 1 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --port "$PORT" --no-enable-prefix-caching
)
BENCH_ARGS=(
    --backend vllm --model "$MODEL" --tokenizer "$MODEL" --dataset-name random
    --host localhost --port "$PORT" --random-output-len "$OUTPUT_LEN" --ignore-eos
    --request-rate inf --max-concurrency "$CONCURRENCY" --trust-remote-code
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95 --save-result
)
# miss: INPUT_LEN random tokens per request. hit: a shared prefix of INPUT_LEN tokens and nothing
# else, so every request of the bench has the same prompt.
MISS_PROMPT_ARGS=(--random-prefix-len 0 --random-input-len "$INPUT_LEN")
HIT_PROMPT_ARGS=(--random-prefix-len "$INPUT_LEN" --random-input-len 0)
# --native pauses vLLM for every sample long enough to slow the bench to 1.8x (smoke run 11-03), and
# the stripped IAXL library gives no function names anyway, so only Python frames are read.
PYSPY_ARGS=(--rate "$PYSPY_RATE" --threads --format raw --full-filenames)
# pause: py-spy stops vLLM for each sample, so every stack it reads is consistent, but its busy
# main-thread time came out 1.59x the kernel's CPU time and TTFT 1.5x (smoke run 11-03).
# nonblocking: no stop; stacks that change while being read come back broken or not at all, so its
# busy main-thread time came out 0.80x the kernel's (smoke run 11-03). The analysis scales it to the
# kernel's CPU time.
PYSPY_MODE=${PYSPY_MODE:-nonblocking}
case "$PYSPY_MODE" in
    pause) ;;
    nonblocking) PYSPY_ARGS+=(--nonblocking) ;;
    *) echo "unknown PYSPY_MODE '$PYSPY_MODE' (pause or nonblocking)" >&2; exit 1 ;;
esac

CODE_MD5=$(cd "$REPO_DIR" && cat $(git ls-files 'kvshrink/*.py' 'iaxl/*.py') \
    "$BENCH_DIR/run_pyspy_bench.sh" "$BENCH_DIR/analyze_pyspy.py" | md5sum | cut -c1-8)
OUT_DIR="$RESULTS_DIR/$(date +%Y%m%d_%H%M%S)_${CODE_MD5}_${WORKLOAD}_${SCENARIO}_${PYSPY_MODE}"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "results: $OUT_DIR"
    echo "serve:   ${SERVE_ENV[*]} vllm serve $MODEL ${SERVE_ARGS[*]}"
    echo "bench:   vllm bench serve ${BENCH_ARGS[*]} --num-prompts $NUM_PROMPTS" \
         "$( [[ "$SCENARIO" == miss ]] && echo "${MISS_PROMPT_ARGS[*]}" || echo "${HIT_PROMPT_ARGS[*]}" )"
    echo "py-spy:  taskset -c $PYSPY_CPUS py-spy record --pid <EngineCore> ${PYSPY_ARGS[*]}"
    exit 0
fi

SERVE_PID=""
ENGINE_CORE_PID=""
PYSPY_PID=""

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

in_container() {
    docker exec -w "$REPO_DIR" "$CONTAINER_NAME" "$@"
}

gpu_memory_used_mib() {
    nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits
}

stop_pyspy() {
    [[ -z "$PYSPY_PID" ]] && return 0
    log "stopping py-spy (pid $PYSPY_PID)"
    in_container kill -INT "$PYSPY_PID" || true
    # On SIGINT py-spy stops sampling and only then writes the output file.
    for _ in $(seq 1 120); do
        ps -p "$PYSPY_PID" > /dev/null || break
        sleep 1
    done
    if ps -p "$PYSPY_PID" > /dev/null; then
        log "ERROR: py-spy $PYSPY_PID did not exit"
        return 1
    fi
    PYSPY_PID=""
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
    stop_pyspy || true
    stop_vllm || true
    # Files written in the container belong to root; give them to the host user.
    docker exec "$CONTAINER_NAME" chown -R "$(id -u):$(id -g)" "$OUT_DIR" 2> /dev/null || true
}

start_vllm() {
    # QAT's usdm driver needs physically contiguous 4 MB buffers once compression starts; that
    # allocation failed before with the page cache filling the host, so reclaim memory first.
    docker exec "$CONTAINER_NAME" bash -c "sync; echo 3 > /proc/sys/vm/drop_caches; \
        echo 1 > /sys/devices/system/node/node0/compact; echo 1 > /sys/devices/system/node/node1/compact"
    log "starting vLLM on GPU $GPU, log $OUT_DIR/vllm.log"
    docker exec -w "$REPO_DIR" -d "$CONTAINER_NAME" bash -c "exec > $OUT_DIR/vllm.log 2>&1; \
        export ${SERVE_ENV[*]}; \
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
    grep -aE "Available KV cache memory|GPU KV cache size|Maximum concurrency|Registered" \
        "$OUT_DIR/vllm.log" | sed -E 's/^.*\] //'
}

cpu_ticks() {
    # utime + stime of a /proc stat file, in clock ticks. The command name in field 2 may contain
    # spaces, so everything up to its ')' is cut first; utime and stime are then fields 12 and 13.
    sed 's/.*) //' "$1" | awk '{print $12 + $13}'
}

run_bench() {
    # run_bench <name> <seed> <num prompts> <prompt args...>
    local name=$1 seed=$2 num_prompts=$3
    shift 3
    local start end
    # The kernel's CPU time of the EngineCore main thread (its thread id is the process id) and of
    # the whole process: the analysis checks py-spy's busy main-thread samples against it.
    local main_stat="/proc/$ENGINE_CORE_PID/task/$ENGINE_CORE_PID/stat"
    local process_stat="/proc/$ENGINE_CORE_PID/stat"
    local main_ticks_start main_ticks_end process_ticks_start process_ticks_end
    log "bench $name: $num_prompts requests, seed $seed"
    start=$(stat -c %s "$OUT_DIR/vllm.log")
    main_ticks_start=$(cpu_ticks "$main_stat")
    process_ticks_start=$(cpu_ticks "$process_stat")
    in_container vllm bench serve "${BENCH_ARGS[@]}" "$@" --seed "$seed" --num-prompts "$num_prompts" \
        --result-dir "$OUT_DIR" --result-filename "bench_$name.json" 2>&1 | tee "$OUT_DIR/bench_$name.log"
    main_ticks_end=$(cpu_ticks "$main_stat")
    process_ticks_end=$(cpu_ticks "$process_stat")
    end=$(stat -c %s "$OUT_DIR/vllm.log")
    # The analysis slices the vLLM log by byte offset: the log has '\r' progress output.
    echo "$name $start $end" >> "$OUT_DIR/bench_offsets.txt"
    echo "$name $main_ticks_start $main_ticks_end $process_ticks_start $process_ticks_end" \
        >> "$OUT_DIR/bench_cpu_ticks.txt"
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

start_pyspy() {
    docker exec -w "$REPO_DIR" -d "$CONTAINER_NAME" bash -c "exec taskset -c $PYSPY_CPUS \
        py-spy record --pid $ENGINE_CORE_PID ${PYSPY_ARGS[*]} -o $OUT_DIR/pyspy_enginecore.txt \
        > $OUT_DIR/pyspy_enginecore.stdout 2>&1"
    for _ in $(seq 1 30); do
        PYSPY_PID=$(pgrep -f "^py-spy record --pid $ENGINE_CORE_PID " || true)
        [[ -n "$PYSPY_PID" ]] && grep -q "Sampling process" "$OUT_DIR/pyspy_enginecore.stdout" 2> /dev/null && break
        sleep 1
    done
    if [[ -z "$PYSPY_PID" ]]; then
        log "ERROR: py-spy did not start; see $OUT_DIR/pyspy_enginecore.stdout"
        return 1
    fi
    log "py-spy sampling EngineCore $ENGINE_CORE_PID (py-spy pid $PYSPY_PID)"
}

write_meta() {
    cat > "$OUT_DIR/meta.json" << EOF
{
  "workload": "$WORKLOAD",
  "scenario": "$SCENARIO",
  "model": "$MODEL",
  "input_len": $INPUT_LEN,
  "output_len": $OUTPUT_LEN,
  "concurrency": $CONCURRENCY,
  "num_prompts": $NUM_PROMPTS,
  "seeds": {"warmup": $WARMUP_SEED, "baseline": $BASELINE_SEED, "pyspy": $PYSPY_SEED, "hit": $HIT_SEED},
  "gpu": $GPU,
  "cpu_bind": "$CPU_BIND",
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "max_model_len": $MAX_MODEL_LEN,
  "serve_env": "${SERVE_ENV[*]}",
  "pyspy_rate": $PYSPY_RATE,
  "pyspy_mode": "$PYSPY_MODE",
  "pyspy_args": "${PYSPY_ARGS[*]}",
  "clock_ticks_per_second": $(getconf CLK_TCK),
  "pyspy_cpus": "$PYSPY_CPUS",
  "pyspy_version": "$(in_container py-spy --version)",
  "vllm_version": "$(in_container python3 -c 'import vllm; print(vllm.__version__)' 2> /dev/null | tail -1)",
  "serve_pid": $SERVE_PID,
  "engine_core_pid": $ENGINE_CORE_PID,
  "git_head": "$(git -C "$REPO_DIR" rev-parse --short HEAD)",
  "code_md5": "$CODE_MD5",
  "repo_dir": "$REPO_DIR"
}
EOF
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
log "results: $OUT_DIR"

start_vllm
write_meta
in_container py-spy dump --pid "$ENGINE_CORE_PID" --native > "$OUT_DIR/pyspy_dump_enginecore.txt" 2>&1

run_bench warmup "$WARMUP_SEED" "$WARMUP_PROMPTS" "${MISS_PROMPT_ARGS[@]}"
if [[ "$SCENARIO" == miss ]]; then
    run_bench baseline "$BASELINE_SEED" "$NUM_PROMPTS" "${MISS_PROMPT_ARGS[@]}"
    start_pyspy
    run_bench pyspy "$PYSPY_SEED" "$NUM_PROMPTS" "${MISS_PROMPT_ARGS[@]}"
else
    # The number of prompts changes how the dataset draws its prefix, so the store run has to be
    # the measured bench itself; its first request(s) save the prompt.
    run_bench store "$HIT_SEED" "$NUM_PROMPTS" "${HIT_PROMPT_ARGS[@]}"
    run_bench baseline "$HIT_SEED" "$NUM_PROMPTS" "${HIT_PROMPT_ARGS[@]}"
    start_pyspy
    run_bench pyspy "$HIT_SEED" "$NUM_PROMPTS" "${HIT_PROMPT_ARGS[@]}"
fi
stop_pyspy
stop_vllm

in_container python3 "$BENCH_DIR/analyze_pyspy.py" "$OUT_DIR"
log "done: $OUT_DIR/summary.md"
