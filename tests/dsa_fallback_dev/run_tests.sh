#!/bin/bash
# DSA auto fallback tests on 6000d: step 1 calibration, 2 forced crossovers, 3 KVStore calibration, 4 vLLM TP=8.
# Usage: RESULT_DIR=<dir> bash tests/dsa_fallback_dev/run_tests.sh   (STEPS="1 2 3 4", DRY_RUN=1 prints commands)
# Needs the container started by ./start.sh in another tmux window, logging to $RESULT_DIR/start.log,
# with MODEL=/media/model-space so both models are mounted.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/dsa_fallback_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
STEPS=${STEPS:-"1 2 3 4"}
DRY_RUN=${DRY_RUN:-0}
CONTAINER_READY_TIMEOUT_S=${CONTAINER_READY_TIMEOUT_S:-1800}
SERVE_READY_TIMEOUT_S=${SERVE_READY_TIMEOUT_S:-1200}
PORT=${PORT:-8123}
GPU_BUSY_MIB=500
SHAPE_784K=(2 64 784 4 128)
INT64_MAX=9223372036854775807
MODELS=${MODELS:-"/media/model-space/Qwen3-32B /media/model-space/Qwen3.8-27B"}
# Same KVStore settings for every step: no compression, so only the copy path is tested.
NOZIP_ENV=(-e IAXL_KV_COMPRESSION=0 -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=0)
SERVE_RUN_DIR=""
SERVE_EXEC_PID=""

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*" | tee -a "$RESULT_DIR/progress.log"
}

summary() {
    echo "$*" | tee -a "$RESULT_DIR/summary.txt"
}

wait_container_ready() {
    local waited_s=0 start_log="$RESULT_DIR/start.log" container_start_line=""
    log "waiting for container $CONTAINER + pip install -e . (watching $start_log)"
    while true; do
        # The typed command line "echo START_SH_EXIT=$?" is in start.log too, so require a digit.
        if grep -a -q -E "START_SH_EXIT=[0-9]+" "$start_log" 2>/dev/null; then
            log "ERROR: start.sh exited, see $start_log"
            return 1
        fi
        # docker build output also has "Successfully installed", so only read lines after the container is up.
        if [[ -z "$container_start_line" && "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == true ]]; then
            container_start_line=$(wc -l <"$start_log")
            log "container $CONTAINER running, waiting for pip install -e . (start.log line $container_start_line)"
        fi
        if [[ -n "$container_start_line" ]]; then
            if tail -n +"$container_start_line" "$start_log" | grep -a -q -E "error: subprocess-exited-with-error|^ERROR: (Failed|Could not)"; then
                log "ERROR: pip install -e . failed in the container, see $start_log"
                return 1
            fi
            if tail -n +"$container_start_line" "$start_log" | grep -a -q "Successfully installed"; then
                log "container $CONTAINER ready (waited ${waited_s}s)"
                return 0
            fi
        fi
        if ((waited_s >= CONTAINER_READY_TIMEOUT_S)); then
            log "ERROR: container not ready after ${CONTAINER_READY_TIMEOUT_S}s, see $start_log"
            return 1
        fi
        sleep 10
        waited_s=$((waited_s + 10))
    done
}

# Someone else on our GPUs would distort the runs, so stop instead of testing.
require_gpus_free() {
    local gpu used_mib
    for gpu in "$@"; do
        used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        if ((used_mib > GPU_BUSY_MIB)); then
            log "ERROR: GPU$gpu busy (${used_mib} MiB used)"
            return 1
        fi
    done
}

# Runs one command in the container; the timeout runs inside so a hang cannot keep holding the GPU.
run_in_container() {
    local log_file=$1 timeout_s=$2
    shift 2
    local command=(docker exec -w "$REPO_DIR" "$@")
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: ${command[*]} (timeout ${timeout_s}s) > $log_file"
        return 0
    fi
    "${command[@]}" >"$log_file" 2>&1
}

with_setvars() {
    echo bash "$DEV_DIR/with_setvars.sh" timeout "$1"
}

step1() {
    local dir="$RESULT_DIR/step1" ok=1 enabled
    mkdir -p "$dir"
    require_gpus_free 0 || return 1
    run_in_container "$dir/switch_default.log" 120 -e IAXL_DSA_GD_ENABLE=1 "$CONTAINER" \
        timeout 120 python3 -c "import torch; from iaxl import torch_ext as t; print('ENABLED', t.dsa_auto_fallback_enabled())"
    run_in_container "$dir/switch_fallback0.log" 120 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_DSA_AUTO_FALLBACK_ENABLE=0 "$CONTAINER" \
        timeout 120 python3 -c "import torch; from iaxl import torch_ext as t; print('ENABLED', t.dsa_auto_fallback_enabled())"
    run_in_container "$dir/switch_gd0.log" 120 -e IAXL_DSA_GD_ENABLE=0 "$CONTAINER" \
        timeout 120 python3 -c "import torch; from iaxl import torch_ext as t; print('ENABLED', t.dsa_auto_fallback_enabled())"
    run_in_container "$dir/calibrate.log" 300 -e TP_SIZE=1 -e IAXL_DSA_GD_ENABLE=1 "${NOZIP_ENV[@]}" "$CONTAINER" \
        $(with_setvars 300) python3 "$DEV_DIR/calibrate_once.py" --gpu 0 --calls 3
    [[ "$DRY_RUN" == 1 ]] && return 0
    for name in switch_default:True switch_fallback0:False switch_gd0:False; do
        enabled=$(grep -a -o "ENABLED .*" "$dir/${name%%:*}.log")
        if [[ "$enabled" == "ENABLED ${name##*:}" ]]; then
            summary "step1 ${name%%:*}: PASS ($enabled)"
        else
            summary "step1 ${name%%:*}: FAIL (got '$enabled', expected ENABLED ${name##*:}, see $dir/${name%%:*}.log)"
            ok=0
        fi
    done
    local calls distinct
    calls=$(grep -a -c "^CALL " "$dir/calibrate.log")
    distinct=$(grep -a "^CALL " "$dir/calibrate.log" | awk '{print $3, $4}' | sort -u | wc -l)
    if ((calls == 3 && distinct == 1)); then
        summary "step1 calibrate: PASS 3 calls, identical result: $(grep -a "^CALL 1" "$dir/calibrate.log")"
    else
        summary "step1 calibrate: FAIL $calls calls, $distinct distinct results, see $dir/calibrate.log"
        ok=0
    fi
    summary "step1 calibrate details: $(grep -a -E "^(CALL|FREE_MIB)" "$dir/calibrate.log" | tr '\n' ';')"
    ((ok == 1))
}

step2() {
    local dir="$RESULT_DIR/step2" ok=1 combo h2d d2h h2d_backend d2h_backend log_file
    mkdir -p "$dir"
    # H2D crossover, D2H crossover, expected H2D backend, expected D2H backend
    for combo in "$INT64_MAX $INT64_MAX DSA DSA" "0 0 CUDA CUDA" "$INT64_MAX 0 DSA CUDA" "0 $INT64_MAX CUDA DSA"; do
        read -r h2d d2h h2d_backend d2h_backend <<<"$combo"
        log_file="$dir/h2d_${h2d_backend}_d2h_${d2h_backend}.log"
        require_gpus_free 0 || return 1
        run_in_container "$log_file" 600 -e TP_SIZE=1 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_DSA_AUTO_FALLBACK_ENABLE=0 \
            -e IAXL_DDR_POOL_SIZE_GB=8 -e IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0 "${NOZIP_ENV[@]}" "$CONTAINER" \
            $(with_setvars 600) python3 "$DEV_DIR/forced_crossover_kvstore.py" --h2d-crossover "$h2d" --d2h-crossover "$d2h" \
            --data-source mock --shape "${SHAPE_784K[@]}"
        [[ "$DRY_RUN" == 1 ]] && continue
        local result backend_lines
        result=$(grep -a -o "RESULT .*" "$log_file")
        backend_lines=$(grep -a -o "copy_chunks_batch [A-Z0-9]*: using [A-Za-z ]* backend" "$log_file" | sort | tr '\n' ';')
        if [[ "$result" == "RESULT PASS" ]] && check_backends "$log_file" H2D "$h2d_backend" && check_backends "$log_file" D2H "$d2h_backend"; then
            summary "step2 H2D=$h2d_backend D2H=$d2h_backend: PASS ($(grep -a "^COMPARE" "$log_file")) backends: $backend_lines"
        else
            summary "step2 H2D=$h2d_backend D2H=$d2h_backend: FAIL ($result) backends: $backend_lines, see $log_file"
            ok=0
        fi
    done
    ((ok == 1))
}

# The direction logs its expected backend and never the other one.
check_backends() {
    local log_file=$1 direction=$2 expected=$3 other
    if [[ "$expected" == DSA ]]; then
        expected="Intel DSA"
        other="CUDA"
    else
        other="Intel DSA"
    fi
    grep -a -q "copy_chunks_batch $direction: using $expected backend" "$log_file" &&
        ! grep -a -q "copy_chunks_batch $direction: using $other backend" "$log_file"
}

step3() {
    local dir="$RESULT_DIR/step3" ok=1 ranks name log_file crossover_lines verify_line
    mkdir -p "$dir"
    for name in ranks1_16k ranks1_784k ranks2_16k; do
        ranks=${name:5:1}
        log_file="$dir/$name.log"
        local shape_args=()
        [[ "$name" == *784k ]] && shape_args=(--shape "${SHAPE_784K[@]}")
        require_gpus_free 0 1 || return 1
        run_in_container "$log_file" 600 "$CONTAINER" \
            timeout 600 bash benchmark/kvstore/kvstore_benchmark.sh --dsa --ranks "$ranks" --data-source mock "${shape_args[@]}"
        local exit_code=$?
        [[ "$DRY_RUN" == 1 ]] && continue
        crossover_lines=$(grep -a -c "DSA crossover measured by rank 0" "$log_file")
        verify_line=$(grep -a -o "Verification: .*" "$log_file" | tail -1)
        if ((exit_code == 0 && crossover_lines == ranks)) && [[ "$verify_line" == "Verification: passed" ]]; then
            summary "step3 $name: PASS ($crossover_lines crossover lines, $verify_line)"
        else
            summary "step3 $name: FAIL (exit $exit_code, $crossover_lines crossover lines, '$verify_line'), see $log_file"
            ok=0
        fi
        summary "step3 $name crossover lines: $(grep -a -o "\[rank[0-9]*\] DSA crossover.*" "$log_file" | tr '\n' ';')"
        summary "step3 $name backends: $(grep -a -o "copy_chunks_batch [A-Z0-9]*: using [A-Za-z ]* backend" "$log_file" | sort | uniq -c | tr '\n' ';')"
    done
    log_file="$dir/timeout.log"
    require_gpus_free 0 || return 1
    run_in_container "$log_file" 300 -e TP_SIZE=1 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_API_TIMEOUT=5 -e IAXL_DDR_POOL_SIZE_GB=1 \
        "${NOZIP_ENV[@]}" "$CONTAINER" $(with_setvars 300) python3 "$DEV_DIR/timeout_kvstore.py"
    [[ "$DRY_RUN" == 1 ]] && return 0
    if grep -a -q "^RAISED .*worker rank 0 did not answer GET /v1/health" "$log_file"; then
        summary "step3 timeout: PASS ($(grep -a "^RAISED" "$log_file"))"
    else
        summary "step3 timeout: FAIL, see $log_file"
        ok=0
    fi
    ((ok == 1))
}

descendants_of() {
    local child
    for child in $(ps -o pid= --ppid "$1"); do
        echo "$child"
        descendants_of "$child"
    done
}

# Stops the vLLM started by serve_one (exact PIDs only: the container shares the host PID namespace).
stop_serve() {
    [[ -z "$SERVE_RUN_DIR" ]] && return 0
    local script_pid vllm_pid tree_pids alive waited_s=0
    script_pid=$(cat "$SERVE_RUN_DIR/serve.pid" 2>/dev/null)
    vllm_pid=$(pgrep -P "${script_pid:-0}" -f "vllm serve" | head -1)
    if [[ -n "$vllm_pid" ]]; then
        tree_pids=$(echo "$vllm_pid" $(descendants_of "$vllm_pid"))
        log "stopping vLLM pid $vllm_pid (process tree: $tree_pids)"
        docker exec "$CONTAINER" kill -TERM "$vllm_pid"
        while true; do
            alive=$(ps -o pid= -p "${tree_pids// /,}" | tr '\n' ' ')
            [[ -z "${alive// /}" ]] && break
            if ((waited_s >= 120)); then
                log "vLLM processes still alive after 120s, kill -9: $alive"
                docker exec "$CONTAINER" kill -9 $alive
                break
            fi
            sleep 5
            waited_s=$((waited_s + 5))
        done
    fi
    [[ -n "$SERVE_EXEC_PID" ]] && wait "$SERVE_EXEC_PID" 2>/dev/null
    SERVE_RUN_DIR=""
}

serve_one() {
    local model=$1 fallback=$2 run_dir=$3 waited_s=0
    mkdir -p "$run_dir"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: MODEL=$model TP_SIZE=8 IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback PORT=$PORT examples/kvshrink-vllm-serve.sh > $run_dir/serve.log"
        echo "DRY_RUN: python3 $DEV_DIR/e2e_requests.py --port $PORT --model $model --output $run_dir/outputs.json"
        return 0
    fi
    require_gpus_free 0 1 2 3 4 5 6 7 || return 1
    log "starting vLLM: $model TP=8 IAXL_DSA_AUTO_FALLBACK_ENABLE=$fallback, log $run_dir/serve.log"
    docker exec -w "$REPO_DIR" -e MODEL="$model" -e TP_SIZE=8 -e PORT="$PORT" -e IAXL_DSA_GD_ENABLE=1 \
        -e IAXL_DSA_AUTO_FALLBACK_ENABLE="$fallback" -e IAXL_DDR_POOL_SIZE_GB=16 "${NOZIP_ENV[@]}" "$CONTAINER" \
        bash -c "echo \$\$ > $run_dir/serve.pid; exec bash examples/kvshrink-vllm-serve.sh" >"$run_dir/serve.log" 2>&1 &
    SERVE_EXEC_PID=$!
    SERVE_RUN_DIR=$run_dir
    while ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVE_EXEC_PID" 2>/dev/null; then
            log "ERROR: vLLM exited before it was ready, see $run_dir/serve.log"
            SERVE_RUN_DIR=""
            return 1
        fi
        if ((waited_s >= SERVE_READY_TIMEOUT_S)); then
            log "ERROR: vLLM not ready after ${SERVE_READY_TIMEOUT_S}s, see $run_dir/serve.log"
            stop_serve
            return 1
        fi
        sleep 5
        waited_s=$((waited_s + 5))
    done
    log "vLLM ready after ${waited_s}s, sending requests"
    echo "ready_after_s $waited_s" >"$run_dir/ready.txt"
    if ! timeout 1800 python3 "$DEV_DIR/e2e_requests.py" --port "$PORT" --model "$model" \
        --output "$run_dir/outputs.json" >"$run_dir/requests.log" 2>&1; then
        log "ERROR: requests failed, see $run_dir/requests.log"
        stop_serve
        return 1
    fi
    stop_serve
}

step4() {
    local dir="$RESULT_DIR/step4" ok=1 model model_dir fallback
    mkdir -p "$dir"
    for model in $MODELS; do
        model_dir="$dir/$(basename "$model")"
        for fallback in 1 0; do
            serve_one "$model" "$fallback" "$model_dir/fallback$fallback" || return 1
        done
        [[ "$DRY_RUN" == 1 ]] && continue
        if python3 "$DEV_DIR/e2e_check.py" "$model_dir" >"$model_dir/check.txt" 2>&1; then
            summary "step4 $(basename "$model"): PASS, see $model_dir/check.txt"
        else
            summary "step4 $(basename "$model"): FAIL, see $model_dir/check.txt"
            ok=0
        fi
        summary "$(grep -E "^(PASS|FAIL) " "$model_dir/check.txt" | sed "s|^|step4 $(basename "$model") |")"
    done
    ((ok == 1))
}

trap stop_serve EXIT

mkdir -p "$RESULT_DIR"
cd "$REPO_DIR" || exit 1
log "start: steps=$STEPS result_dir=$RESULT_DIR"
if [[ "$DRY_RUN" != 1 ]]; then
    wait_container_ready || exit 1
    {
        echo "git: $(git rev-parse HEAD) branch $(git branch --show-current)"
        echo "image: $(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
        git diff --stat
        md5sum iaxl/csrc/kv_xfer/{cuda,dsa}.cpp iaxl/csrc/include/kv_xfer.h iaxl/csrc/torch_ext/torch_ext.cpp \
            iaxl/kvstore/{api,kvstore}.py "$DEV_DIR"/*.py "$DEV_DIR"/*.sh
    } >"$RESULT_DIR/env.txt"
fi

for step in $STEPS; do
    step_started=$(date +%s)
    log "step $step started"
    if "step$step"; then
        log "step $step PASS ($(($(date +%s) - step_started))s)"
    else
        log "step $step FAIL ($(($(date +%s) - step_started))s), see $RESULT_DIR/summary.txt"
        # Step 3's two-rank kvstore benchmark is expected to fail; the other steps build on each other.
        [[ "$step" == 3 ]] || exit 1
    fi
done
log "ALL DONE. Summary: $RESULT_DIR/summary.txt"
