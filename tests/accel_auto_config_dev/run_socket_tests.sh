#!/bin/bash
# Tests of the socket-aware DSA calibration and of setvars.sh binding ranks by CUDA_VISIBLE_DEVICES.
# Step 1: CPU/DSA binding only. Step 2: calibration. Step 3: recipe qat-standalone on the host.
# Step 4: benchmarks (run_tests.sh step 2) and vLLM Qwen3.8-27B TP=4 on GPU 0-3 and on GPU 4-7.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/accel_auto_config_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:-$REPO_DIR/$DEV_DIR/results/socket_$(date +%Y%m%d_%H%M%S)}
STEPS=${STEPS:-"1 2 3 4"}
SERVE_READY_TIMEOUT_S=1200
PORT=${PORT:-8123}
GPU_BUSY_MIB=500
MODEL=/media/model-space/Qwen3.8-27B
SERVE_RUN_DIR=""
SERVE_EXEC_PID=""
BASE_ENV=(-e DP_SIZE=1 -e IAXL_KV_COMPRESSION=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=0
    -e IAXL_DSA_GD_ENABLE=1)

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*" | tee -a "$RESULT_DIR/progress.log"
}

summary() {
    echo "$*" | tee -a "$RESULT_DIR/summary.txt"
}

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

# Sources setvars.sh in the container with the given docker exec options; writes <step>/<name>.stdout/.stderr/.result.
probe() {
    local step=$1 name=$2 prefix="$RESULT_DIR/$1/$2" started_ns
    shift 2
    log "setvars probe $name ($*)"
    started_ns=$(date +%s%N)
    docker exec -w "$REPO_DIR" "$@" "$CONTAINER" timeout 600 bash "$DEV_DIR/setvars_probe.sh" \
        >"$prefix.stdout" 2>"$prefix.stderr"
    echo "exit_code $? elapsed_ms $((($(date +%s%N) - started_ns) / 1000000))" >"$prefix.result"
}

probe_value() {
    grep -a "^PROBE $3=" "$RESULT_DIR/$1/$2.stdout" | cut -d= -f2-
}

stderr_line() {
    grep -a -- "$3" "$RESULT_DIR/$1/$2.stderr" | head -1
}

# "cpu <sockets>|dsa <sockets>" of a probe's VLLM_CPU_OMP_THREADS_BIND and KVSHRINK_DSA_DEVICES.
binding_sockets() {
    python3 "$DEV_DIR/binding_sockets.py" "$(probe_value "$1" "$2" VLLM_CPU_OMP_THREADS_BIND)" \
        "$(probe_value "$1" "$2" KVSHRINK_DSA_DEVICES)" | paste -sd'|'
}

used_wqs() {
    grep -a -o "dsa_init: mapped [0-9]* WQ(s) from \$IAXL_DSA_WQS (first [^)]*)" "$RESULT_DIR/$1/$2.stderr" |
        sed 's/.*(first \(.*\))/\1/' | sort -V | tr '\n' ' '
}

binding_case() {
    local name=$1 want_sockets=$2 sockets
    shift 2
    probe step1 "$name" -e TP_SIZE=4 "${BASE_ENV[@]}" "$@"
    sockets=$(binding_sockets step1 "$name")
    if [[ "$(probe_value step1 "$name" setvars_status)" == 0 && "$sockets" == "$want_sockets" ]]; then
        summary "step1 $name: PASS CPUs $(probe_value step1 "$name" VLLM_CPU_OMP_THREADS_BIND)," \
            "DSA $(probe_value step1 "$name" KVSHRINK_DSA_DEVICES), sockets '$sockets'"
        return 0
    fi
    summary "step1 $name: FAIL status $(probe_value step1 "$name" setvars_status), sockets '$sockets' (want '$want_sockets')," \
        "CPUs $(probe_value step1 "$name" VLLM_CPU_OMP_THREADS_BIND), DSA $(probe_value step1 "$name" KVSHRINK_DSA_DEVICES)," \
        "see $RESULT_DIR/step1/$name.*"
    return 1
}

step1() {
    local ok=1 uuids gpu
    mkdir -p "$RESULT_DIR/step1"
    binding_case 1a_unset "cpu 0 0 0 0|dsa 0 0 0 0" || ok=0
    if [[ "$(probe_value step1 1a_unset VLLM_CPU_OMP_THREADS_BIND)" != "0-15|16-31|32-47|48-63" ||
        "$(probe_value step1 1a_unset KVSHRINK_DSA_DEVICES)" != "wq0.0|wq2.0|wq4.0|wq6.0" ]]; then
        summary "step1 1a_unset: FAIL values differ from before the change (0-15|16-31|32-47|48-63, wq0.0|wq2.0|wq4.0|wq6.0)"
        ok=0
    fi
    binding_case 1b_0246 "cpu 0 0 1 1|dsa 0 0 1 1" -e CUDA_VISIBLE_DEVICES=0,2,4,6 || ok=0
    binding_case 1c_4567 "cpu 1 1 1 1|dsa 1 1 1 1" -e CUDA_VISIBLE_DEVICES=4,5,6,7 || ok=0
    uuids=""
    for gpu in 4 5 6 7; do
        uuids+="${uuids:+,}$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")"
    done
    binding_case 1d_4567_uuid "cpu 1 1 1 1|dsa 1 1 1 1" -e CUDA_VISIBLE_DEVICES="$uuids" || ok=0
    if [[ "$(probe_value step1 1d_4567_uuid VLLM_CPU_OMP_THREADS_BIND)" != "$(probe_value step1 1c_4567 VLLM_CPU_OMP_THREADS_BIND)" ||
        "$(probe_value step1 1d_4567_uuid KVSHRINK_DSA_DEVICES)" != "$(probe_value step1 1c_4567 KVSHRINK_DSA_DEVICES)" ]]; then
        summary "step1 1d_4567_uuid: FAIL UUIDs give a different binding than indices 4,5,6,7"
        ok=0
    fi
    probe step1 1e_gpu99 -e TP_SIZE=4 "${BASE_ENV[@]}" -e CUDA_VISIBLE_DEVICES=0,1,2,99
    if [[ "$(probe_value step1 1e_gpu99 setvars_status)" == 1 ]] &&
        grep -a -q "requires 4 GPUs, but only 3 were found among 0,1,2,99" "$RESULT_DIR/step1/1e_gpu99.stderr"; then
        summary "step1 1e_gpu99: PASS setvars.sh returned 1: $(stderr_line step1 1e_gpu99 "requires 4 GPUs")"
    else
        summary "step1 1e_gpu99: FAIL, see $RESULT_DIR/step1/1e_gpu99.*"
        ok=0
    fi
    ((ok == 1))
}

calibration_case() {
    local name=$1 want_workers=$2 want_wqs=$3 tp=$4 dsa_line
    shift 4
    probe step2 "$name" -e IAXL_IN_CONTAINER=1 -e TP_SIZE="$tp" "${BASE_ENV[@]}" "$@"
    dsa_line=$(stderr_line step2 "$name" "\[accel_auto_config\] DSA crossover")
    if [[ "$(probe_value step2 "$name" setvars_status)" == 0 && "$dsa_line" == *"$want_workers"* &&
        "$(used_wqs step2 "$name")" == "$want_wqs" ]]; then
        summary "step2 $name: PASS setvars $(probe_value step2 "$name" setvars_ms) ms, $(stderr_line step2 "$name" "done in")," \
            "WQs used $(used_wqs step2 "$name")"
        summary "    $dsa_line"
        return 0
    fi
    summary "step2 $name: FAIL status $(probe_value step2 "$name" setvars_status), DSA line '$dsa_line'," \
        "WQs used '$(used_wqs step2 "$name")' (want '$want_wqs'), see $RESULT_DIR/step2/$name.*"
    return 1
}

failure_case() {
    local name=$1 want_message=$2 leftover
    shift 2
    probe step2 "$name" -e IAXL_IN_CONTAINER=1 -e TP_SIZE=4 "${BASE_ENV[@]}" "$@"
    leftover=$(pgrep -f "tools/accel_auto_config.py" | tr '\n' ' ')
    if [[ "$(probe_value step2 "$name" setvars_status)" == 1 && -z "$leftover" ]] &&
        grep -a -q -F "$want_message" "$RESULT_DIR/step2/$name.stderr"; then
        summary "step2 $name: PASS setvars.sh returned 1: '$(stderr_line step2 "$name" "failed:")'," \
            "DSA processes started: $(grep -a -c "dsa_init: mapped" "$RESULT_DIR/step2/$name.stderr"), no leftover process"
        return 0
    fi
    summary "step2 $name: FAIL status $(probe_value step2 "$name" setvars_status), leftover '$leftover'," \
        "'$(stderr_line step2 "$name" "failed:")', see $RESULT_DIR/step2/$name.*"
    return 1
}

step2() {
    local ok=1
    mkdir -p "$RESULT_DIR/step2"
    require_gpus_free 0 1 2 3 4 5 6 7 || return 1
    calibration_case 2a_tp4 "4 of 4 workers (socket 0)" "wq0.0 wq2.0 wq4.0 wq6.0 " 4 -e CUDA_VISIBLE_DEVICES=0,1,2,3 || ok=0
    calibration_case 2b_tp8 "4 of 8 workers (socket 0)" "wq0.0 wq2.0 wq4.0 wq6.0 " 8 \
        -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 || ok=0
    calibration_case 2c_0246 "2 of 4 workers (socket 0)" "wq0.0 wq2.0 " 4 -e CUDA_VISIBLE_DEVICES=0,2,4,6 || ok=0
    calibration_case 2d_4567 "4 of 4 workers (socket 1)" "wq8.0 wq10.0 wq12.0 wq14.0 " 4 -e CUDA_VISIBLE_DEVICES=4,5,6,7 || ok=0
    failure_case 2e_dsa_other_socket \
        "[accel_auto_config] failed: rank0: GPU 4 is on socket 1 but DSA wq0.0 is on socket 0; fix KVSHRINK_DSA_DEVICES" \
        -e CUDA_VISIBLE_DEVICES=4,5,6,7 -e "KVSHRINK_DSA_DEVICES=wq0.0|wq2.0|wq4.0|wq6.0" || ok=0
    failure_case 2f_rank3_bad_cpus "[accel_auto_config] failed: dsa_rank3 exited with code 1" \
        -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e "VLLM_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|9000-9015" || ok=0
    failure_case 2g_rank3_bad_wq \
        "[accel_auto_config] failed: rank3: DSA work queue wq99.0 of KVSHRINK_DSA_DEVICES does not exist" \
        -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e "KVSHRINK_DSA_DEVICES=wq0.0|wq2.0|wq4.0|wq99.0" || ok=0
    ((ok == 1))
}

# On the host, as start.sh would: source setvars.sh with recipe qat-standalone for one GPU, print its binding.
recipe_case() {
    local name=$1 gpu=$2 want_cpus=$3 want_wq=$4 prefix="$RESULT_DIR/step3/$1" cpus wq
    log "recipe qat-standalone on the host, CUDA_VISIBLE_DEVICES=$gpu"
    env -i HOME="$HOME" PATH="$PATH" TERM=dumb bash -c "cd '$REPO_DIR' && IAXL_RECIPE=qat-standalone CUDA_VISIBLE_DEVICES=$gpu \
        IAXL_DSA_GD_ENABLE=1 source setvars.sh && echo \"RESULT VLLM_CPU_OMP_THREADS_BIND=\$VLLM_CPU_OMP_THREADS_BIND\" &&
        echo \"RESULT KVSHRINK_DSA_DEVICES=\$KVSHRINK_DSA_DEVICES\"" >"$prefix.log" 2>&1
    cpus=$(grep -a "^RESULT VLLM_CPU_OMP_THREADS_BIND=" "$prefix.log" | cut -d= -f2-)
    wq=$(grep -a "^RESULT KVSHRINK_DSA_DEVICES=" "$prefix.log" | cut -d= -f2-)
    if [[ "$cpus" == "$want_cpus" && "$wq" == "$want_wq" ]]; then
        summary "step3 $name: PASS CPUs $cpus, DSA $wq (same as GPU $gpu's rank in the 8-GPU binding before the change)"
        return 0
    fi
    summary "step3 $name: FAIL CPUs '$cpus' (want $want_cpus), DSA '$wq' (want $want_wq), see $prefix.log"
    return 1
}

step3() {
    local ok=1
    mkdir -p "$RESULT_DIR/step3"
    recipe_case 3a_gpu2 2 32-47 wq4.0 || ok=0
    recipe_case 3b_gpu5 5 80-95 wq10.0 || ok=0
    ((ok == 1))
}

descendants_of() {
    local child
    for child in $(ps -o pid= --ppid "$1"); do
        echo "$child"
        descendants_of "$child"
    done
}

stop_serve() {
    [[ -z "$SERVE_RUN_DIR" ]] && return 0
    local script_pid vllm_pid="" pid tree_pids alive waited_s=0
    script_pid=$(cat "$SERVE_RUN_DIR/serve.pid" 2>/dev/null)
    for pid in $(descendants_of "${script_pid:-0}"); do
        if ps -o args= -p "$pid" | grep -q "vllm serve"; then
            vllm_pid=$pid
            break
        fi
    done
    if [[ -n "$vllm_pid" ]]; then
        tree_pids=$(echo "$vllm_pid" $(descendants_of "$vllm_pid"))
        log "stopping vLLM pid $vllm_pid"
        docker exec "$CONTAINER" kill -TERM "$vllm_pid"
        while true; do
            alive=$(ps -o pid= -p "${tree_pids// /,}" | tr '\n' ' ')
            [[ -z "${alive// /}" ]] && break
            if ((waited_s >= 120)); then
                docker exec "$CONTAINER" kill -9 $alive
                break
            fi
            sleep 5
            waited_s=$((waited_s + 5))
        done
    fi
    [[ -n "$SERVE_EXEC_PID" ]] && wait "$SERVE_EXEC_PID" 2>/dev/null
    cp "$REPO_DIR/log.kvshrink-vllm" "$SERVE_RUN_DIR/log.kvshrink-vllm" 2>/dev/null
    SERVE_RUN_DIR=""
    SERVE_EXEC_PID=""
}

serve_case() {
    local name=$1 gpus=$2 want_socket=$3 run_dir="$RESULT_DIR/step4/$1" waited_s=0 cpus wqs sockets
    mkdir -p "$run_dir"
    require_gpus_free ${gpus//,/ } || return 1
    log "starting serve $name on GPU $gpus, log $run_dir/serve.log"
    docker exec -w "$REPO_DIR" -e IAXL_IN_CONTAINER=1 -e MODEL="$MODEL" -e TP_SIZE=4 "${BASE_ENV[@]}" \
        -e CUDA_VISIBLE_DEVICES="$gpus" -e PORT="$PORT" -e IAXL_DDR_POOL_SIZE_GB=16 "$CONTAINER" \
        bash -c "echo \$\$ > $run_dir/serve.pid; exec bash examples/kvshrink-vllm-serve.sh" >"$run_dir/serve.log" 2>&1 &
    SERVE_EXEC_PID=$!
    SERVE_RUN_DIR=$run_dir
    while ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVE_EXEC_PID" 2>/dev/null || ((waited_s >= SERVE_READY_TIMEOUT_S)); then
            stop_serve
            summary "step4 $name: FAIL vLLM not ready after ${waited_s}s, see $run_dir/serve.log"
            return 1
        fi
        sleep 5
        waited_s=$((waited_s + 5))
    done
    log "vLLM ready after ${waited_s}s, sending requests"
    timeout 1800 python3 tests/dsa_fallback_dev/e2e_requests.py --port "$PORT" --model "$MODEL" \
        --output "$run_dir/outputs.json" >"$run_dir/requests.log" 2>&1
    local requests_exit=$?
    stop_serve
    if ((requests_exit != 0)); then
        summary "step4 $name: FAIL requests failed, see $run_dir/requests.log"
        return 1
    fi
    cpus=$(grep -a -o "^  VLLM_CPU_OMP_THREADS_BIND=.*" "$run_dir/serve.log" | head -1 | cut -d= -f2-)
    wqs=$(grep -a -o "^  KVSHRINK_DSA_DEVICES=.*" "$run_dir/serve.log" | head -1 | cut -d= -f2-)
    sockets=$(python3 "$DEV_DIR/binding_sockets.py" "$cpus" "$wqs" | paste -sd'|')
    python3 "$DEV_DIR/e2e_check.py" "$run_dir" qat 4 >"$run_dir/check.txt" 2>&1
    if [[ "$sockets" == "cpu $want_socket $want_socket $want_socket $want_socket|dsa $want_socket $want_socket $want_socket $want_socket" ]]; then
        echo "PASS vLLM bound to socket $want_socket: CPUs $cpus, DSA $wqs" >>"$run_dir/check.txt"
    else
        echo "FAIL vLLM binding sockets '$sockets' (want socket $want_socket): CPUs $cpus, DSA $wqs" >>"$run_dir/check.txt"
    fi
    if grep -q "^FAIL " "$run_dir/check.txt"; then
        summary "step4 $name: FAIL, see $run_dir/check.txt"
    else
        summary "step4 $name: PASS (ready after ${waited_s}s), see $run_dir/check.txt"
    fi
    summary "$(grep -E '^(PASS|FAIL) |^accel_auto_config done|^DSA summaries|^QAT ranked|^vLLM \[kv_zip\]' "$run_dir/check.txt" | sed 's|^|    |')"
    summary "    $(grep -a -o "\[accel_auto_config\] DSA crossover.*" "$run_dir/serve.log" | head -1)"
    grep -q "^FAIL " "$run_dir/check.txt" && return 1
    return 0
}

step4() {
    local ok=1
    mkdir -p "$RESULT_DIR/step4"
    require_gpus_free 0 1 || return 1
    log "benchmarks: run_tests.sh step 2"
    RESULT_DIR="$RESULT_DIR/step4/benchmarks" STEPS=2 bash "$DEV_DIR/run_tests.sh" >"$RESULT_DIR/step4/benchmarks.log" 2>&1
    if grep -q "FAIL" "$RESULT_DIR/step4/benchmarks/summary.txt"; then
        summary "step4 benchmarks: FAIL, see $RESULT_DIR/step4/benchmarks/summary.txt"
        ok=0
    else
        summary "step4 benchmarks: PASS ($(grep -c PASS "$RESULT_DIR/step4/benchmarks/summary.txt") checks)"
    fi
    summary "$(sed 's|^|    |' "$RESULT_DIR/step4/benchmarks/summary.txt")"
    serve_case 4a_gpu0123 0,1,2,3 0 || ok=0
    serve_case 4b_gpu4567 4,5,6,7 1 || ok=0
    ((ok == 1))
}

trap stop_serve EXIT

mkdir -p "$RESULT_DIR"
cd "$REPO_DIR" || exit 1
log "start: steps=$STEPS result_dir=$RESULT_DIR"
{
    echo "git: $(git rev-parse HEAD) branch $(git branch --show-current)"
    git diff --stat
    md5sum tools/accel_auto_config.py tools/auto_config.sh setvars.sh examples/kvshrink-vllm-serve.sh \
        examples/recipes/qat-standalone/setvars.pre.sh "$DEV_DIR"/*.py "$DEV_DIR"/*.sh
} >"$RESULT_DIR/env.txt"

for step in $STEPS; do
    step_started=$(date +%s)
    log "step $step started"
    if "step$step"; then
        log "step $step PASS ($(($(date +%s) - step_started))s)"
    else
        log "step $step FAIL ($(($(date +%s) - step_started))s), see $RESULT_DIR/summary.txt"
    fi
done
log "ALL DONE. Summary: $RESULT_DIR/summary.txt"
