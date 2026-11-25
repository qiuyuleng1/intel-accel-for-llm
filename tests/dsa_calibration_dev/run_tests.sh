#!/bin/bash
# DSA calibration tests on 6000d: step 1 runs tools/calibration/dsa_calibration.py alone, step 2 runs vLLM
# (Qwen3.8-27B TP=4, no compression) through examples/kvshrink-vllm-serve.sh.
# Usage: bash tests/dsa_calibration_dev/run_tests.sh   (STEPS="1 2", RESULT_DIR=<dir>, DRY_RUN=1 prints commands)
# Needs the iaxl.vllm container (started with MODEL=/media/model-space) and the extension already built.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/dsa_calibration_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:-$REPO_DIR/$DEV_DIR/results/tests_$(date +%Y%m%d_%H%M%S)}
STEPS=${STEPS:-"1 2"}
DRY_RUN=${DRY_RUN:-0}
SERVE_READY_TIMEOUT_S=1200
SERVE_FAIL_TIMEOUT_S=180
PORT=${PORT:-8123}
GPU_BUSY_MIB=500
MODEL=${SERVE_MODEL:-/media/model-space/Qwen3.8-27B}
SERVE_TP_SIZE=${SERVE_TP_SIZE:-4}
SERVE_GPUS=${SERVE_GPUS:-0,1,2,3}
# No compression, so only the copy path is tested.
NOZIP_ENV=(-e IAXL_KV_COMPRESSION=0 -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=0)
SERVE_RUN_DIR=""
SERVE_EXEC_PID=""

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S') UTC] $*" | tee -a "$RESULT_DIR/progress.log"
}

summary() {
    echo "$*" | tee -a "$RESULT_DIR/summary.txt"
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

# Runs the calibration once in the container: calibrate <name> [docker exec -e options].
calibrate() {
    local name=$1
    shift
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: docker exec -e IAXL_DSA_GD_ENABLE=1 $* $CONTAINER run_calibration.sh $RESULT_DIR/step1/$name"
        return 0
    fi
    log "calibration $name ($*)"
    docker exec -w "$REPO_DIR" -e IAXL_DSA_GD_ENABLE=1 "${NOZIP_ENV[@]}" "$@" "$CONTAINER" \
        bash "$DEV_DIR/run_calibration.sh" "$RESULT_DIR/step1/$name"
}

# A normal run: exit 0, one result line on stdout, one main-process line, every worker used real DSA in both
# directions and the DSA work queues are exactly the per-rank ones from setvars.sh.
check_normal_run() {
    local name=$1 workers=$2 prefix="$RESULT_DIR/step1/$1" exit_code elapsed_ms result_lines stdout_lines
    local main_lines summary_line dsa_h2d dsa_d2h cuda_lines expected_wqs used_wqs
    read -r _ exit_code _ elapsed_ms <"$prefix.result"
    stdout_lines=$(wc -l <"$prefix.stdout")
    result_lines=$(grep -a -c -E '^[0-9]+ [0-9]+$' "$prefix.stdout")
    main_lines=$(grep -a -c "^\[dsa_calibration\]" "$prefix.stderr")
    summary_line=$(grep -a "^\[dsa_calibration\] crossover" "$prefix.stderr")
    dsa_h2d=$(grep -a -c "copy_chunks_batch H2D: using Intel DSA backend" "$prefix.stderr")
    dsa_d2h=$(grep -a -c "copy_chunks_batch D2H: using Intel DSA backend" "$prefix.stderr")
    cuda_lines=$(grep -a -c "using CUDA backend" "$prefix.stderr")
    expected_wqs=$(grep -a -o "KVSHRINK_DSA_DEVICES=[^ ]*" "$prefix.setvars" | head -1 | cut -d= -f2 | tr '|' '\n' |
        cut -d, -f1 | sort | tr '\n' ' ')
    used_wqs=$(grep -a -o "dsa_init: mapped [0-9]* WQ(s) from \$IAXL_DSA_WQS (first [^)]*)" "$prefix.stderr" |
        sed 's/.*(first \(.*\))/\1/' | sort | tr '\n' ' ')
    if ((exit_code == 0 && stdout_lines == 1 && result_lines == 1 && main_lines == 1 && dsa_h2d == workers &&
        dsa_d2h == workers && cuda_lines == 0)) && [[ "$expected_wqs" == "$used_wqs" ]]; then
        summary "step1 $name: PASS ${elapsed_ms} ms, stdout '$(cat "$prefix.stdout")', $summary_line, WQs $used_wqs"
        return 0
    fi
    summary "step1 $name: FAIL exit $exit_code, stdout $stdout_lines lines ($result_lines result), main lines $main_lines," \
        "DSA lines H2D $dsa_h2d D2H $dsa_d2h (want $workers), CUDA lines $cuda_lines," \
        "WQs used '$used_wqs' expected '$expected_wqs', see $prefix.*"
    return 1
}

step1() {
    local ok=1 run exit_code elapsed_ms failed_line leftover
    mkdir -p "$RESULT_DIR/step1"
    require_gpus_free 0 1 2 3 4 5 6 7 || return 1
    calibrate tp4_run1 -e TP_SIZE=4 -e DP_SIZE=1
    [[ "$DRY_RUN" == 1 ]] || check_normal_run tp4_run1 4 || ok=0
    for run in 2 3; do
        calibrate "tp4_run$run" -e TP_SIZE=4 -e DP_SIZE=1
        [[ "$DRY_RUN" == 1 ]] || check_normal_run "tp4_run$run" 4 || ok=0
    done
    calibrate tp2_dp2 -e TP_SIZE=2 -e DP_SIZE=2
    [[ "$DRY_RUN" == 1 ]] || check_normal_run tp2_dp2 4 || ok=0
    calibrate tp8 -e TP_SIZE=8 -e DP_SIZE=1
    [[ "$DRY_RUN" == 1 ]] || check_normal_run tp8 8 || ok=0
    # Rank 3 gets a GPU that does not exist, so it dies while the other ranks wait for it.
    calibrate tp4_rank3_fails -e TP_SIZE=4 -e DP_SIZE=1 -e CALIBRATION_GPUS=0,1,2,99
    [[ "$DRY_RUN" == 1 ]] && return 0
    read -r _ exit_code _ elapsed_ms <"$RESULT_DIR/step1/tp4_rank3_fails.result"
    failed_line=$(grep -a "^\[dsa_calibration\] failed: rank3" "$RESULT_DIR/step1/tp4_rank3_fails.stderr")
    leftover=$(pgrep -f "tools/calibration/dsa_calibration.py" | tr '\n' ' ')
    if ((exit_code != 0 && elapsed_ms < 60000)) && [[ -n "$failed_line" && -z "$leftover" ]]; then
        summary "step1 tp4_rank3_fails: PASS exit $exit_code after ${elapsed_ms} ms, '$failed_line', no leftover process"
    else
        summary "step1 tp4_rank3_fails: FAIL exit $exit_code after ${elapsed_ms} ms, '$failed_line', leftover '$leftover'"
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

# Stops the vLLM started by start_serve; vLLM runs in the serve script's { ... } | tee subshell, so search the tree.
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
    cp "$REPO_DIR/log.kvshrink-vllm" "$SERVE_RUN_DIR/log.kvshrink-vllm" 2>/dev/null
    SERVE_RUN_DIR=""
    SERVE_EXEC_PID=""
}

# start_serve <run_dir> <IAXL_DSA_GD_ENABLE> <CUDA_VISIBLE_DEVICES>
start_serve() {
    local run_dir=$1 dsa=$2 gpus=$3
    mkdir -p "$run_dir"
    log "starting serve: $MODEL TP=$SERVE_TP_SIZE GPUs $gpus IAXL_DSA_GD_ENABLE=$dsa, log $run_dir/serve.log"
    docker exec -w "$REPO_DIR" -e MODEL="$MODEL" -e TP_SIZE="$SERVE_TP_SIZE" -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES="$gpus" \
        -e PORT="$PORT" -e IAXL_DSA_GD_ENABLE="$dsa" -e IAXL_DDR_POOL_SIZE_GB=16 "${NOZIP_ENV[@]}" "$CONTAINER" \
        bash -c "echo \$\$ > $run_dir/serve.pid; exec bash examples/kvshrink-vllm-serve.sh" >"$run_dir/serve.log" 2>&1 &
    SERVE_EXEC_PID=$!
    SERVE_RUN_DIR=$run_dir
}

# serve_and_check <name> <IAXL_DSA_GD_ENABLE>: start vLLM, send 3 rounds of requests, stop it, run e2e_check.py.
serve_and_check() {
    local name=$1 dsa=$2 run_dir="$RESULT_DIR/step2/$1" waited_s=0
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: serve $name IAXL_DSA_GD_ENABLE=$dsa GPUs $SERVE_GPUS, e2e_requests.py, e2e_check.py"
        return 0
    fi
    require_gpus_free ${SERVE_GPUS//,/ } || return 1
    start_serve "$run_dir" "$dsa" "$SERVE_GPUS"
    while ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVE_EXEC_PID" 2>/dev/null; then
            log "ERROR: vLLM exited before it was ready, see $run_dir/serve.log"
            stop_serve
            summary "step2 $name: FAIL vLLM exited before ready, see $run_dir/serve.log"
            return 1
        fi
        if ((waited_s >= SERVE_READY_TIMEOUT_S)); then
            log "ERROR: vLLM not ready after ${SERVE_READY_TIMEOUT_S}s"
            stop_serve
            summary "step2 $name: FAIL not ready after ${SERVE_READY_TIMEOUT_S}s, see $run_dir/serve.log"
            return 1
        fi
        sleep 5
        waited_s=$((waited_s + 5))
    done
    log "vLLM ready after ${waited_s}s, sending requests"
    echo "ready_after_s $waited_s" >"$run_dir/ready.txt"
    timeout 1800 python3 tests/dsa_fallback_dev/e2e_requests.py --port "$PORT" --model "$MODEL" \
        --output "$run_dir/outputs.json" >"$run_dir/requests.log" 2>&1
    local requests_exit=$?
    stop_serve
    if ((requests_exit != 0)); then
        summary "step2 $name: FAIL requests failed, see $run_dir/requests.log"
        return 1
    fi
    if python3 "$DEV_DIR/e2e_check.py" "$run_dir" "$dsa" "$SERVE_TP_SIZE" >"$run_dir/check.txt" 2>&1; then
        summary "step2 $name: PASS (ready after ${waited_s}s), see $run_dir/check.txt"
        summary "$(grep -E "^(PASS|FAIL) " "$run_dir/check.txt" | sed "s|^|    |")"
        return 0
    fi
    summary "step2 $name: FAIL, see $run_dir/check.txt"
    summary "$(grep -E "^(PASS|FAIL) " "$run_dir/check.txt" | sed "s|^|    |")"
    return 1
}

# Calibration fails on rank 3 (GPU 99 does not exist): the serve script must stop before vLLM, with a non-zero exit.
serve_calibration_fails() {
    local run_dir="$RESULT_DIR/step2/calibration_fails" waited_s=0 exit_code failed_line vllm_lines
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN: serve calibration_fails IAXL_DSA_GD_ENABLE=1 GPUs 0,1,2,99"
        return 0
    fi
    require_gpus_free 0 1 2 || return 1
    start_serve "$run_dir" 1 0,1,2,99
    while kill -0 "$SERVE_EXEC_PID" 2>/dev/null; do
        if ((waited_s >= SERVE_FAIL_TIMEOUT_S)); then
            log "ERROR: serve still running after ${SERVE_FAIL_TIMEOUT_S}s"
            stop_serve
            summary "step2 calibration_fails: FAIL serve still running after ${SERVE_FAIL_TIMEOUT_S}s"
            return 1
        fi
        sleep 2
        waited_s=$((waited_s + 2))
    done
    wait "$SERVE_EXEC_PID"
    exit_code=$?
    SERVE_EXEC_PID=""
    stop_serve
    failed_line=$(grep -a "\[dsa_calibration\] failed" "$run_dir/log.kvshrink-vllm")
    vllm_lines=$(grep -a -c "APIServer\|EngineCore" "$run_dir/serve.log")
    if ((exit_code != 0 && vllm_lines == 0)) && [[ -n "$failed_line" ]]; then
        summary "step2 calibration_fails: PASS exit $exit_code after ~${waited_s}s, no vLLM lines, log.kvshrink-vllm has '$failed_line'"
        return 0
    fi
    summary "step2 calibration_fails: FAIL exit $exit_code, vLLM lines $vllm_lines, failed line in log.kvshrink-vllm '$failed_line'"
    return 1
}

step2() {
    local ok=1
    mkdir -p "$RESULT_DIR/step2"
    serve_and_check dsa_on 1 || ok=0
    serve_and_check dsa_off 0 || ok=0
    serve_calibration_fails || ok=0
    ((ok == 1))
}

# Only the calibration failure case of step 2 (STEPS=2c).
step2c() {
    mkdir -p "$RESULT_DIR/step2"
    serve_calibration_fails
}

# Only the DSA-on vLLM case of step 2 (STEPS=2a), e.g. with SERVE_MODEL / SERVE_TP_SIZE / SERVE_GPUS.
step2a() {
    mkdir -p "$RESULT_DIR/step2"
    serve_and_check dsa_on 1
}

trap stop_serve EXIT

mkdir -p "$RESULT_DIR"
cd "$REPO_DIR" || exit 1
log "start: steps=$STEPS result_dir=$RESULT_DIR"
if [[ "$DRY_RUN" != 1 ]]; then
    {
        echo "git: $(git rev-parse HEAD) branch $(git branch --show-current)"
        echo "image: $(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
        echo "so: $(ls -l --time-style=+%F_%T iaxl/torch_ext*.so)"
        git diff --stat
        md5sum iaxl/csrc/kv_xfer/{cuda,dsa}.cpp iaxl/csrc/include/env.h iaxl/csrc/common/envs.c \
            examples/kvshrink-vllm-serve.sh tools/calibration/dsa_calibration.py "$DEV_DIR"/*.py "$DEV_DIR"/*.sh
    } >"$RESULT_DIR/env.txt"
fi

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
