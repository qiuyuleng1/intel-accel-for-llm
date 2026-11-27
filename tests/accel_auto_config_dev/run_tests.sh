#!/bin/bash
# Tests of accel_auto_config (tools/auto_config.sh + tools/accel_auto_config.py) in container iaxl.vllm.
# Step 1: source setvars.sh only. Step 2: the benchmarks. Step 3: vLLM Qwen3.8-27B TP=4 on GPU 0-3.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/accel_auto_config_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:-$REPO_DIR/$DEV_DIR/results/tests_$(date +%Y%m%d_%H%M%S)}
STEPS=${STEPS:-"1 2 3"}
SERVE_READY_TIMEOUT_S=1200
PORT=${PORT:-8123}
GPU_BUSY_MIB=500
MODEL=/media/model-space/Qwen3.8-27B
SERVE_GPUS=0,1,2,3
SERVE_RUN_DIR=""
SERVE_EXEC_PID=""
TP4_ENV=(-e TP_SIZE=4 -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e IAXL_KV_COMPRESSION=1 -e IAXL_CPU_ZIP_ENABLE=0)

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

in_container() {
    docker exec -w "$REPO_DIR" "$@"
}

# Sources setvars.sh in the container with the given docker exec options; writes <name>.stdout/.stderr/.result.
probe() {
    local name=$1 prefix="$RESULT_DIR/step1/$1" started_ns
    shift
    log "setvars probe $name ($*)"
    started_ns=$(date +%s%N)
    in_container "$@" "$CONTAINER" timeout 600 bash "$DEV_DIR/setvars_probe.sh" >"$prefix.stdout" 2>"$prefix.stderr"
    echo "exit_code $? elapsed_ms $((($(date +%s%N) - started_ns) / 1000000))" >"$prefix.result"
}

probe_value() {
    grep -a "^PROBE $2=" "$RESULT_DIR/step1/$1.stdout" | cut -d= -f2-
}

stderr_count() {
    grep -a -c -- "$2" "$RESULT_DIR/step1/$1.stderr"
}

stderr_line() {
    grep -a -- "$2" "$RESULT_DIR/step1/$1.stderr" | head -1
}

# Both calibrations ran on <workers> ranks: values exported, QAT ranks 32..1024 KB, DSA used each rank's own WQ.
check_full_run() {
    local name=$1 workers=$2 h2d d2h segments sorted_kb qat_order expected_wqs used_wqs qat_inits iaa_inits
    h2d=$(probe_value "$name" IAXL_DSA_H2D_CROSSOVER_KB)
    d2h=$(probe_value "$name" IAXL_DSA_D2H_CROSSOVER_KB)
    segments=$(probe_value "$name" IAXL_ZIP_SEGMENT_SIZE_KB)
    sorted_kb=$(echo "$segments" | tr ',' '\n' | sort -n | tr '\n' ' ')
    qat_order=$(stderr_line "$name" "QAT segment sizes" | grep -o "[0-9]* KB [0-9.]*" | awk '{print $1}' | paste -sd,)
    expected_wqs=$(grep -a -o "KVSHRINK_DSA_DEVICES=[^ ]*" "$RESULT_DIR/step1/$name.stdout" | head -1 | cut -d= -f2 |
        tr '|' '\n' | cut -d, -f1 | sort | tr '\n' ' ')
    used_wqs=$(grep -a -o "dsa_init: mapped [0-9]* WQ(s) from \$IAXL_DSA_WQS (first [^)]*)" "$RESULT_DIR/step1/$name.stderr" |
        sed 's/.*(first \(.*\))/\1/' | sort | tr '\n' ' ')
    qat_inits=$(stderr_count "$name" "\[qat_zip\] init done")
    iaa_inits=$(stderr_count "$name" "\[iaa_zip\] ready")
    if [[ "$(probe_value "$name" setvars_status)" == 0 && "$h2d" =~ ^[0-9]+$ && "$d2h" =~ ^[0-9]+$ &&
        "$sorted_kb" == "32 64 128 256 512 1024 " && "$segments" == "$qat_order" && "$expected_wqs" == "$used_wqs" &&
        "$(stderr_count "$name" "\[accel_auto_config\] DSA crossover")" == 1 && "$qat_inits" == 1 && "$iaa_inits" == 0 ]] &&
        ((h2d % 128 == 0 && d2h % 128 == 0)); then
        summary "step1 $name: PASS setvars $(probe_value "$name" setvars_ms) ms, $(stderr_line "$name" "done in")," \
            "H2D $h2d KB, D2H $d2h KB, segments $segments, $workers workers, WQs $used_wqs"
        summary "    $(stderr_line "$name" "DSA crossover")"
        summary "    $(stderr_line "$name" "QAT segment sizes")"
        return 0
    fi
    summary "step1 $name: FAIL status $(probe_value "$name" setvars_status), H2D '$h2d' D2H '$d2h' segments '$segments'" \
        "(QAT line order '$qat_order'), WQs used '$used_wqs' expected '$expected_wqs', qat inits $qat_inits," \
        "iaa inits $iaa_inits, see $RESULT_DIR/step1/$name.*"
    return 1
}

step1() {
    local ok=1 run name uuid expected_uuid gpu exit_code elapsed_ms leftover
    mkdir -p "$RESULT_DIR/step1"
    require_gpus_free 0 1 2 3 || return 1

    probe 1a_no_in_container "${TP4_ENV[@]}" -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0
    if [[ "$(probe_value 1a_no_in_container setvars_status)" == 0 &&
        "$(stderr_count 1a_no_in_container "\[accel_auto_config\]")" == 0 &&
        "$(probe_value 1a_no_in_container IAXL_DSA_H2D_CROSSOVER_KB)" == "<unset>" &&
        "$(probe_value 1a_no_in_container IAXL_DSA_D2H_CROSSOVER_KB)" == "<unset>" &&
        "$(probe_value 1a_no_in_container IAXL_ZIP_SEGMENT_SIZE_KB)" == "<unset>" &&
        "$(probe_value 1a_no_in_container docker_run_args_has_in_container)" == yes ]] &&
        grep -a -q "^  IAXL_ZIP_SEGMENT_SIZE_KB=unset$" "$RESULT_DIR/step1/1a_no_in_container.stdout"; then
        summary "step1 1a_no_in_container: PASS no measurement, 3 variables unset (printed 'unset'), DOCKER_RUN_ARGS has -e IAXL_IN_CONTAINER=1"
    else
        summary "step1 1a_no_in_container: FAIL, see $RESULT_DIR/step1/1a_no_in_container.*"
        ok=0
    fi

    for run in 1 2 3; do
        probe "1b_tp4_run$run" -e IAXL_IN_CONTAINER=1 "${TP4_ENV[@]}" -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 \
            -e IAXL_IAA_ZIP_ENABLE=0
        check_full_run "1b_tp4_run$run" 4 || ok=0
    done

    if require_gpus_free 4 5 6 7; then
        probe 1d_tp8 -e IAXL_IN_CONTAINER=1 -e TP_SIZE=8 -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            -e IAXL_KV_COMPRESSION=1 -e IAXL_CPU_ZIP_ENABLE=0 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 \
            -e IAXL_IAA_ZIP_ENABLE=0
        check_full_run 1d_tp8 8 || ok=0
    else
        summary "step1 1d_tp8: SKIPPED, GPU 4-7 busy"
    fi

    probe 1e_dsa_off -e IAXL_IN_CONTAINER=1 "${TP4_ENV[@]}" -e IAXL_DSA_GD_ENABLE=0 -e IAXL_QAT_ZIP_ENABLE=1 \
        -e IAXL_IAA_ZIP_ENABLE=0
    if [[ "$(probe_value 1e_dsa_off setvars_status)" == 0 &&
        "$(stderr_count 1e_dsa_off "\[accel_auto_config\] DSA")" == 0 &&
        "$(stderr_count 1e_dsa_off "\[accel_auto_config\] QAT segment sizes")" == 1 &&
        "$(probe_value 1e_dsa_off IAXL_DSA_H2D_CROSSOVER_KB)" == "<unset>" &&
        "$(probe_value 1e_dsa_off IAXL_ZIP_SEGMENT_SIZE_KB)" =~ ^[0-9,]+$ ]]; then
        summary "step1 1e_dsa_off: PASS QAT only, segments $(probe_value 1e_dsa_off IAXL_ZIP_SEGMENT_SIZE_KB)," \
            "$(stderr_line 1e_dsa_off "done in")"
    else
        summary "step1 1e_dsa_off: FAIL, see $RESULT_DIR/step1/1e_dsa_off.*"
        ok=0
    fi

    probe 1f_iaa_only -e IAXL_IN_CONTAINER=1 "${TP4_ENV[@]}" -e IAXL_DSA_GD_ENABLE=0 -e IAXL_QAT_ZIP_ENABLE=0 \
        -e IAXL_IAA_ZIP_ENABLE=1
    if [[ "$(probe_value 1f_iaa_only setvars_status)" == 0 &&
        "$(probe_value 1f_iaa_only IAXL_ZIP_SEGMENT_SIZE_KB)" == 2048 &&
        "$(stderr_count 1f_iaa_only "\[accel_auto_config\] \(DSA\|QAT\)")" == 0 &&
        "$(stderr_line 1f_iaa_only "done in")" == "[accel_auto_config] done in 0.0 s" ]]; then
        summary "step1 1f_iaa_only: PASS IAXL_ZIP_SEGMENT_SIZE_KB=2048 without measuring, $(stderr_line 1f_iaa_only "done in")"
    else
        summary "step1 1f_iaa_only: FAIL, see $RESULT_DIR/step1/1f_iaa_only.*"
        ok=0
    fi

    probe 1g_preset -e IAXL_IN_CONTAINER=1 "${TP4_ENV[@]}" -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 \
        -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_DSA_H2D_CROSSOVER_KB=0 -e IAXL_DSA_D2H_CROSSOVER_KB=0 -e IAXL_ZIP_SEGMENT_SIZE_KB=512
    if [[ "$(probe_value 1g_preset setvars_status)" == 0 &&
        "$(stderr_count 1g_preset "\[accel_auto_config\] \(DSA\|QAT\)")" == 0 &&
        "$(probe_value 1g_preset IAXL_DSA_H2D_CROSSOVER_KB)" == 0 &&
        "$(probe_value 1g_preset IAXL_DSA_D2H_CROSSOVER_KB)" == 0 &&
        "$(probe_value 1g_preset IAXL_ZIP_SEGMENT_SIZE_KB)" == 512 ]]; then
        summary "step1 1g_preset: PASS preset 0, 0, 512 kept, nothing measured, $(stderr_line 1g_preset "done in")"
    else
        summary "step1 1g_preset: FAIL, see $RESULT_DIR/step1/1g_preset.*"
        ok=0
    fi

    probe 1h_rank3_fails -e IAXL_IN_CONTAINER=1 -e TP_SIZE=4 -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES=0,1,2,99 \
        -e IAXL_KV_COMPRESSION=1 -e IAXL_CPU_ZIP_ENABLE=0 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0
    read -r _ exit_code _ elapsed_ms <"$RESULT_DIR/step1/1h_rank3_fails.result"
    leftover=$(pgrep -f "tools/accel_auto_config.py" | tr '\n' ' ')
    if [[ "$(probe_value 1h_rank3_fails setvars_status)" == 1 && -z "$leftover" ]] &&
        grep -a -q "^\[accel_auto_config\] failed: dsa_rank3 exited with code" "$RESULT_DIR/step1/1h_rank3_fails.stderr" &&
        ((elapsed_ms < 60000)); then
        summary "step1 1h_rank3_fails: PASS setvars.sh returned 1 after ${elapsed_ms} ms," \
            "'$(stderr_line 1h_rank3_fails "failed:")', no leftover process"
    else
        summary "step1 1h_rank3_fails: FAIL status '$(probe_value 1h_rank3_fails setvars_status)', ${elapsed_ms} ms," \
            "leftover '$leftover', see $RESULT_DIR/step1/1h_rank3_fails.*"
        ok=0
    fi
    in_container -e IAXL_IN_CONTAINER=1 -e TP_SIZE=4 -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES=0,1,2,99 -e MODEL="$MODEL" \
        -e PORT="$PORT" -e IAXL_KV_COMPRESSION=1 -e IAXL_CPU_ZIP_ENABLE=0 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 \
        -e IAXL_IAA_ZIP_ENABLE=0 "$CONTAINER" timeout 300 bash examples/kvshrink-vllm-serve.sh \
        >"$RESULT_DIR/step1/1h_serve.log" 2>&1
    exit_code=$?
    if ((exit_code != 0)) && ! grep -a -q "vLLM API server\|api_server" "$RESULT_DIR/step1/1h_serve.log"; then
        summary "step1 1h_serve_stops: PASS 'bash examples/kvshrink-vllm-serve.sh' exit $exit_code before vllm serve"
    else
        summary "step1 1h_serve_stops: FAIL exit $exit_code, see $RESULT_DIR/step1/1h_serve.log"
        ok=0
    fi

    in_container "$CONTAINER" timeout 300 python3 "$DEV_DIR/fork_gpu_probe.py" 0 1 2 3 \
        >"$RESULT_DIR/step1/1i_fork_gpu.log" 2>&1
    local fork_ok=1
    for gpu in 0 1 2 3; do
        uuid=$(grep -a "^GPU $gpu " "$RESULT_DIR/step1/1i_fork_gpu.log" | awk '{print $3}')
        expected_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu" | sed 's/^GPU-//')
        [[ -n "$uuid" && "$uuid" == "$expected_uuid" ]] || fork_ok=0
    done
    if ((fork_ok == 1)) && [[ "$(grep -a -c "visible 1$" "$RESULT_DIR/step1/1i_fork_gpu.log")" == 4 ]]; then
        summary "step1 1i_fork_gpu: PASS 4 forked children each saw only its own GPU (UUIDs match nvidia-smi)"
    else
        summary "step1 1i_fork_gpu: FAIL, see $RESULT_DIR/step1/1i_fork_gpu.log"
        ok=0
    fi
    ((ok == 1))
}

step2() {
    local ok=1 exit_code log_file
    mkdir -p "$RESULT_DIR/step2"
    require_gpus_free 0 1 || return 1

    log_file="$RESULT_DIR/step2/2a_kvstore.log"
    log "kvstore_benchmark.sh --qat --dsa"
    in_container -e IAXL_IN_CONTAINER=1 "$CONTAINER" timeout 1200 bash benchmark/kvstore/kvstore_benchmark.sh --qat --dsa \
        --data-source mock >"$log_file" 2>&1
    exit_code=$?
    if ((exit_code == 0)) && grep -a -q "^Verification: passed" "$log_file" &&
        [[ "$(grep -a -c "^\[accel_auto_config\] DSA crossover" "$log_file")" == 1 &&
        "$(grep -a -c "^\[accel_auto_config\] QAT segment sizes" "$log_file")" == 1 ]]; then
        summary "step2 2a_kvstore: PASS both calibrations ran, then 'Verification: passed'"
        summary "    $(grep -a "^\[accel_auto_config\] DSA crossover" "$log_file")"
        summary "    $(grep -a "^\[accel_auto_config\] QAT segment sizes" "$log_file")"
        summary "    $(grep -a -E "^(Compression ratio|Compress throughput|Decompress throughput):" "$log_file" | paste -sd'|')"
    else
        summary "step2 2a_kvstore: FAIL exit $exit_code, see $log_file"
        ok=0
    fi

    log_file="$RESULT_DIR/step2/2b_sweep.log"
    log "kvstore_segment_sweep.sh --backends qat --sizes 512"
    in_container -e IAXL_IN_CONTAINER=1 "$CONTAINER" timeout 1800 bash benchmark/kvstore/kvstore_segment_sweep.sh \
        --backends qat --sizes 512 >"$log_file" 2>&1
    if grep -a -E "^\| 512 " "$log_file" | grep -v -q failed; then
        summary "step2 2b_sweep: PASS $(grep -a -E '^\| 512 ' "$log_file")"
    else
        summary "step2 2b_sweep: FAIL, see $log_file"
        ok=0
    fi

    log_file="$RESULT_DIR/step2/2c_tensor_xfer.log"
    log "tensor_xfer_benchmark.sh --ranks 2"
    in_container -e IAXL_IN_CONTAINER=1 "$CONTAINER" timeout 1200 bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh \
        --ranks 2 >"$log_file" 2>&1
    exit_code=$?
    if ((exit_code == 0)) && [[ "$(grep -a -c "^\[accel_auto_config\] DSA crossover" "$log_file")" == 0 &&
        "$(grep -a -c "^  IAXL_DSA_H2D_CROSSOVER_KB=0$" "$log_file")" == 1 ]]; then
        summary "step2 2c_tensor_xfer: PASS exit 0, no DSA calibration (crossovers preset to 0)," \
            "QAT calibration lines: $(grep -a -c "^\[accel_auto_config\] QAT segment sizes" "$log_file")"
    else
        summary "step2 2c_tensor_xfer: FAIL exit $exit_code, see $log_file"
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
    local name=$1 zip_case=$2 run_dir="$RESULT_DIR/step3/$1" waited_s=0
    shift 2
    mkdir -p "$run_dir"
    require_gpus_free ${SERVE_GPUS//,/ } || return 1
    log "starting serve $name ($*), log $run_dir/serve.log"
    docker exec -w "$REPO_DIR" -e IAXL_IN_CONTAINER=1 -e MODEL="$MODEL" -e TP_SIZE=4 -e DP_SIZE=1 \
        -e CUDA_VISIBLE_DEVICES="$SERVE_GPUS" -e PORT="$PORT" -e IAXL_DDR_POOL_SIZE_GB=16 -e IAXL_KV_COMPRESSION=1 \
        -e IAXL_DSA_GD_ENABLE=1 "$@" "$CONTAINER" \
        bash -c "echo \$\$ > $run_dir/serve.pid; exec bash examples/kvshrink-vllm-serve.sh" >"$run_dir/serve.log" 2>&1 &
    SERVE_EXEC_PID=$!
    SERVE_RUN_DIR=$run_dir
    while ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        if ! kill -0 "$SERVE_EXEC_PID" 2>/dev/null || ((waited_s >= SERVE_READY_TIMEOUT_S)); then
            stop_serve
            summary "step3 $name: FAIL vLLM not ready after ${waited_s}s, see $run_dir/serve.log"
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
        summary "step3 $name: FAIL requests failed, see $run_dir/requests.log"
        return 1
    fi
    if python3 "$DEV_DIR/e2e_check.py" "$run_dir" "$zip_case" 4 >"$run_dir/check.txt" 2>&1; then
        summary "step3 $name: PASS (ready after ${waited_s}s), see $run_dir/check.txt"
    else
        summary "step3 $name: FAIL, see $run_dir/check.txt"
    fi
    summary "$(grep -E '^(PASS|FAIL) |^accel_auto_config done|^DSA summaries|^QAT ranked|^vLLM \[kv_zip\]' "$run_dir/check.txt" | sed 's|^|    |')"
    grep -q "^FAIL " "$run_dir/check.txt" && return 1
    return 0
}

step3() {
    local ok=1
    mkdir -p "$RESULT_DIR/step3"
    serve_case 3a_qat qat -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=0 || ok=0
    serve_case 3b_iaa_only iaa -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=1 -e IAXL_CPU_ZIP_ENABLE=0 || ok=0
    serve_case 3c_cpu_zip none -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=1 || ok=0
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
        benchmark/kvstore/kvstore_benchmark.sh benchmark/tensor_xfer/tensor_xfer_benchmark.sh "$DEV_DIR"/*.py "$DEV_DIR"/*.sh
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
