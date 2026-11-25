#!/bin/bash
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/zip_calibration_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:-$REPO_DIR/$DEV_DIR/results/tests_$(date +%Y%m%d_%H%M%S)}
STEPS=${STEPS:-"build 1 2 3"}
SERVE_READY_TIMEOUT_S=1200
PORT=${PORT:-8123}
GPU_BUSY_MIB=500
MODEL=/media/model-space/Qwen3.8-27B
SERVE_GPUS=0,1,2,3
SERVE_RUN_DIR=""
SERVE_EXEC_PID=""

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

stepbuild() {
    log "building in $CONTAINER"
    in_container "$CONTAINER" bash -c 'pip install -e . --no-build-isolation' >"$RESULT_DIR/build.log" 2>&1
    local exit_code=$?
    if ((exit_code == 0)); then
        summary "build: PASS ($(ls -l --time-style=+%F_%T iaxl/torch_ext*.so | awk '{print $6}'))"
        return 0
    fi
    summary "build: FAIL exit $exit_code, see $RESULT_DIR/build.log"
    return 1
}

segment_case() {
    local name=$1 setting=$2 tensor_kb=$3 want_segment_kb=$4 want_segments=$5 log_file="$RESULT_DIR/step1/$1.log"
    local env_args=(-e TP_SIZE=1)
    [[ "$setting" != unset ]] && env_args+=(-e "IAXL_ZIP_SEGMENT_SIZE_KB=$setting")
    in_container "${env_args[@]}" "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh timeout 300 \
        python3 "$DEV_DIR/segment_check.py" tools/calibration roundtrip "$tensor_kb" >"$log_file" 2>&1
    local line block segment segments
    line=$(grep -a -o "\[kv_zip\] block_size=[0-9]* B segment_size=[0-9]* B segments=[0-9]*" "$log_file" | head -1)
    read -r block segment segments <<<"$(echo "$line" | sed -E 's/.*block_size=([0-9]+) B segment_size=([0-9]+) B segments=([0-9]+)/\1 \2 \3/')"
    if [[ "$segment" == "$((want_segment_kb * 1024))" && "$segments" == "$want_segments" ]] &&
        grep -a -q "^ROUNDTRIP PASS" "$log_file"; then
        summary "step1 $name: PASS IAXL_ZIP_SEGMENT_SIZE_KB=$setting tensor ${tensor_kb} KB -> $line, roundtrip PASS"
        return 0
    fi
    summary "step1 $name: FAIL IAXL_ZIP_SEGMENT_SIZE_KB=$setting tensor ${tensor_kb} KB -> '$line'" \
        "(want segment ${want_segment_kb} KB, $want_segments segments), $(grep -a -o '^ROUNDTRIP.*' "$log_file"), see $log_file"
    return 1
}

invalid_case() {
    local name=$1 setting=$2 log_file="$RESULT_DIR/step1/$1.log"
    in_container -e "IAXL_ZIP_SEGMENT_SIZE_KB=$setting" "$CONTAINER" \
        timeout 120 python3 -c "from iaxl import torch_ext" >"$log_file" 2>&1
    local exit_code=$?
    if ((exit_code != 0)) && grep -a -q "IAXL_ZIP_SEGMENT_SIZE_KB=.* must be up to 16 comma-separated" "$log_file"; then
        summary "step1 $name: PASS '$setting' rejected (exit $exit_code): $(grep -a -o '\[iaxl\] ERROR.*' "$log_file")"
        return 0
    fi
    summary "step1 $name: FAIL '$setting' exit $exit_code, see $log_file"
    return 1
}

step1() {
    local ok=1 kb
    mkdir -p "$RESULT_DIR/step1"
    require_gpus_free 0 || return 1
    for kb in 64 784 2048; do
        segment_case "1a_unset_${kb}k" unset "$kb" "$kb" 1 || ok=0
    done
    segment_case 1b_zero_784k 0 784 784 1 || ok=0
    segment_case 1c_128_784k 128 784 128 6 || ok=0
    segment_case 1d_256_32_64k 256,32 64 32 2 || ok=0
    segment_case 1e_256_32_784k 256,32 784 256 3 || ok=0
    segment_case 1f_2048_512_2048k 2048,512 2048 2048 1 || ok=0
    invalid_case 1g_letters abc || ok=0
    invalid_case 1g_negative -1 || ok=0
    invalid_case 1g_empty_item 1,,2 || ok=0
    invalid_case 1g_17_items 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 || ok=0
    in_container -e TP_SIZE=1 -e IAXL_ZIP_SEGMENT_SIZE_KB=256 "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh \
        timeout 300 python3 "$DEV_DIR/segment_check.py" tools/calibration save 784 "$RESULT_DIR/step1/1h_blocks.pt" \
        >"$RESULT_DIR/step1/1h_save.log" 2>&1
    in_container -e TP_SIZE=1 -e IAXL_ZIP_SEGMENT_SIZE_KB=32 "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh \
        timeout 300 python3 "$DEV_DIR/segment_check.py" tools/calibration load 784 "$RESULT_DIR/step1/1h_blocks.pt" \
        >"$RESULT_DIR/step1/1h_load.log" 2>&1
    if grep -a -q "segment_size=262144 B" "$RESULT_DIR/step1/1h_save.log" && grep -a -q "^LOAD_DECOMPRESS PASS" "$RESULT_DIR/step1/1h_load.log"; then
        summary "step1 1h_compress_256_decompress_32: PASS (compressed with 256 KB segments, decompressed in a process set to 32)"
    else
        summary "step1 1h_compress_256_decompress_32: FAIL, see $RESULT_DIR/step1/1h_*.log"
        ok=0
    fi
    rm -f "$RESULT_DIR/step1/1h_blocks.pt"
    in_container -e TP_SIZE=1 "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh timeout 600 \
        python3 tests/iaxl_unittest.py -v >"$RESULT_DIR/step1/iaxl_unittest.log" 2>&1
    if grep -a -q "^OK" "$RESULT_DIR/step1/iaxl_unittest.log"; then
        summary "step1 iaxl_unittest: PASS ($(grep -a -o '^Ran .*' "$RESULT_DIR/step1/iaxl_unittest.log"))"
    else
        summary "step1 iaxl_unittest: FAIL, see $RESULT_DIR/step1/iaxl_unittest.log"
        ok=0
    fi
    ((ok == 1))
}

calibrate() {
    local name=$1
    shift
    log "zip calibration $name ($*)"
    in_container -e IAXL_KV_COMPRESSION=1 "$@" "$CONTAINER" bash "$DEV_DIR/run_zip_calibration.sh" "$RESULT_DIR/step2/$name"
}

check_calibration() {
    local name=$1 workers=$2 backend=$3 prefix="$RESULT_DIR/step2/$1" exit_code elapsed_ms ranked sorted_kb
    local summary_line config_lines want_threads init_lines other_init_lines threads_ok=1 line
    read -r _ exit_code _ elapsed_ms <"$prefix.result"
    ranked=$(cat "$prefix.stdout")
    sorted_kb=$(echo "$ranked" | tr ',' '\n' | sort -n | tr '\n' ' ')
    summary_line=$(grep -a "^\[zip_calibration\] $backend segment sizes" "$prefix.stderr")
    config_lines=$(grep -a -c "^\[iaxl\] config:" "$prefix.stderr")
    if [[ "$backend" == QAT ]]; then
        want_threads=$(sed -n 's/^IAXL_QAT_INSTANCE_NUM=//p' "$prefix.env")
        init_lines=$(grep -a -c "\[qat_zip\] init done" "$prefix.stderr")
        other_init_lines=$(grep -a -c "\[iaa_zip\] ready" "$prefix.stderr")
    else
        want_threads=$(sed -n 's/^IAXL_IAA_INSTANCE_NUM=//p' "$prefix.env")
        init_lines=$(grep -a -c "\[iaa_zip\] ready" "$prefix.stderr")
        other_init_lines=$(grep -a -c "\[qat_zip\] init done" "$prefix.stderr")
    fi
    while read -r line; do
        [[ "$line" == *"omp_threads=$want_threads "* ]] || threads_ok=0
        [[ "$line" == *"cpu_zip=OFF"* ]] || threads_ok=0
    done < <(grep -a "^\[iaxl\] config:" "$prefix.stderr")
    if ((exit_code == 0 && config_lines == workers && init_lines == workers && other_init_lines == 0 && threads_ok == 1)) &&
        [[ "$(wc -l <"$prefix.stdout")" == 1 && "$sorted_kb" == "16 32 64 128 256 512 1024 2048 " && -n "$summary_line" ]]; then
        summary "step2 $name: PASS ${elapsed_ms} ms, stdout '$ranked', $summary_line, $workers workers with only $backend, omp_threads=$want_threads"
        return 0
    fi
    summary "step2 $name: FAIL exit $exit_code, stdout '$ranked', summary '$summary_line', config lines $config_lines," \
        "$backend init $init_lines, other init $other_init_lines, omp_threads=$want_threads ok=$threads_ok, see $prefix.*"
    return 1
}

step2() {
    local ok=1 exit_code elapsed_ms failed_line leftover gpu_ratio cpu_line cpu_ratio
    mkdir -p "$RESULT_DIR/step2"
    require_gpus_free 0 1 2 3 || return 1
    calibrate 2a_qat_tp4 -e TP_SIZE=4 -e DP_SIZE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0
    check_calibration 2a_qat_tp4 4 QAT || ok=0
    calibrate 2b_qat_iaa_tp4 -e TP_SIZE=4 -e DP_SIZE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=1
    check_calibration 2b_qat_iaa_tp4 4 QAT || ok=0
    calibrate 2c_iaa_tp4 -e TP_SIZE=4 -e DP_SIZE=1 -e IAXL_QAT_ZIP_ENABLE=0 -e IAXL_IAA_ZIP_ENABLE=1
    check_calibration 2c_iaa_tp4 4 IAA || ok=0
    calibrate 2d_qat_tp2_dp2 -e TP_SIZE=2 -e DP_SIZE=2 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0
    check_calibration 2d_qat_tp2_dp2 4 QAT || ok=0
    calibrate 2e_rank3_fails -e TP_SIZE=4 -e DP_SIZE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0 -e CALIBRATION_GPUS=0,1,2,99
    read -r _ exit_code _ elapsed_ms <"$RESULT_DIR/step2/2e_rank3_fails.result"
    failed_line=$(grep -a "^\[zip_calibration\] failed: rank3" "$RESULT_DIR/step2/2e_rank3_fails.stderr")
    leftover=$(pgrep -f "tools/calibration/zip_calibration.py" | tr '\n' ' ')
    if ((exit_code != 0 && elapsed_ms < 60000)) && [[ -n "$failed_line" && -z "$leftover" ]]; then
        summary "step2 2e_rank3_fails: PASS exit $exit_code after ${elapsed_ms} ms, '$failed_line', no leftover process"
    else
        summary "step2 2e_rank3_fails: FAIL exit $exit_code after ${elapsed_ms} ms, '$failed_line', leftover '$leftover'"
        ok=0
    fi
    in_container -e TP_SIZE=1 "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh timeout 600 \
        python3 "$DEV_DIR/segment_check.py" tools/calibration mock_ratio >"$RESULT_DIR/step2/2f_gpu.log" 2>&1
    in_container -e TP_SIZE=1 -e CUDA_VISIBLE_DEVICES= "$CONTAINER" bash tests/dsa_fallback_dev/with_setvars.sh timeout 900 \
        python3 "$DEV_DIR/segment_check.py" tools/calibration mock_ratio >"$RESULT_DIR/step2/2f_cpu.log" 2>&1
    gpu_ratio=$(grep -a -o "^MOCK device cuda ratio [0-9.]*" "$RESULT_DIR/step2/2f_gpu.log" | awk '{print $5}')
    cpu_line=$(grep -a -o "^MOCK device cpu ratio [0-9.]*" "$RESULT_DIR/step2/2f_cpu.log")
    cpu_ratio=$(echo "$cpu_line" | awk '{print $5}')
    if [[ -n "$gpu_ratio" && -n "$cpu_ratio" ]] && awk -v g="$gpu_ratio" -v c="$cpu_ratio" 'BEGIN {d = (c - g) / g; exit !(d < 0.02 && d > -0.02)}'; then
        summary "step2 2f_mock_on_cpu: PASS generated on cpu, ratio cpu $cpu_ratio vs cuda $gpu_ratio"
    else
        summary "step2 2f_mock_on_cpu: FAIL cpu '$cpu_line' cuda ratio '$gpu_ratio', see $RESULT_DIR/step2/2f_*.log"
        ok=0
    fi
    in_container "$CONTAINER" bash benchmark/kvstore/kvstore_segment_sweep.sh --backends qat --sizes 512 \
        >"$RESULT_DIR/step2/2g_sweep.log" 2>&1
    if grep -a -E "^\| 512 " "$RESULT_DIR/step2/2g_sweep.log" | grep -v -q failed; then
        summary "step2 2g_segment_sweep: PASS $(grep -a -E '^\| 512 ' "$RESULT_DIR/step2/2g_sweep.log")"
    else
        summary "step2 2g_segment_sweep: FAIL, see $RESULT_DIR/step2/2g_sweep.log"
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
    local name=$1 backend=$2 run_dir="$RESULT_DIR/step3/$1" waited_s=0
    shift 2
    mkdir -p "$run_dir"
    require_gpus_free ${SERVE_GPUS//,/ } || return 1
    log "starting serve $name ($*), log $run_dir/serve.log"
    docker exec -w "$REPO_DIR" -e MODEL="$MODEL" -e TP_SIZE=4 -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES="$SERVE_GPUS" \
        -e PORT="$PORT" -e IAXL_DDR_POOL_SIZE_GB=16 -e IAXL_KV_COMPRESSION=1 -e IAXL_DSA_GD_ENABLE=1 "$@" "$CONTAINER" \
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
    if python3 "$DEV_DIR/e2e_zip_check.py" "$run_dir" "$backend" >"$run_dir/check.txt" 2>&1; then
        summary "step3 $name: PASS (ready after ${waited_s}s), see $run_dir/check.txt"
    else
        summary "step3 $name: FAIL, see $run_dir/check.txt"
    fi
    summary "$(grep -E '^(PASS|FAIL) |^zip calibration|^vLLM \[kv_zip\]' "$run_dir/check.txt" | sed 's|^|    |')"
    grep -q "^FAIL " "$run_dir/check.txt" && return 1
    return 0
}

step3() {
    local ok=1
    mkdir -p "$RESULT_DIR/step3"
    serve_case 3a_qat QAT -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0 -e IAXL_CPU_ZIP_ENABLE=0 || ok=0
    serve_case 3b_qat_iaa QAT -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=1 -e IAXL_CPU_ZIP_ENABLE=0 || ok=0
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
    md5sum iaxl/csrc/include/env.h iaxl/csrc/common/envs.c iaxl/csrc/kv_zip/kv_zip.cpp examples/kvshrink-vllm-serve.sh \
        setvars.sh tools/calibration/*.py "$DEV_DIR"/*.py "$DEV_DIR"/*.sh
} >"$RESULT_DIR/env.txt"

for step in $STEPS; do
    step_started=$(date +%s)
    log "step $step started"
    if "step$step"; then
        log "step $step PASS ($(($(date +%s) - step_started))s)"
    else
        log "step $step FAIL ($(($(date +%s) - step_started))s), see $RESULT_DIR/summary.txt"
        [[ "$step" == build ]] && exit 1
    fi
done
log "ALL DONE. Summary: $RESULT_DIR/summary.txt"
