#!/bin/bash
# Run on b300 (agreed 11-17), on the HOST, one step after another. Both models get a
# 64 GB ScratchPool per rank (8 GB in every earlier run), 8 host OpenMP threads per
# rank and 4 QAT instances per QAT device (setvars.sh in both repos); DDR pools as
# before (Kimi-K3 600 GB, Qwen3.8-27B 1000 GB).
#   1. DSA to 2 WQs per device: Kimi-K3 TP8 has 8 ranks for the 4 DSA devices
#      (../intel-accel-for-llm-kimi-k3/tools/setup_dsa_cnt.sh 2, as on 11-13).
#   2. Kimi-K3 TP8 (../intel-accel-for-llm-kimi-k3/test-kimi-k3.sh, Python
#      frontend): kvshrink-nozip (DSA only), then kvshrink (QAT 4 + IAA 4 + DSA).
#      8 threads is the most a TP8 rank's 12 CPUs allow (setvars.sh keeps 4 for
#      vLLM). KEEP_LAST_SERVE=0 stops its last vLLM.
#   3. Wait until nothing serves on port 8000 and every GPU shows 0 MiB, then DSA
#      back to 1 WQ per device, the state of every earlier Qwen3.8-27B run
#      (setup_dsa_cnt.sh 1, as on 11-16).
#   4. Qwen3.8-27B TP4 with the Rust frontend (test-qwen38-27B-rust.sh):
#      kvshrink-zip-dsa-omp8, kvshrink-nozip-dsa-omp8, kvshrink-nozip-nodsa-omp8.
# A failed Kimi-K3 run does not stop the Qwen run. A failed DSA step, or GPUs
# still in use after GPU_FREE_TIMEOUT_S, stops everything: the next run would
# measure on a machine in an unknown state.
#
# Both run directory names are fixed at the start (the Qwen one carries the start
# time of this script, not of the Qwen run), so both pullers on the 6000d host can
# be started right away; this script prints their commands. Each test script
# writes its markers after every vLLM restart and the pullers copy the results
# back then.
#
# Log: tests/bench_dev/logs/night-<time>.log
# Print the plan without changing DSA or starting vLLM: DRY_RUN=1 bash night-kimi-then-qwen.sh
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_ROOT"

KIMI_REPO=$(cd -- "$REPO_ROOT/../intel-accel-for-llm-kimi-k3" && pwd)
# The image setup_dsa_cnt.sh ran with on 11-13 and 11-16 (it has accel-config).
DSA_IMAGE=vllm-iaxl-dev-kimi-k3-qiuyu:1.6.0
# DSA devices, one per NUMA node; the odd device numbers are IAA.
DSA_DEVICE_IDS="0 2 4 6"
DRY_RUN=${DRY_RUN:-0}
PORT=8000
GPU_FREE_TIMEOUT_S=300
START_TIME=$(date +%Y%m%d_%H%M%S)
KIMI_RUN=Kimi-K3_tp8_omp8_scratch64g_$START_TIME
KIMI_CONFIGS="kvshrink-nozip kvshrink"
QWEN_RUN=Qwen3.8-27B_tp4_rust_omp8_scratch64g_$START_TIME
QWEN_SETTINGS="kvshrink-zip-dsa-omp8 kvshrink-nozip-dsa-omp8 kvshrink-nozip-nodsa-omp8"
SCRATCH_POOL_SIZE_GB=64

mkdir -p tests/bench_dev/logs
exec > >(tee -a "tests/bench_dev/logs/night-$START_TIME.log") 2>&1

# State of one DSA work queue: "enabled", "disabled", or "missing" when it does not exist.
wq_state() {
    local wq=$1
    if [ -f "/sys/bus/dsa/devices/$wq/state" ]; then
        cat "/sys/bus/dsa/devices/$wq/state"
    else
        echo "missing"
    fi
}

# Configures every DSA device with wq_count WQs and checks the result: WQ 0 of every
# device enabled, WQ 1 enabled only with 2 WQs per device.
set_dsa_wq_count() {
    local wq_count=$1
    local expected_second_wq_state=disabled
    if [ "$wq_count" = "2" ]; then
        expected_second_wq_state=enabled
    fi
    echo ""
    echo "### DSA: $wq_count WQ per device ($(date +'%F %T'))"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] (cd $KIMI_REPO && IMAGE=$DSA_IMAGE bash tools/setup_dsa_cnt.sh $wq_count)"
        return 0
    fi
    if ! (cd "$KIMI_REPO" && IMAGE=$DSA_IMAGE bash tools/setup_dsa_cnt.sh "$wq_count"); then
        echo "ERROR: tools/setup_dsa_cnt.sh $wq_count failed" >&2
        return 1
    fi
    local device_id first_wq_state second_wq_state
    for device_id in $DSA_DEVICE_IDS; do
        first_wq_state=$(wq_state "wq$device_id.0")
        second_wq_state=$(wq_state "wq$device_id.1")
        if [ "$first_wq_state" != "enabled" ] || [ "$second_wq_state" != "$expected_second_wq_state" ]; then
            echo "ERROR: wq$device_id.0 is $first_wq_state, wq$device_id.1 is $second_wq_state" \
                "(expected enabled, $expected_second_wq_state)" >&2
            return 1
        fi
    done
    echo "    checked: wq0.0 wq2.0 wq4.0 wq6.0 enabled, wq0.1 wq2.1 wq4.1 wq6.1 $expected_second_wq_state"
}

# Waits until nothing serves on PORT and every GPU shows 0 MiB used.
wait_for_free_gpus() {
    local waited=0 used_mib=""
    while (( waited <= GPU_FREE_TIMEOUT_S )); do
        used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
        if [ "$used_mib" = "0" ] && ! curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "    GPUs free (0 MiB in total), nothing serves on port $PORT"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "ERROR: after ${GPU_FREE_TIMEOUT_S}s the GPUs still use $used_mib MiB in total or port $PORT still serves" >&2
    return 1
}

echo "==========================================================="
echo "Run $START_TIME: Kimi-K3 TP8, then Qwen3.8-27B TP4 (ScratchPool ${SCRATCH_POOL_SIZE_GB} GB per rank)"
echo "  Kimi-K3:     $KIMI_REPO/bench-results/$KIMI_RUN  configs=[$KIMI_CONFIGS]"
echo "  Qwen3.8-27B: $REPO_ROOT/bench-results/$QWEN_RUN  settings=[$QWEN_SETTINGS]"
echo "  pullers to start on the 6000d host, each in its own tmux window:"
echo "    cd /home/pese/qiuyu/B300/intel-accel-for-llm-kimi-k3 && bash tests/bench_dev/pull-b300-kimi-results.sh $KIMI_RUN"
echo "    cd /home/pese/qiuyu/B300/intel-accel-for-llm-b300 && bash tests/bench_dev/pull-b300-bench-results.sh $QWEN_RUN"
echo "==========================================================="

if [ "$DRY_RUN" != "1" ]; then
    echo "Before starting:"
    if ! wait_for_free_gpus; then
        exit 1
    fi
fi

if ! set_dsa_wq_count 2; then
    exit 1
fi

echo ""
echo "### Kimi-K3 ($(date +'%F %T'))"
(cd "$KIMI_REPO" && RUN_NAME=$KIMI_RUN LAYOUTS=tp8 CONFIGS="$KIMI_CONFIGS" SCRATCH_POOL_SIZE_GB=$SCRATCH_POOL_SIZE_GB \
    KEEP_LAST_SERVE=0 DRY_RUN=$DRY_RUN bash test-kimi-k3.sh)
kimi_status=$?
echo "### Kimi-K3 exited with status $kimi_status ($(date +'%F %T'))"

if [ "$DRY_RUN" != "1" ]; then
    echo "After Kimi-K3:"
    if ! wait_for_free_gpus; then
        exit 1
    fi
fi

if ! set_dsa_wq_count 1; then
    exit 1
fi

echo ""
echo "### Qwen3.8-27B ($(date +'%F %T'))"
RUN_NAME=$QWEN_RUN SCRATCH_POOL_SIZE_GB=$SCRATCH_POOL_SIZE_GB SETTINGS="$QWEN_SETTINGS" DRY_RUN=$DRY_RUN \
    bash test-qwen38-27B-rust.sh
qwen_status=$?

echo ""
echo "Done ($(date +'%F %T')): Kimi-K3 status $kimi_status, Qwen3.8-27B status $qwen_status"
