#!/bin/bash
# b300 host setup and checks before the Kimi-K3 DP launches, where the 2 GPUs of a NUMA node
# share its DSA / QAT / IAA device:
#   1. DSA: 2 dedicated WQs per device (tools/setup_dsa_cnt.sh 2), because a dedicated WQ opens
#      in one process only. Revert with: IMAGE=<iaxl image> bash tools/setup_dsa_cnt.sh 1
#   2. QAT: NumProcesses / NumberDcInstances of /etc/4xxx_dev*.conf (2 processes per device).
#   3. setvars.sh per-rank accelerator mapping for 8 workers, all accelerators on.
#   4. benchmark/kvstore/kvstore_benchmark.sh --ranks 8 --qat --iaa --dsa: 8 processes put and
#      get through the shared accelerators at once and verify the data (no vLLM).
# Runs on the b300 HOST (not in the container); output in tests/bench_dev/results/k3-accel-share-*/.
set -uo pipefail

DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$DEV_DIR/../.." && pwd)
CONTAINER=iaxl.vllm-kimi-k3-qiuyu
IMAGE=vllm-iaxl-dev-kimi-k3-qiuyu:1.6.0
OUT=$DEV_DIR/results/k3-accel-share-$(date +%Y%m%d_%H%M%S)
# results/ is created by the (root) container, so make the output dir there too.
docker exec "$CONTAINER" bash -c "mkdir -p '$OUT' && chmod 777 '$OUT'" || exit 1
exec > >(tee -a "$OUT/run.log") 2>&1

wq_state() {
    local wq
    for wq in /sys/bus/dsa/devices/wq*; do
        echo "$(basename "$wq") mode=$(<"$wq/mode") type=$(<"$wq/type") state=$(<"$wq/state") size=$(<"$wq/size")"
    done
}

echo "== 1. DSA WQs before"
wq_state
if ! IMAGE=$IMAGE bash "$REPO_DIR/tools/setup_dsa_cnt.sh" 2; then
    echo "ERROR: setup_dsa_cnt.sh 2 failed"
    exit 1
fi
echo "== 1. DSA WQs after"
wq_state

echo "== 2. QAT config"
for conf in /etc/4xxx_dev*.conf; do
    echo "$conf: $(sudo grep -E '^(ServicesEnabled|NumProcesses|NumberDcInstances)' "$conf" | tr '\n' ' ')"
done

echo "== 3. per-rank mapping, 8 workers"
docker exec -w "$REPO_DIR" "$CONTAINER" bash -c '
    export TP_SIZE=8 DP_SIZE=1 IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 IAXL_DSA_GD_ENABLE=1
    export IAXL_KV_COMPRESSION=1 IAXL_CPU_ZIP_ENABLE=0
    source setvars.sh > /dev/null || exit 1
    echo "KVSHRINK_QAT_DEVICES=$KVSHRINK_QAT_DEVICES"
    echo "KVSHRINK_DSA_DEVICES=$KVSHRINK_DSA_DEVICES"
    echo "KVSHRINK_IAA_DEVICES=$KVSHRINK_IAA_DEVICES"
    echo "VLLM_CPU_OMP_THREADS_BIND=$VLLM_CPU_OMP_THREADS_BIND"
' || { echo "ERROR: setvars.sh failed for 8 workers"; exit 1; }

echo "== 4. kvstore benchmark, 8 ranks, QAT + IAA + DSA"
docker exec -w "$REPO_DIR" "$CONTAINER" timeout 1800 \
    bash benchmark/kvstore/kvstore_benchmark.sh --ranks 8 --qat --iaa --dsa --data-source mock
echo "== kvstore benchmark exit $?"
echo "ALL_DONE $OUT"
