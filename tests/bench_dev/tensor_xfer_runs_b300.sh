#!/bin/bash
# tensor_xfer_benchmark.sh (all 4 methods, both directions) at 1, 4 and 8 ranks; results in bench-results/<run>/ranks<N>/.
# On the b300 host, in tmux: RUN_NAME=<name> RANKS_LIST="1 4 8" bash tests/bench_dev/tensor_xfer_runs_b300.sh
# Each ranks<N>/ gets SETTING_DONE and the run ALL_DONE once ranks 1, 4 and 8 are all done, for pull-b300-bench-results.sh.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
CONTAINER=iaxl.vllm-fddeef0-qiuyu
RUN_NAME=${RUN_NAME:-tensor_xfer_$(date +%Y%m%d_%H%M%S)}
RESULTS_ROOT=bench-results/$RUN_NAME
RANKS_LIST=${RANKS_LIST:-"1 4 8"}
RUN_TIMEOUT_S=3600

# Per-rank binding: GPU r of CUDA_VISIBLE_DEVICES, CPUs, QAT/IAA node and DSA WQ on that GPU's NUMA node.
rank_env() {
    case "$1" in
        1) echo "" ;;
        4) echo "CUDA_VISIBLE_DEVICES=0,2,4,6 VLLM_CPU_OMP_THREADS_BIND='0-23|24-47|48-71|72-95'" \
            "KVSHRINK_QAT_DEVICES='0|1|2|3' KVSHRINK_IAA_DEVICES='0|1|2|3'" \
            "KVSHRINK_DSA_DEVICES='wq0.0|wq2.0|wq4.0|wq6.0'" ;;
        # Two ranks per NUMA node: needs 2 DSA WQs per device (tools/setup_dsa_cnt.sh 2).
        8) echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
            "VLLM_CPU_OMP_THREADS_BIND='0-11|12-23|24-35|36-47|48-59|60-71|72-83|84-95'" \
            "KVSHRINK_QAT_DEVICES='0|0|1|1|2|2|3|3' KVSHRINK_IAA_DEVICES='0|0|1|1|2|2|3|3'" \
            "KVSHRINK_DSA_DEVICES='wq0.0|wq0.1|wq2.0|wq2.1|wq4.0|wq4.1|wq6.0|wq6.1'" ;;
        *) echo "Unknown rank count: $1" >&2; return 1 ;;
    esac
}

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/run.log") 2>&1
cp "${BASH_SOURCE[0]}" "$RESULTS_ROOT/"

for ranks in $RANKS_LIST; do
    run_dir=$RESULTS_ROOT/ranks$ranks
    mkdir -p "$run_dir"
    echo "### ranks $ranks ($(date +'%F %T')): $(rank_env "$ranks") -> $run_dir"
    docker exec -w "$REPO_DIR" "$CONTAINER" bash -lc \
        "$(rank_env "$ranks") timeout $RUN_TIMEOUT_S bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --ranks $ranks --direction both --methods cuda batch triton iaxl --output-dir $REPO_DIR/$run_dir" \
        > "$run_dir/benchmark.log" 2>&1
    status=$?
    echo "status=$status end=$(date +'%F %T')" | tee "$run_dir/SETTING_DONE"
done

if [ -f "$RESULTS_ROOT/ranks1/SETTING_DONE" ] && [ -f "$RESULTS_ROOT/ranks4/SETTING_DONE" ] \
    && [ -f "$RESULTS_ROOT/ranks8/SETTING_DONE" ]; then
    echo "end=$(date +'%F %T')" | tee "$RESULTS_ROOT/ALL_DONE"
fi
