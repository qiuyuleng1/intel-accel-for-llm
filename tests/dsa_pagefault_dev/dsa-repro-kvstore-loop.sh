#!/bin/bash
#
# Reproduce "dsa batch failed" without vLLM. Runs the unchanged kvstore
# benchmark (4 ranks on GPUs 0-3, DSA only: no --qat/--iaa, so compression is
# off and QAT/IAA never start; 400 KB blocks, 16 layers, ScratchPool 16 GB and
# DDR pool 15 GB per rank, like 11-02) ITERATIONS times,
# each run a fresh set of processes, and counts the DSA failures of every run.
#
# DROP_CACHES chooses when to run `echo 3 > /proc/sys/vm/drop_caches` in a run:
#   none       never
#   setup      DROP_DELAY_S seconds after the run starts, i.e. while the ranks
#              still build their GPU data and no DSA copy has run yet (on 11-03
#              another job dropped caches ~5 s before the iter05 failures)
#   dsa_start  as soon as a rank logs its ScratchPool allocation, which happens
#              right before its first DSA copy
#
# COMPACT_MEMORY=1 runs `echo 1 > /proc/sys/vm/compact_memory` over and over
# from the first DSA copy until the run ends. Compaction moves movable pages
# (ordinary malloc memory) of every process to other physical pages.
#
# Every run also records how many pages the kernel migrated and swapped out
# while it ran (/proc/vmstat, whole machine), and is killed after RUN_TIMEOUT_S.
#
# PINNED_HOST_REGISTER=1 sets PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True,
# so torch pinned memory (ScratchPool) is private memory pinned with cudaHostRegister
# instead of a shared mapping.
#
# Runs on the HOST (needs docker exec; DROP_CACHES and COMPACT_MEMORY need
# passwordless sudo):
#   ITERATIONS=10 DROP_CACHES=setup bash tests/dsa_pagefault_dev/dsa-repro-kvstore-loop.sh
set -o pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

ITERATIONS=${ITERATIONS:-20}
DROP_CACHES=${DROP_CACHES:-none}
DROP_DELAY_S=${DROP_DELAY_S:-9}
COMPACT_MEMORY=${COMPACT_MEMORY:-0}
# A normal run takes ~30 s. A rank that aborts (e.g. QAT fails to start) leaves
# the benchmark's scheduler waiting forever, so give up after this long;
# `timeout` kills the whole process group, i.e. the scheduler and its ranks.
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-300}
# Stop when this many runs in a row fail (e.g. another job took GPUs 0-3).
MAX_FAILED_RUNS_IN_A_ROW=3
PINNED_HOST_REGISTER=${PINNED_HOST_REGISTER:-0}
CONTAINER=${CONTAINER:-iaxl.vllm-1103-dsa-pagefault-dev}
BENCH_ENV="export TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 IAXL_SCRATCH_POOL_SIZE_GB=16 IAXL_DDR_POOL_TOTAL_SIZE_GB=60"
if [[ "$PINNED_HOST_REGISTER" == 1 ]]; then
    BENCH_ENV="$BENCH_ENV PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True"
fi
BENCH_CMD="$BENCH_ENV; timeout -k 30 $RUN_TIMEOUT_S bash benchmark/kvstore/kvstore_benchmark.sh --ranks 4 --dsa --data-source mock --kv-cache-shape 2 9221 400 1 256 --num-layers 16"

case "$DROP_CACHES" in
    none | setup | dsa_start) ;;
    *)
        echo "DROP_CACHES must be none, setup or dsa_start, got '$DROP_CACHES'" >&2
        exit 1
        ;;
esac
if [[ "$COMPACT_MEMORY" != 0 && "$COMPACT_MEMORY" != 1 ]]; then
    echo "COMPACT_MEMORY must be 0 or 1, got '$COMPACT_MEMORY'" >&2
    exit 1
fi
if [[ "$DROP_CACHES" != none || "$COMPACT_MEMORY" == 1 ]] && ! sudo -n true; then
    echo "DROP_CACHES=$DROP_CACHES COMPACT_MEMORY=$COMPACT_MEMORY needs passwordless sudo" >&2
    exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-tests/dsa_pagefault_dev/results/dsa-repro-kvstore-dsaonly-drop_${DROP_CACHES}-compact${COMPACT_MEMORY}-$(date +'%Y%m%d_%H%M%S')}
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
echo "$BENCH_CMD" > "$RESULTS_DIR/command.txt"
git rev-parse HEAD > "$RESULTS_DIR/git_commit.txt"
md5sum "${BASH_SOURCE[0]}" benchmark/kvstore/*.py benchmark/kvstore/*.sh iaxl/csrc/kv_xfer/dsa.cpp \
    iaxl/csrc/dsa_gd/*.c > "$RESULTS_DIR/code_md5.txt"
echo "ITERATIONS=$ITERATIONS DROP_CACHES=$DROP_CACHES DROP_DELAY_S=$DROP_DELAY_S" \
    "COMPACT_MEMORY=$COMPACT_MEMORY RUN_TIMEOUT_S=$RUN_TIMEOUT_S PINNED_HOST_REGISTER=$PINNED_HOST_REGISTER" \
    | tee "$RESULTS_DIR/config.txt"

# Prints the time the drop started; the kernel logs it when the drop has finished, seconds later.
drop_caches() {
    date +%T.%3N
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

# Returns when the run logs its first ScratchPool allocation (its first DSA copy) or exits.
wait_for_first_dsa() {
    local log=$1
    local bench_pid=$2
    while kill -0 "$bench_pid" 2> /dev/null && ! grep -q "ScratchPool: " "$log"; do
        sleep 0.1
    done
}

# Machine-wide counter from /proc/vmstat, e.g. pgmigrate_success.
read_vmstat() {
    awk -v key="$1" '$1 == key {print $2}' /proc/vmstat
}

# Time of the first ScratchPool allocation (= first DSA copy) in a run log.
first_dsa_time() {
    grep -m1 "ScratchPool: " "$1" | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}"
}

# DSA failure lines carry no time; use the last timestamped line before the first one.
first_failure_time() {
    awk '
        match($0, /[0-9][0-9]:[0-9][0-9]:[0-9][0-9],[0-9][0-9][0-9]/) {
            last_time = substr($0, RSTART, RLENGTH)
        }
        /dsa batch failed|dsa op failed/ {
            print (last_time == "" ? "?" : last_time)
            exit
        }
    ' "$1"
}

# e.g. "0x5:4,0x3:1" (status of each failed batch, not of the descriptors inside it)
failure_statuses() {
    grep -oE "dsa batch failed, status=0x[0-9a-f]+" "$1" | cut -d= -f2 | sort | uniq -c | awk '{printf "%s%s:%s", (NR > 1 ? "," : ""), $2, $1}'
}

failed_runs_in_a_row=0
for iteration in $(seq -w 1 "$ITERATIONS"); do
    log="$RESULTS_DIR/iter$iteration.log"
    host_used_gib=$(free -g | awk '/^Mem:/ {print $3}')
    migrated_before=$(read_vmstat pgmigrate_success)
    swapped_out_before=$(read_vmstat pswpout)
    start_time=$(date +%T)
    docker exec -w "$REPO_ROOT" "$CONTAINER" bash -c "$BENCH_CMD" > "$log" 2>&1 &
    bench_pid=$!

    drop_time=-
    if [[ "$DROP_CACHES" == setup ]]; then
        sleep "$DROP_DELAY_S"
        drop_time=$(drop_caches)
    elif [[ "$DROP_CACHES" == dsa_start ]]; then
        wait_for_first_dsa "$log" "$bench_pid"
        drop_time=$(drop_caches)
    fi
    compactions=0
    if [[ "$COMPACT_MEMORY" == 1 ]]; then
        wait_for_first_dsa "$log" "$bench_pid"
        while kill -0 "$bench_pid" 2> /dev/null; do
            sudo -n sh -c 'echo 1 > /proc/sys/vm/compact_memory'
            compactions=$((compactions + 1))
        done
    fi

    wait "$bench_pid"
    exit_code=$?
    migrated_pages=$(($(read_vmstat pgmigrate_success) - migrated_before))
    swapped_out_pages=$(($(read_vmstat pswpout) - swapped_out_before))
    clean_log="$log.clean"
    tr -d '\r' < "$log" > "$clean_log"
    dsa_failed=$(grep -cE "dsa batch failed|dsa op failed" "$clean_log")
    statuses=-
    failure_time=-
    if ((dsa_failed > 0)); then
        statuses=$(failure_statuses "$clean_log")
        failure_time=$(first_failure_time "$clean_log")
    fi
    verification=$(grep -oE "Verification: [a-zA-Z]+" "$clean_log" | cut -d' ' -f2)
    rm -f "$clean_log"

    echo "iter=$iteration start=$start_time exit=$exit_code dsa_failed=$dsa_failed statuses=$statuses" \
        "first_dsa=$(first_dsa_time "$log") first_failure_near=$failure_time drop_caches_at=$drop_time" \
        "compactions=$compactions migrated_pages=$migrated_pages swapped_out_pages=$swapped_out_pages" \
        "verification=${verification:-none} host_used_before_gib=$host_used_gib" | tee -a "$SUMMARY"
    if ((exit_code == 0)); then
        failed_runs_in_a_row=0
    else
        failed_runs_in_a_row=$((failed_runs_in_a_row + 1))
        echo "run $iteration exited with $exit_code (124 = killed after ${RUN_TIMEOUT_S}s), see $log" | tee -a "$SUMMARY"
    fi
    if ((failed_runs_in_a_row >= MAX_FAILED_RUNS_IN_A_ROW)); then
        echo "$failed_runs_in_a_row runs in a row failed, stopping" | tee -a "$SUMMARY"
        break
    fi
done

# Every drop_caches the kernel logged today, including other jobs' drops.
sudo -n dmesg -T 2> /dev/null | grep "drop_caches" | grep "$(date '+%b %e')" > "$RESULTS_DIR/dmesg_drop_caches.txt"
sudo -n dmesg -T 2> /dev/null | grep "$(date '+%b %e')" | grep -iE "idxd|dmar|iommu|pasid|prq|page request|gdr" \
    > "$RESULTS_DIR/dmesg_dsa_iommu.txt"
runs_failed=$(grep -c "dsa_failed=[1-9]" "$SUMMARY")
runs_total=$(grep -c "^iter=" "$SUMMARY")
echo "runs with DSA failures: $runs_failed / $runs_total" | tee -a "$SUMMARY"
echo "results: $RESULTS_DIR"
