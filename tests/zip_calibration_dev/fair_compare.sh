#!/bin/bash
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
PERF=/usr/lib/linux-tools/6.8.0-134-generic/perf
IAA_DEVICES="1 3 5 7 9 11 13 15"
BENCH_ARGS="--shape 2 128 512 8 128 --num-layers 4 --data-source mock"
COMMON_ENV=(-e IAXL_ZIP_SRC_CAP=4194304 -e IAXL_ZIP_DST_CAP=4194304 -e IAXL_ZIP_SEGMENT_SIZE_KB=512)

qat_requests() {
    local dir
    for dir in $(sudo -n sh -c 'ls -d /sys/kernel/debug/qat_4xxx_*'); do
        echo "$(basename "$dir") $(sudo -n cat "$dir/fw_counters" | awk -F: '/Firmware Requests/ {s += $2} END {print s}')"
    done
}

run_case() {
    local name=$1 backend=$2
    shift 2
    local log="$RESULT_DIR/$name.log" events="" device
    for device in $IAA_DEVICES; do
        events="$events,iax$device/event_category=1,event=0x2/"
    done
    qat_requests >"$RESULT_DIR/$name.qat_before"
    sudo -n "$PERF" stat -a -x, -o "$RESULT_DIR/$name.iaa_perf.csv" -e "${events#,}" -- \
        docker exec -w "$REPO_DIR" "${COMMON_ENV[@]}" "$@" "$CONTAINER" \
        bash benchmark/kvstore/kvstore_benchmark.sh "--$backend" $BENCH_ARGS >"$log" 2>&1
    echo "exit $?" >"$RESULT_DIR/$name.exit"
    qat_requests >"$RESULT_DIR/$name.qat_after"
    {
        echo "== $name ($backend, $*) exit $(cut -d' ' -f2 "$RESULT_DIR/$name.exit")"
        grep -a -m1 -o 'qat_zip=.*omp_threads=[0-9]*' "$log"
        grep -a -E '^(Compression ratio|Compress throughput|Decompress throughput|Verification):' "$log"
        grep -a -E 'Bound rank 0: IAXL_QAT_DEVICES|\[qat_zip\] init done|\[iaa_zip\] (config|ready)' "$log" | sed 's/^.*\] //'
        echo "QAT firmware requests during the run (device: delta):"
        join "$RESULT_DIR/$name.qat_before" "$RESULT_DIR/$name.qat_after" | awk '{printf "  %s %d\n", $1, $3 - $2}'
        echo "IAA perf iaxN/event_category=1,event=0x2 during the run (device: count):"
        grep -v '^#' "$RESULT_DIR/$name.iaa_perf.csv" | awk -F, 'NF > 2 {split($3, p, "/"); printf "  %s %s\n", p[1], $1}'
    } >>"$RESULT_DIR/summary.txt"
}

mkdir -p "$RESULT_DIR"
cd "$REPO_DIR" || exit 1
run_case qat_4dev qat -e KVSHRINK_QAT_DEVICES=0,1,2,3 -e IAXL_QAT_DEVICES=0,1,2,3 -e IAXL_QAT_INSTANCE_NUM=16
run_case iaa_4dev iaa -e IAXL_IAA_DEVICES=0 -e IAXL_IAA_INSTANCE_NUM=16
run_case iaa_sweep_config iaa
run_case qat_sweep_config qat
echo "ALL DONE" >>"$RESULT_DIR/summary.txt"
