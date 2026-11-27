#!/bin/bash
# Per-step timing of accel_auto_config at TP=4 and TP=8, using the test copy with timestamps
# (accel_auto_config_timed.py); tools/accel_auto_config.py itself is not changed.
set -uo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEV_DIR=tests/accel_auto_config_dev
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:-$REPO_DIR/$DEV_DIR/results/timing_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULT_DIR"
cd "$REPO_DIR" || exit 1
md5sum tools/accel_auto_config.py "$DEV_DIR/accel_auto_config_timed.py" >"$RESULT_DIR/env.txt"

for tp in 4 8; do
    gpus=$(seq -s, 0 $((tp - 1)))
    for gpu in ${gpus//,/ }; do
        used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        if ((used_mib > 500)); then
            echo "TP=$tp skipped: GPU$gpu busy (${used_mib} MiB used)" | tee -a "$RESULT_DIR/summary.txt"
            continue 2
        fi
    done
    docker exec -w "$REPO_DIR" -e TP_SIZE="$tp" -e DP_SIZE=1 -e CUDA_VISIBLE_DEVICES="$gpus" -e IAXL_KV_COMPRESSION=1 \
        -e IAXL_CPU_ZIP_ENABLE=0 -e IAXL_DSA_GD_ENABLE=1 -e IAXL_QAT_ZIP_ENABLE=1 -e IAXL_IAA_ZIP_ENABLE=0 "$CONTAINER" \
        bash -c 'source setvars.sh >/dev/null || exit 1
            export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
            started_ns=$(date +%s%N)
            timeout 300 python3 tests/accel_auto_config_dev/accel_auto_config_timed.py
            exit_code=$?
            echo "[timing-wall] python exit $exit_code wall_ms $((($(date +%s%N) - started_ns) / 1000000))" >&2' \
        >"$RESULT_DIR/tp$tp.stdout" 2>"$RESULT_DIR/tp$tp.stderr"
    {
        echo "== TP=$tp"
        grep -a "^\[timing-wall\]\|^\[accel_auto_config\]" "$RESULT_DIR/tp$tp.stderr"
        python3 "$DEV_DIR/timing_report.py" "$RESULT_DIR/tp$tp.stderr"
    } | tee -a "$RESULT_DIR/summary.txt"
done
echo "ALL DONE. Summary: $RESULT_DIR/summary.txt"
