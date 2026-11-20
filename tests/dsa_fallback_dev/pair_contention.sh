#!/bin/bash
# Two calibrations at the same time on a GPU pair, each GPU with its own DSA WQ:
# GPU0+GPU1 share one PCIe switch uplink, GPU0+GPU2 do not. 20 calls per GPU so the two processes
# overlap even if one starts a bit later.
# Usage: RESULT_DIR=<dir> bash tests/dsa_fallback_dev/pair_contention.sh   (needs container iaxl.vllm)
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONTAINER=${CONTAINER:-iaxl.vllm}
RESULT_DIR=${RESULT_DIR:?set RESULT_DIR}
mkdir -p "$RESULT_DIR"

calibrate() {
    local gpu=$1 wq=$2 log_file=$3
    docker exec -w "$REPO_DIR" -e IAXL_DSA_GD_ENABLE=1 -e IAXL_DSA_WQS="$wq" "$CONTAINER" \
        timeout 120 python3 tests/dsa_fallback_dev/calibrate_once.py --gpu "$gpu" --calls 20 >"$log_file" 2>&1
}

for pair in "0:wq0.0 1:wq2.0" "0:wq0.0 2:wq4.0" "0:wq0.0"; do
    name="gpu$(echo "$pair" | sed 's/:wq[0-9.]*//g; s/ /_gpu/g')"
    for member in $pair; do
        calibrate "${member%%:*}" "${member##*:}" "$RESULT_DIR/${name}_gpu${member%%:*}.log" &
    done
    wait
    for member in $pair; do
        log_file="$RESULT_DIR/${name}_gpu${member%%:*}.log"
        echo "== $name GPU${member%%:*} (${member##*:}): H2D crossover counts over 20 calls"
        grep -a "^CALL" "$log_file" | awk '{print $3}' | sort | uniq -c
        echo "   H2D GB/s at the chosen block size (first 5 calls):"
        grep -a "dsa calibrate GPU.* H2D" "$log_file" | head -5 | sed 's/^/   /'
    done
done
