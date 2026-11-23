#!/bin/bash
# Runs tools/calibration/dsa_calibration.py in the container after setvars.sh, like the serve script does.
# Usage: run_calibration.sh <output prefix>; writes <prefix>.setvars/.stdout/.stderr/.result.
# CALIBRATION_GPUS, when set, becomes CUDA_VISIBLE_DEVICES of the calibration only (setvars.sh keeps the real GPUs).
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
prefix=$1
source "$REPO_DIR/setvars.sh" >"$prefix.setvars" 2>&1 || exit 1
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
if [[ -n "${CALIBRATION_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$CALIBRATION_GPUS
fi
started_ns=$(date +%s%N)
timeout 300 python3 "$REPO_DIR/tools/calibration/dsa_calibration.py" >"$prefix.stdout" 2>"$prefix.stderr"
exit_code=$?
echo "exit_code $exit_code elapsed_ms $((($(date +%s%N) - started_ns) / 1000000))" >"$prefix.result"
