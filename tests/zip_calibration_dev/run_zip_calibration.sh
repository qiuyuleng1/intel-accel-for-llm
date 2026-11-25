#!/bin/bash
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
prefix=$1
source "$REPO_DIR/setvars.sh" >"$prefix.setvars" 2>&1 || exit 1
export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
env | grep -E '^IAXL_(QAT|IAA|CPU)_(ZIP_ENABLE|INSTANCE_NUM)=|^IAXL_OMP_THREAD_NUM=|^IAXL_CPU_ZIP_THREADS=' | sort >"$prefix.env"
if [[ -n "${CALIBRATION_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$CALIBRATION_GPUS
fi
started_ns=$(date +%s%N)
timeout 600 python3 "$REPO_DIR/tools/calibration/zip_calibration.py" >"$prefix.stdout" 2>"$prefix.stderr"
exit_code=$?
echo "exit_code $exit_code elapsed_ms $((($(date +%s%N) - started_ns) / 1000000))" >"$prefix.result"
