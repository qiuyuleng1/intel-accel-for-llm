#!/bin/bash
# Copy each finished setting of one test-qwen38-27B-rust.sh run from b300 into
# this repo's bench-results/, once, right after the setting finished; copy the
# run's top-level files (script copy, run log, ALL_DONE) after the last one.
#
# b300 cannot reach the 6000d host, so this host asks: every CHECK_INTERVAL_S
# seconds (default 60) one ssh lists the run's <setting>/SETTING_DONE markers
# and ALL_DONE (no data moved). A setting that has a marker is complete
# (test-qwen38-27B-rust.sh writes it after the setting's vLLM stopped) and is
# copied while the next setting is still starting vLLM, before anything is
# measured; rsync on b300 runs at the lowest CPU and IO priority. After a
# successful copy this script writes <setting>/SYNCED here and never copies that
# setting again. Exits once ALL_DONE, the top-level files and every setting are
# copied.
#
# Usage, in a tmux session on the 6000d host, with the run directory name that
# test-qwen38-27B-rust.sh prints ("results -> bench-results/<name>"):
#   bash tests/bench_dev/pull-b300-bench-results.sh <run dir name>
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE_HOST=b300
# Overridable to test against a dry run under tests/bench_dev/results.
REMOTE_RESULTS=${REMOTE_RESULTS:-/home/intel/qiuyu/intel-accel-for-llm-b300/bench-results}
LOCAL_RESULTS=${LOCAL_RESULTS:-$REPO_ROOT/bench-results}
CHECK_INTERVAL_S=${CHECK_INTERVAL_S:-60}
RUN_NAME=${1:?usage: bash tests/bench_dev/pull-b300-bench-results.sh <run dir name>}
REMOTE_RSYNC="nice -n 19 ionice -c 3 rsync"

mkdir -p "$LOCAL_RESULTS/$RUN_NAME"
echo "$(date +'%F %T') watching $REMOTE_HOST:$REMOTE_RESULTS/$RUN_NAME every ${CHECK_INTERVAL_S}s"
while true; do
    # "; true": a missing marker or run directory is not an ssh failure.
    if ! markers=$(ssh -o ConnectTimeout=30 "$REMOTE_HOST" \
        "cd $REMOTE_RESULTS/$RUN_NAME 2> /dev/null && ls -d */SETTING_DONE ALL_DONE 2> /dev/null; true"); then
        echo "$(date +'%F %T') cannot reach $REMOTE_HOST, next check in ${CHECK_INTERVAL_S}s" >&2
        sleep "$CHECK_INTERVAL_S"
        continue
    fi
    all_done=0
    failed_copies=0
    for marker in $markers; do
        if [ "$marker" = "ALL_DONE" ]; then
            all_done=1
            continue
        fi
        setting=${marker%/SETTING_DONE}
        if [ -f "$LOCAL_RESULTS/$RUN_NAME/$setting/SYNCED" ]; then
            continue
        fi
        echo "$(date +'%F %T') $setting finished on $REMOTE_HOST, copying"
        if rsync -a --partial --rsync-path="$REMOTE_RSYNC" -e "ssh -o ConnectTimeout=30" \
            "$REMOTE_HOST:$REMOTE_RESULTS/$RUN_NAME/$setting/" "$LOCAL_RESULTS/$RUN_NAME/$setting/"; then
            date +'%F %T' > "$LOCAL_RESULTS/$RUN_NAME/$setting/SYNCED"
            echo "  copied $setting: $(du -sh "$LOCAL_RESULTS/$RUN_NAME/$setting" | cut -f1), $(cat "$LOCAL_RESULTS/$RUN_NAME/$setting/SETTING_DONE")"
        else
            echo "  copying $setting failed (exit $?), retrying at the next check" >&2
            failed_copies=$((failed_copies + 1))
        fi
    done
    if [ "$all_done" = "1" ] && [ "$failed_copies" = "0" ]; then
        # Top-level files only: every setting directory is already here.
        if rsync -a --partial --rsync-path="$REMOTE_RSYNC" -e "ssh -o ConnectTimeout=30" --exclude='*/' \
            "$REMOTE_HOST:$REMOTE_RESULTS/$RUN_NAME/" "$LOCAL_RESULTS/$RUN_NAME/"; then
            echo "$(date +'%F %T') $RUN_NAME finished and copied: $(cat "$LOCAL_RESULTS/$RUN_NAME/ALL_DONE")"
            exit 0
        fi
        echo "  copying the top-level files failed, retrying at the next check" >&2
    fi
    sleep "$CHECK_INTERVAL_S"
done
