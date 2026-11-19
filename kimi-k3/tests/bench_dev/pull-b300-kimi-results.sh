#!/bin/bash
# Copy one test-kimi-k3.sh run from b300 into this repo's bench-results/: the
# whole run directory, once after every finished (layout, config), i.e. after
# every vLLM restart, and once more after the last one.
#
# b300 cannot reach the 6000d host, so this host asks: every CHECK_INTERVAL_S
# seconds (default 60) one ssh lists the run's CONFIG_DONE/<layout>-<config>
# markers and ALL_DONE (no data moved). test-kimi-k3.sh writes a marker after
# its config's benchmarks, so a new marker's copy runs while the next config's
# vLLM is still loading (~14 min for Kimi-K3 on 11-13), before anything is
# measured; rsync on b300 runs at the lowest CPU and IO priority. The serve log
# of the config that is starting is copied as far as it is written; the next
# copy completes it. Copied markers are recorded here in <run>/SYNCED/ and never
# trigger another copy. Exits once ALL_DONE is copied.
#
# Usage, in a tmux session on the 6000d host, with the run directory name that
# test-kimi-k3.sh prints ("results -> bench-results/<name>"):
#   bash tests/bench_dev/pull-b300-kimi-results.sh <run dir name>
set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE_HOST=b300
REMOTE_RESULTS=${REMOTE_RESULTS:-/home/intel/qiuyu/intel-accel-for-llm-kimi-k3/bench-results}
LOCAL_RESULTS=${LOCAL_RESULTS:-$REPO_ROOT/bench-results}
CHECK_INTERVAL_S=${CHECK_INTERVAL_S:-60}
RUN_NAME=${1:?usage: bash tests/bench_dev/pull-b300-kimi-results.sh <run dir name>}
REMOTE_RSYNC="nice -n 19 ionice -c 3 rsync"
LOCAL_RUN=$LOCAL_RESULTS/$RUN_NAME

mkdir -p "$LOCAL_RUN/SYNCED"
echo "$(date +'%F %T') watching $REMOTE_HOST:$REMOTE_RESULTS/$RUN_NAME every ${CHECK_INTERVAL_S}s"
while true; do
    # "; true": a missing marker or run directory is not an ssh failure.
    if ! markers=$(ssh -o ConnectTimeout=30 "$REMOTE_HOST" \
        "cd $REMOTE_RESULTS/$RUN_NAME 2> /dev/null && ls CONFIG_DONE/* ALL_DONE 2> /dev/null; true"); then
        echo "$(date +'%F %T') cannot reach $REMOTE_HOST, next check in ${CHECK_INTERVAL_S}s" >&2
        sleep "$CHECK_INTERVAL_S"
        continue
    fi
    new_markers=""
    for marker in $markers; do
        if [ ! -f "$LOCAL_RUN/SYNCED/$(basename "$marker")" ]; then
            new_markers+=" $marker"
        fi
    done
    if [ -n "$new_markers" ]; then
        echo "$(date +'%F %T') finished on $REMOTE_HOST:$new_markers; copying the run directory"
        if rsync -a --partial --rsync-path="$REMOTE_RSYNC" -e "ssh -o ConnectTimeout=30" \
            "$REMOTE_HOST:$REMOTE_RESULTS/$RUN_NAME/" "$LOCAL_RUN/"; then
            for marker in $new_markers; do
                date +'%F %T' > "$LOCAL_RUN/SYNCED/$(basename "$marker")"
                echo "  copied after $marker: $(cat "$LOCAL_RUN/$marker")"
            done
            echo "  run directory here: $(du -sh "$LOCAL_RUN" | cut -f1)"
        else
            echo "  copying failed (exit $?), retrying at the next check" >&2
        fi
    fi
    if [ -f "$LOCAL_RUN/SYNCED/ALL_DONE" ]; then
        echo "$(date +'%F %T') $RUN_NAME finished and copied"
        exit 0
    fi
    sleep "$CHECK_INTERVAL_S"
done
