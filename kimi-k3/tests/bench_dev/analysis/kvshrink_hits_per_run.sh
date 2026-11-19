#!/bin/bash
# Per kvshrink measurement: distinct per-request "externally-cached tokens" values
# (+ request counts) from the connector's get_num_new_matched_tokens log lines,
# between the creation of the measurement's bench log and the write of its result
# .json. The scheduler calls get_num_new_matched_tokens several times per request
# while it waits, so only the last value of each request id is counted.
# Usage (on b300): bash kvshrink_hits_per_run.sh <bench-results/Kimi-K3_.../layout>
set -euo pipefail
layout_dir=$1
serve_log=$layout_dir/kvshrink/serve.log
for json in $(ls "$layout_dir"/kvshrink/in*/kvshrink_*.json | sort -V); do
    tag=$(basename "$json" .json)
    start=$(date -d "@$(stat -c %W "${json%.json}.log")" '+%m-%d %H:%M:%S')
    end=$(date -d "@$(stat -c %Y "$json")" '+%m-%d %H:%M:%S')
    echo "== $tag  window $start .. $end"
    sed 's/\x1b\[[0-9;]*m//g' "$serve_log" | awk -v s="$start" -v e="$end" '
        /get_num_new_matched_tokens/ {
            if (!match($0, /[0-9][0-9]-[0-9][0-9] [0-9:]{8}/)) next
            ts = substr($0, RSTART, RLENGTH)
            if (ts < s || ts > e) next
            match($0, /req-[^,]+/)
            request_id = substr($0, RSTART, RLENGTH)
            match($0, /externally-cached tokens: [0-9]+/)
            last_hit[request_id] = substr($0, RSTART + 26, RLENGTH - 26)
        }
        END {
            for (request_id in last_hit) count[last_hit[request_id]]++
            for (v in count) printf "   externally-cached %s tokens: %d requests\n", v, count[v]
        }
    ' | sort -k3 -n
done
