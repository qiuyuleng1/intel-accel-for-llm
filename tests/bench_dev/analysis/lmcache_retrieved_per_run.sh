#!/bin/bash
# Per lmcache measurement: distinct "Retrieved N tokens" values (+ line counts) logged
# by the LMCache server between the creation of the measurement's bench log and the
# write of its result .json (birth time / mtime). Warmups run before the bench log is
# created, so they fall outside the window.
# Usage (on b300): bash lmcache_retrieved_per_run.sh <bench-results/Kimi-K3_.../layout>
set -euo pipefail
layout_dir=$1
server_log=$layout_dir/lmcache/lmcache_server.log
for json in $(ls "$layout_dir"/lmcache/in*/lmcache_*.json | sort -V); do
    tag=$(basename "$json" .json)
    start_ts=$(stat -c %W "${json%.json}.log")
    end_ts=$(stat -c %Y "$json")
    start=$(date -d "@$start_ts" '+%Y-%m-%d %H:%M:%S')
    end=$(date -d "@$end_ts" '+%Y-%m-%d %H:%M:%S')
    echo "== $tag  window $start .. $end"
    # The server log lines start with ANSI color codes; strip them before parsing.
    awk -v s="[$start" -v e="[$end" '
        /Retrieved [0-9]+ tokens/ {
            gsub(/\033\[[0-9;]*m/, "")
            ts = $1 " " substr($2, 1, 8)
            if (ts >= s && ts <= e) {
                for (i = 1; i <= NF; i++) {
                    if ($i == "Retrieved") {
                        count[$(i+1)]++
                    }
                }
            }
        }
        END { for (v in count) printf "   Retrieved %s tokens: %d lines\n", v, count[v] }
    ' "$server_log" | sort -k2 -n
done
