#!/bin/bash
# usage: scratch_sampler.sh <out_file>
# Every 0.5 s, append "<unix_time> <max_GiB> <rank0_GiB> ... <rank7_GiB>" with the
# ScratchPool bytes in use of every rank (GET /v1/cache/status via the controller).
out=$1
while true; do
    t=$(date +%s.%N)
    curl -s --max-time 2 http://localhost:18710/v1/cache/status | python3 -c '
import json
import sys
try:
    workers = json.load(sys.stdin)["workers"]
except Exception:
    sys.exit(0)
values = []
for rank in sorted(workers, key=int):
    status = workers[rank]
    if isinstance(status, dict) and status.get("pool_total_tensors"):
        block = status["pool_total_bytes"] / status["pool_total_tensors"]
        values.append(status["pool_in_use"] * block / 2**30)
if values:
    parts = []
    for value in values:
        parts.append("%.2f" % value)
    print("%s %.3f %s" % (sys.argv[1], max(values), " ".join(parts)))
' "$t" >> "$out"
    sleep 0.5
done
