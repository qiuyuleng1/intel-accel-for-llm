# Usage: python3 tests/async_load_dev/analysis/p0_at_step_start.py <trace.json.gz> <serve.log> [ranks]
# Does OMP-Main P0 hold any task when a step's start_load_kv() begins? For each
# step_start (event log, same clock as the queue trace) inside the traced window,
# count tasks that are in P0 and not started yet: submitted at P0 (submit_priority 0)
# or raised into P0 (p0_enter_ns > 0) before t, with start_ns > t. Unzip tasks only.
import collections
import gzip
import json
import re
import sys

trace_path, log_path = sys.argv[1], sys.argv[2]
ranks = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0"]
d = json.load(gzip.open(trace_path))
step_re = re.compile(r"\(Worker_TP(\d+) .*async_load step=(\d+) t=(\d+) step_start")
steps = collections.defaultdict(list)
with open(log_path, errors="replace") as f:
    for line in f:
        m = step_re.search(line)
        if m:
            steps[m.group(1)].append((int(m.group(2)), int(m.group(3))))

for rank in ranks:
    ev = d["workers"][rank]["events"]
    p0 = []
    lo = None
    hi = None
    for e in ev:
        lo = e["enqueue_ns"] if lo is None else min(lo, e["enqueue_ns"])
        hi = e["end_ns"] if hi is None else max(hi, e["end_ns"])
        if not e["label"].startswith("unzip"):
            continue
        if e["submit_priority"] == 0:
            entered = e["enqueue_ns"]
        elif e["p0_enter_ns"] > 0:
            entered = e["p0_enter_ns"]
        else:
            continue
        p0.append((entered, e["start_ns"], e["label"]))
    n_steps = 0
    nonempty = []
    for step, t in steps[rank]:
        # skip the first second of the window: tasks enqueued before tracing was on
        # are missing from the trace
        if t < lo + 1_000_000_000 or t > hi:
            continue
        n_steps += 1
        pending = []
        for entered, start, label in p0:
            if entered < t < start:
                pending.append(label)
        if pending:
            nonempty.append((step, len(pending), pending[0]))
    print(f"rank {rank}: steps in window {n_steps}, steps with P0 non-empty at step_start {len(nonempty)}")
    for step, n, label in nonempty[:8]:
        print(f"   step {step}: {n} pending, e.g. {label[:90]}")
