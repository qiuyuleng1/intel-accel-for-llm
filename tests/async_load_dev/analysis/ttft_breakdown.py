# Per-request TTFT breakdown for one traced cell (diagnosis helper).
# usage: python3 ttft_breakdown.py <tag>.json <label>.serve.log <cell_index>
#   <tag>.json      vllm bench serve result saved with --save-detailed
#   <label>.serve.log  server log of a TRACE=1 run (async_load event lines)
#   <cell_index>    0-based index of the cell among the measured cells of this
#                   server (the prewarm's bench prefix is skipped)
# Client perf_counter, the event-log t= and the TaskQueue trace all use
# CLOCK_MONOTONIC, so all times below are on one clock.
import json
import re
import statistics
import sys

EVENT_RE = re.compile(r"\(Worker_TP0 .*async_load step=(\d+) t=(\d+) (\w+)(.*)")
REQS_RE = re.compile(r"reqs=(\S*)")
KIND_RE = re.compile(r"kind=(\w+)")
BENCH_RE = re.compile(r"cmpl-(bench-[0-9a-f]+-)(\d+)-0-")
NS = 1e9
MS = 1e6

result_path, log_path, cell_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(result_path) as f:
    result = json.load(f)
starts = result["start_times"]
ttfts = result["ttfts"]

prefixes = []
submit = {}
submit_kind = {}
promote = {}
activate = {}
steps = {}
with open(log_path, errors="replace") as f:
    for line in f:
        for match in BENCH_RE.finditer(line):
            if match.group(1) not in prefixes:
                prefixes.append(match.group(1))
        match = EVENT_RE.search(line)
        if not match:
            continue
        step, t, kind, rest = int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)
        if kind == "step_start":
            steps.setdefault(step, [t, None, rest.strip()])
            continue
        if kind == "step_end":
            if step in steps:
                steps[step][1] = t
            continue
        reqs_match = REQS_RE.search(rest)
        if not reqs_match or not reqs_match.group(1):
            continue
        for req in reqs_match.group(1).split(","):
            bench = BENCH_RE.search(req)
            if not bench:
                continue
            key = (bench.group(1), int(bench.group(2)))
            if kind == "submit":
                if key not in submit:
                    submit[key] = t
                    submit_kind[key] = KIND_RE.search(rest).group(1)
            elif kind == "promote":
                promote.setdefault(key, t)
            elif kind == "activate":
                activate.setdefault(key, t)

prefix = prefixes[cell_index + 1]
step_list = []
for step in sorted(steps):
    start, end, info = steps[step]
    if end is not None:
        step_list.append((start, end, info))


def step_at(t):
    """The step whose [start, end] contains t, or the first one starting after t."""
    for start, end, info in step_list:
        if end >= t:
            return start, end, info
    return None


rows = []
for i in range(len(starts)):
    key = (prefix, i)
    if key not in submit or ttfts[i] <= 0:
        continue
    s = starts[i] * NS
    first = s + ttfts[i] * NS
    row = {"i": i, "kind": submit_kind[key], "ttft": ttfts[i] * 1000,
           "to_submit": (submit[key] - s) / MS}
    if key in activate:
        row["submit_to_promote"] = (promote[key] - submit[key]) / MS
        row["promote_to_activate"] = (activate[key] - promote[key]) / MS
        row["activate_to_first"] = (first - activate[key]) / MS
        found = step_at(activate[key])
    else:
        row["submit_to_first"] = (first - submit[key]) / MS
        found = step_at(submit[key])
    if found:
        row["first_step_ms"] = (found[1] - found[0]) / MS
    rows.append(row)

window_start = min(starts) * NS
window_end = 0
for i in range(len(starts)):
    request_end = starts[i] + ttfts[i] + sum(result["itls"][i])
    window_end = max(window_end, request_end * NS)
cell_steps = []
for start, end, info in step_list:
    if start >= window_start and end <= window_end:
        cell_steps.append((end - start) / MS)

print(f"prefix={prefix} requests={len(rows)} of {len(starts)}"
      f" steps_in_window={len(cell_steps)}"
      f" step_ms mean={statistics.fmean(cell_steps):.1f}"
      f" med={sorted(cell_steps)[len(cell_steps) // 2]:.1f}")
keys = ["ttft", "to_submit", "submit_to_promote", "promote_to_activate",
        "activate_to_first", "submit_to_first", "first_step_ms"]
for kind in ("async", "sync"):
    subset = []
    for row in rows:
        if row["kind"] == kind:
            subset.append(row)
    if not subset:
        continue
    parts = []
    for key in keys:
        values = []
        for row in subset:
            if key in row:
                values.append(row[key])
        if values:
            values.sort()
            parts.append(f"{key}={statistics.fmean(values):.0f}/{values[len(values) // 2]:.0f}")
    print(f"  {kind:5s} n={len(subset):3d} " + " ".join(parts))
