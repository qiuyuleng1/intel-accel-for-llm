# OMP-Main time inside activation steps (diagnosis helper).
# usage: python3 step_breakdown.py <tag>.trace.json.gz <label>.serve.log [rank]
# For every forward step of <rank> (default 0) that activates async requests,
# split the step's wall time [step_start, step_end] by what OMP-Main ran:
#   act    unzip tasks of the requests activated in this step (what the forward waits on)
#   other  other unzip tasks (heads / tails of requests not in this forward)
#   zip    zip (store) tasks
#   rel    release (task callable + captures destroyed after the task returned)
#   idle   nothing running
# and compare with steps that activate nothing.
import gzip
import json
import re
import statistics
import sys

sys.path.insert(0, "tests/async_load_dev")
import trace_timeline as tt  # noqa: E402

STEP_RE = re.compile(r"\(Worker_TP(\d+) .*async_load step=(\d+) t=(\d+) (step_start|step_end|activate)(.*)")
REQS_RE = re.compile(r"reqs=(\S*)")
MS = 1e6

trace_path, log_path = sys.argv[1], sys.argv[2]
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
events = tt.load_trace(trace_path)[rank]
t_min = min(e["enqueue_ns"] for e in events)
t_max = max(e["release_ns"] for e in events)

steps = {}
activated = {}
with open(log_path, errors="replace") as f:
    for line in f:
        m = STEP_RE.search(line)
        if not m or int(m.group(1)) != rank:
            continue
        step, t, kind, rest = int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        if t < t_min or t > t_max:
            continue
        if kind == "step_start":
            steps[step] = [t, None]
        elif kind == "step_end" and step in steps:
            steps[step][1] = t
        elif kind == "activate":
            reqs = REQS_RE.search(rest).group(1)
            if reqs:
                activated[step] = set(reqs.split(","))


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def breakdown(s0, s1, act_reqs):
    out = {"act": 0, "other": 0, "zip": 0, "rel": 0}
    for e in events:
        if e["end_ns"] < s0 and e["release_ns"] < s0:
            continue
        if e["start_ns"] > s1:
            continue
        run = overlap(e["start_ns"], e["end_ns"], s0, s1)
        rel = overlap(e["end_ns"], e["release_ns"], s0, s1)
        out["rel"] += rel
        if e["kind"].startswith("zip"):
            out["zip"] += run
        elif e["kind"].startswith("unzip"):
            mine = False
            for r in e["reqs"]:
                if r in act_reqs:
                    mine = True
            if mine:
                out["act"] += run
            else:
                out["other"] += run
    busy = out["act"] + out["other"] + out["zip"] + out["rel"]
    out["idle"] = max(0, (s1 - s0) - busy)
    return out


rows_act = []
rows_plain = []
for step in sorted(steps):
    s0, s1 = steps[step]
    if s1 is None:
        continue
    if step in activated:
        b = breakdown(s0, s1, activated[step])
        b["dur"] = s1 - s0
        b["n"] = len(activated[step])
        rows_act.append(b)
    else:
        rows_plain.append(s1 - s0)


def mean(rows, key):
    values = []
    for r in rows:
        values.append(r[key])
    return statistics.fmean(values) / MS if values else 0.0


print(f"rank {rank}: activation steps={len(rows_act)} other steps={len(rows_plain)}"
      f" other step mean={statistics.fmean(rows_plain) / MS:.1f} ms"
      f" med={sorted(rows_plain)[len(rows_plain) // 2] / MS:.1f} ms")
n_act = []
for r in rows_act:
    n_act.append(r["n"])
print(f"  activation step mean={mean(rows_act, 'dur'):.1f} ms"
      f" (reqs activated per step mean={statistics.fmean(n_act):.1f}):"
      f" act={mean(rows_act, 'act'):.1f} other={mean(rows_act, 'other'):.1f}"
      f" zip={mean(rows_act, 'zip'):.1f} rel={mean(rows_act, 'rel'):.1f}"
      f" idle={mean(rows_act, 'idle'):.1f} ms")
