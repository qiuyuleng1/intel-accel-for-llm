# Usage: python3 tests/async_load_dev/analysis/events1022.py [run_dir]
# Rank-0 async_load event lines of each serve.log in the 10-22 round, split into
# cells: window = (end of the previous cell or of this input length's prewarm,
# end of this cell], end = mtime of <tag>.json; includes the cell's 5 warmups.
# Per config and hit rate: steps, steps with an activation of >= 1 / >= 2
# requests, steps with a sync batch, steps with both, forward-thread wait_ms.
import collections
import datetime
import glob
import os
import re
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "_data/perf-qwen72b/20261022_101029"
SHORT = {
    "priority_3level_first_n_per_req_remaining_by_layer": "rbl_pr",
    "priority_3level_first_n_batch_reqs_remaining_by_layer": "rbl_br",
    "priority_3level_first_n_per_req": "prio",
    "batch_reqs_async_load_submit": "batch",
    "sync": "sync",
}
MAPS = {"L0-4-8": "4/8", "L0-8-16": "8/16", "L0-16-32": "16/32"}
line_re = re.compile(
    r"^\(Worker_TP0 pid=\d+\) DEBUG (\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*async_load step=(\d+) "
    r"t=\d+ (step_start|step_end wait_ms=([\d.]+)|submit kind=sync |activate reqs=(\S*))"
)
cell_re = re.compile(r"_in(\d+)_hr(\d+)_c(\d+)\.json$")


def cfg_of(label):
    for long, short in SHORT.items():
        if label == long:
            return short
        if label.startswith(long + "_L"):
            return f"{short} {MAPS[label[len(long) + 1:]]}"
    raise SystemExit(f"unknown label {label}")


def ts(s):
    return datetime.datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


# cell -> stats; key (cfg, input_len, hit, conc)
cells = {}
for serve_log in sorted(glob.glob(f"{run_dir}/*.serve.log")):
    label = os.path.basename(serve_log)[: -len(".serve.log")]
    cfg = cfg_of(label)
    bounds = []  # (start, end, key)
    ends = []
    for js in glob.glob(f"{run_dir}/{label}_in*_hr*_c*.json"):
        m = cell_re.search(js)
        ends.append((os.path.getmtime(js), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    ends.sort()
    prev = 0.0
    prev_len = None
    for end, n, hit, conc in ends:
        start = prev
        if n != prev_len:
            start = max(start, os.path.getmtime(f"{run_dir}/{label}.in{n}.prewarm.log"))
        bounds.append((start, end, (cfg, n, hit, conc)))
        prev, prev_len = end, n
    steps = collections.defaultdict(lambda: {"t": 0.0, "sync": False, "act": 0, "wait": 0.0})
    with open(serve_log, errors="replace") as f:
        for line in f:
            if not line.startswith("(Worker_TP0") or "async_load step=" not in line:
                continue
            m = line_re.search(line)
            if not m:
                continue
            st = steps[int(m.group(2))]
            kind = m.group(3)
            if kind == "step_start":
                st["t"] = ts(m.group(1))
            elif kind.startswith("step_end"):
                st["wait"] = float(m.group(4))
            elif kind.startswith("submit"):
                st["sync"] = True
            elif m.group(5):
                st["act"] = m.group(5).count(",") + 1
    for start, end, key in bounds:
        s = {"steps": 0, "act1": 0, "act2": 0, "sync": 0, "sync_act": 0, "wait_ms": 0.0}
        for st in steps.values():
            if start < st["t"] <= end:
                s["steps"] += 1
                s["act1"] += st["act"] >= 1
                s["act2"] += st["act"] >= 2
                s["sync"] += st["sync"]
                s["sync_act"] += st["sync"] and st["act"] >= 1
                s["wait_ms"] += st["wait"]
        cells[key] = s

cfgs = sorted({k[0] for k in cells}, key=lambda c: (c.split()[-1], c))
print("cfg | hit | steps | act>=1 | act>=2 | sync steps | sync+act steps | wait s")
for cfg in cfgs:
    for hit in (80, 98, 100):
        tot = collections.Counter()
        for key, s in cells.items():
            if key[0] == cfg and key[2] == hit:
                for k, v in s.items():
                    tot[k] += v
        print(f"{cfg} | {hit} | {tot['steps']} | {tot['act1']} | {tot['act2']} | "
              f"{tot['sync']} | {tot['sync_act']} | {tot['wait_ms'] / 1000:.1f}")
print("\npaired wait_ms per map (a / b, summed over the 27 cells; per-cell a<b count)")
for a, b in (("rbl_pr", "prio"), ("rbl_br", "prio"), ("batch", "prio"), ("rbl_br", "batch")):
    for mp in ("4/8", "8/16", "16/32"):
        wa = wb = 0.0
        less = n = 0
        for key, s in cells.items():
            if key[0] == f"{a} {mp}":
                other = cells[(f"{b} {mp}",) + key[1:]]
                wa += s["wait_ms"]
                wb += other["wait_ms"]
                n += 1
                less += s["wait_ms"] < other["wait_ms"]
        print(f"{a}/{b} {mp}: {wa / 1000:.1f} s / {wb / 1000:.1f} s = {wa / wb:.2f}, "
              f"a lower in {less}/{n} cells")
print("\nhit 98 per cell: wait s (sync+act steps) for prio / rbl_pr / rbl_br / batch")
for mp in ("4/8", "8/16", "16/32"):
    for key in sorted(k for k in cells if k[0] == f"prio {mp}" and k[2] == 98):
        cols = []
        for s in ("prio", "rbl_pr", "rbl_br", "batch"):
            c = cells[(f"{s} {mp}",) + key[1:]]
            cols.append(f"{c['wait_ms'] / 1000:.1f} ({c['sync_act']})")
        print(f"{mp} in{key[1]} c{key[3]} | {' | '.join(cols)}")
