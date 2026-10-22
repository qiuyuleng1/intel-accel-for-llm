# Usage: python3 tests/async_load_dev/analysis/fixed_layers.py <run_dir>
# 10-21 round (perf-qwen72b-bench.sh FIXED_LAYERS / REPEATS): per cell and config the
# mean TTFT of each repeat, their mean, repeat spread, ratio vs sync; geomean per config;
# async share and selected layers per config from serve.log (connector lines).
import collections
import csv
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys

run_dir = sys.argv[1]
rows = list(csv.DictReader(open(f"{run_dir}/summary.csv")))


def cfg_of(r):
    if r["scheme"] == "sync":
        return "sync"
    if r["dynamic_map"] != "none":
        return "mapB"
    return "N" + ("all" if r["fixed_layers"] == "-1" else r["fixed_layers"])


order = ["mapB", "N4", "N8", "N16", "N32", "N48", "N64", "Nall", "sync"]
data = collections.defaultdict(dict)  # (hit, conc) -> cfg -> {rep: row}
for r in rows:
    data[(int(r["hit_rate"]), int(r["concurrency"]))].setdefault(cfg_of(r), {})[int(r["repeat"])] = r

print("== mean TTFT ms: r1 / r2 -> mean (spread = |r1-r2|/mean), ratio vs sync mean")
spreads = []
for cell in sorted(data):
    s = statistics.mean(float(x["mean_ttft_ms"]) for x in data[cell]["sync"].values())
    print(f"-- hit {cell[0]} c{cell[1]} (sync mean {s:.0f})")
    for c in order:
        reps = data[cell][c]
        v = [float(reps[k]["mean_ttft_ms"]) for k in sorted(reps)]
        m = statistics.mean(v)
        sp = abs(v[0] - v[1]) / m * 100 if len(v) == 2 else 0
        spreads.append(sp)
        p95 = statistics.mean(float(x["p95_ttft_ms"]) for x in reps.values())
        tpot = statistics.mean(float(x["mean_tpot_ms"]) for x in reps.values())
        dur = statistics.mean(float(x["duration"]) for x in reps.values())
        print(f"   {c:5s} {v[0]:7.0f} / {v[1]:7.0f} -> {m:7.0f}  spread {sp:4.0f}%  vs sync {m / s:5.2f}"
              f"  p95 {p95:6.0f}  tpot {tpot:4.0f}  dur {dur:5.0f}")
print(f"repeat spread: median {statistics.median(spreads):.0f}%  max {max(spreads):.0f}%  "
      f">=20%: {sum(1 for x in spreads if x >= 20)}/{len(spreads)}")

print("== geomean of (config mean / sync mean) over cells")
for c in order:
    out = []
    for sel, name in ((lambda k: True, "all"), (lambda k: k[0] == 95, "h95"), (lambda k: k[0] == 98, "h98"),
                      (lambda k: k[0] == 100, "h100"), (lambda k: k[1] == 4, "c4"), (lambda k: k[1] == 8, "c8")):
        l = []
        for cell in data:
            if sel(cell):
                m = statistics.mean(float(x["mean_ttft_ms"]) for x in data[cell][c].values())
                s = statistics.mean(float(x["mean_ttft_ms"]) for x in data[cell]["sync"].values())
                l.append(math.log(m / s))
        out.append(f"{name}={(math.exp(statistics.mean(l)) - 1) * 100:+.0f}%")
    print(f"   {c:5s} " + " ".join(out))

print("== async share / selected layers per config and cell (connector lines in the cell window)")
conn_re = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*get_num_new_matched_tokens, req-\S+, "
                     r"externally-cached tokens: (\d+), .*async=(\w+), selected_load_layers=(-?\d+)")
labels = {"mapB": "priority_3level_first_n_per_req_L0-16-32", "N4": "priority_3level_first_n_per_req_N4",
          "N16": "priority_3level_first_n_per_req_N16", "N32": "priority_3level_first_n_per_req_N32",
          "Nall": "priority_3level_first_n_per_req_Nall"}
for c, label in labels.items():
    lines = []
    for line in open(f"{run_dir}/{label}.serve.log", errors="replace"):
        m = conn_re.search(line)
        if m:
            lines.append(m.groups())
    parts = []
    for path in sorted(glob.glob(f"{run_dir}/{label}_in32000_hr*_c*_r*.json")):
        d = json.load(open(path))
        end = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
        start = end - datetime.timedelta(seconds=d["duration"])
        n = a = 0
        sel = collections.Counter()
        for ts, tok, asy, lay in lines:
            t = datetime.datetime.strptime(f"{end.year}-{ts}", "%Y-%m-%d %H:%M:%S")
            if start <= t <= end and int(tok) > 0:
                n += 1
                if asy == "True":
                    a += 1
                    sel[lay] += 1
        tag = os.path.basename(path)[len(label) + 1:-5]
        parts.append(f"{tag}:{100 * a / max(n, 1):.0f}%{dict(sel)}")
    print(f"   {c}: " + "  ".join(parts))
