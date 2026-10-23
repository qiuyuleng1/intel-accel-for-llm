# Usage: python3 tests/async_load_dev/analysis/profile1022.py [run_dir]
# Forward-thread wait profile of prio / rbl_pr / rbl_br in the 10-22 round, from
# the rank-0 async_load event lines (same cell windows as events1022.py).
# Per config and hit rate: steps, total wait_ms, wait split by the step's
# activation (none / 1 request / >= 2 requests), p99 and max step wait,
# save_ms; then per map the per-cell paired comparison.
import collections
import datetime
import glob
import os
import re
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "_data/perf-qwen72b/20261022_101029"
SCHEMES = {
    "priority_3level_first_n_per_req_remaining_by_layer": "rbl_pr",
    "priority_3level_first_n_batch_reqs_remaining_by_layer": "rbl_br",
    "priority_3level_first_n_per_req": "prio",
}
MAPS = {"L0-4-8": "4/8", "L0-8-16": "8/16", "L0-16-32": "16/32"}
line_re = re.compile(
    r"^\(Worker_TP0 pid=\d+\) DEBUG (\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*async_load step=(\d+) "
    r"t=\d+ (step_start|step_end wait_ms=([\d.]+) save_ms=([\d.]+)|activate reqs=(\S*))"
)
cell_re = re.compile(r"_in(\d+)_hr(\d+)_c(\d+)\.json$")
# Per-request events: first async submit (head), promote, activate; t in ns.
req_re = re.compile(
    r"^\(Worker_TP0 pid=\d+\) DEBUG (\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*async_load step=(\d+) "
    r"t=(\d+) (submit kind=async reqs|promote reqs|activate reqs)=(\S+)"
)


def ts(s):
    return datetime.datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def pctl(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


cells = {}  # (cfg, input_len, hit, conc) -> list of step dicts
req_cells = {}  # (cfg, input_len, hit, conc) -> list of request timelines
for serve_log in sorted(glob.glob(f"{run_dir}/*.serve.log")):
    label = os.path.basename(serve_log)[: -len(".serve.log")]
    cfg = None
    for long, short in SCHEMES.items():
        if label.startswith(long + "_L") and label[len(long) + 1:] in MAPS:
            cfg = f"{short} {MAPS[label[len(long) + 1:]]}"
    if cfg is None:
        continue
    ends = []
    for js in glob.glob(f"{run_dir}/{label}_in*_hr*_c*.json"):
        m = cell_re.search(js)
        ends.append((os.path.getmtime(js), int(m.group(1)), int(m.group(2)), int(m.group(3))))
    ends.sort()
    bounds = []
    prev = 0.0
    prev_len = None
    for end, n, hit, conc in ends:
        start = prev
        if n != prev_len:
            start = max(start, os.path.getmtime(f"{run_dir}/{label}.in{n}.prewarm.log"))
        bounds.append((start, end, (cfg, n, hit, conc)))
        prev, prev_len = end, n
    steps = collections.defaultdict(lambda: {"t": 0.0, "act": 0, "wait": 0.0, "save": 0.0})
    reqs = {}
    with open(serve_log, errors="replace") as f:
        for line in f:
            if not line.startswith("(Worker_TP0") or "async_load step=" not in line:
                continue
            r = req_re.search(line)
            if r:
                t = int(r.group(3))
                for rid in r.group(5).split(","):
                    if r.group(4).startswith("submit"):
                        if rid not in reqs:
                            reqs[rid] = {"wall": ts(r.group(1)), "sub": t}
                    elif rid in reqs and r.group(4).startswith("promote"):
                        reqs[rid]["pro"] = t
                    elif rid in reqs and "act" not in reqs[rid]:
                        reqs[rid]["act"] = t
                        reqs[rid]["act_step"] = int(r.group(2))
            m = line_re.search(line)
            if not m:
                continue
            st = steps[int(m.group(2))]
            kind = m.group(3)
            if kind == "step_start":
                st["t"] = ts(m.group(1))
            elif kind.startswith("step_end"):
                st["wait"] = float(m.group(4))
                st["save"] = float(m.group(5))
            elif m.group(6):
                st["act"] = m.group(6).count(",") + 1
    for start, end, key in bounds:
        sel = []
        for st in steps.values():
            if start < st["t"] <= end:
                sel.append(st)
        cells[key] = sel
        rsel = []
        for rq in reqs.values():
            if start < rq["wall"] <= end and "pro" in rq and "act" in rq:
                rq["act_wait"] = steps[rq["act_step"]]["wait"]
                rsel.append(rq)
        req_cells[key] = rsel


def summary(step_lists):
    s = {"steps": 0, "wait": 0.0, "save": 0.0, "n0": 0, "w0": 0.0, "n1": 0, "w1": 0.0,
         "n2": 0, "w2": 0.0, "waits": []}
    for sl in step_lists:
        for st in sl:
            s["steps"] += 1
            s["wait"] += st["wait"]
            s["save"] += st["save"]
            s["waits"].append(st["wait"])
            cat = "0" if st["act"] == 0 else ("1" if st["act"] == 1 else "2")
            s["n" + cat] += 1
            s["w" + cat] += st["wait"]
    return s


order = []
for mp in ("4/8", "8/16", "16/32"):
    for sch in ("prio", "rbl_pr", "rbl_br"):
        order.append(f"{sch} {mp}")
print("cfg | hit | steps | wait s | no-act steps / wait s | act=1 steps / wait s / ms per step | "
      "act>=2 steps / wait s / ms per step | p99 step wait ms | max ms | save s")
for cfg in order:
    for hit in (80, 98, 100, "all"):
        sl = []
        for key, v in cells.items():
            if key[0] == cfg and (hit == "all" or key[2] == hit):
                sl.append(v)
        s = summary(sl)
        m1 = s["w1"] / s["n1"] if s["n1"] else 0
        m2 = s["w2"] / s["n2"] if s["n2"] else 0
        print(f"{cfg} | {hit} | {s['steps']} | {s['wait'] / 1000:.1f} | {s['n0']} / {s['w0'] / 1000:.1f} | "
              f"{s['n1']} / {s['w1'] / 1000:.1f} / {m1:.0f} | {s['n2']} / {s['w2'] / 1000:.1f} / {m2:.0f} | "
              f"{pctl(s['waits'], 0.99):.0f} | {max(s['waits']):.0f} | {s['save'] / 1000:.1f}")
print("\nper activated request: mean wait of the step it was activated in (ms), activated requests")
print("cfg | hit 80 | hit 98 | hit 100 | all")
for cfg in order:
    cols = []
    for hit in (80, 98, 100, "all"):
        w = 0.0
        n = 0
        for key, v in cells.items():
            if key[0] == cfg and (hit == "all" or key[2] == hit):
                for st in v:
                    w += st["wait"] * st["act"]
                    n += st["act"]
        cols.append(f"{w / n:.0f} ({n})" if n else "-")
    print(f"{cfg} | {' | '.join(cols)}")
print("\nasync request timeline (ms, mean over requests): submit->promote / promote->activate / "
      "activation-step wait / sum, requests")
print("cfg | hit 80 | hit 98 | hit 100")
for cfg in order:
    cols = []
    for hit in (80, 98, 100):
        a = b = c = 0.0
        n = 0
        for key, v in req_cells.items():
            if key[0] == cfg and key[2] == hit:
                for rq in v:
                    a += (rq["pro"] - rq["sub"]) / 1e6
                    b += (rq["act"] - rq["pro"]) / 1e6
                    c += rq["act_wait"]
                    n += 1
        cols.append(f"{a / n:.0f} / {b / n:.0f} / {c / n:.0f} / {(a + b + c) / n:.0f} ({n})")
    print(f"{cfg} | {' | '.join(cols)}")
print("\nper-cell paired total wait (a vs b): a lower / higher cells, geomean of a/b over cells with b > 0")
for mp in ("4/8", "8/16", "16/32"):
    for a, b in (("rbl_pr", "prio"), ("rbl_br", "prio"), ("rbl_br", "rbl_pr")):
        lower = higher = 0
        logs = []
        for key, v in cells.items():
            if key[0] != f"{a} {mp}":
                continue
            wa = sum(st["wait"] for st in v)
            wb = sum(st["wait"] for st in cells[(f"{b} {mp}",) + key[1:]])
            lower += wa < wb
            higher += wa > wb
            if wa > 0 and wb > 0:
                logs.append(wa / wb)
        g = 1.0
        if logs:
            prod = 1.0
            for x in logs:
                prod *= x
            g = prod ** (1 / len(logs))
        print(f"{a}/{b} {mp}: lower {lower}, higher {higher}, geomean {g:.2f}")
