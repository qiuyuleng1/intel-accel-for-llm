# Imported by req_analyze1022.py: collect(run_dir) returns the per-request
# async-load timeline of prio / rbl_pr / rbl_br in the 10-22 round, from the
# async_load event lines of ALL 8 TP ranks (profile1022.py used rank 0 only).
# Cell windows as in profile1022.py.
import collections
import datetime
import glob
import os
import re
import sys

SCHEMES = {
    "priority_3level_first_n_per_req_remaining_by_layer": "rbl_pr",
    "priority_3level_first_n_batch_reqs_remaining_by_layer": "rbl_br",
    "priority_3level_first_n_per_req": "prio",
}
MAPS = {"L0-4-8": "4/8", "L0-8-16": "8/16", "L0-16-32": "16/32"}
ev_re = re.compile(
    r"^\(Worker_TP(\d) pid=\d+\) DEBUG (\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*async_load "
    r"step=(\d+) t=(\d+) (.*)$"
)
match_re = re.compile(
    r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+), .*"
    r"async=(\w+), selected_load_layers=(-?\d+), async_load_layers=(-?\d+)"
)
cell_re = re.compile(r"_in(\d+)_hr(\d+)_c(\d+)\.json$")


def ts(s):
    return datetime.datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def _collect_log(run_dir, serve_log, out):
    label = os.path.basename(serve_log)[: -len(".serve.log")]
    cfg = None
    for long, short in SCHEMES.items():
        if label.startswith(long + "_L") and label[len(long) + 1:] in MAPS:
            cfg = (short, MAPS[label[len(long) + 1:]])
    if cfg is None:
        return
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
        bounds.append((start, end, (n, hit, conc)))
        prev, prev_len = end, n
    # steps[rank][step] = [start_t, end_t, wait_ms, nreq]
    steps = [collections.defaultdict(lambda: [0, 0, 0.0, 0]) for _ in range(8)]
    reqs = {}  # rid -> dict
    acts = {}  # step -> list of activated rids (rank 0)
    sync_steps = {}  # step -> list of sync rids (rank 0)
    sync_wall = []
    matched = {}
    with open(serve_log, errors="replace") as f:
        for line in f:
            if line.startswith("(EngineCore") and "get_num_new_matched_tokens" in line:
                m = match_re.search(line)
                if m:
                    matched[m.group(1)] = (int(m.group(2)), m.group(3) == "True", int(m.group(5)))
                continue
            if not line.startswith("(Worker_TP") or "async_load step=" not in line:
                continue
            m = ev_re.search(line)
            if not m:
                continue
            rank = int(m.group(1))
            step = int(m.group(3))
            t = int(m.group(4))
            rest = m.group(5)
            if rest.startswith("step_start"):
                st = steps[rank][step]
                st[0] = t
                st[3] = int(rest.split("reqs=")[1])
            elif rest.startswith("step_end"):
                st = steps[rank][step]
                st[1] = t
                st[2] = float(rest.split("wait_ms=")[1].split()[0])
            elif rest.startswith("submit"):
                kind = rest.split("kind=")[1].split()[0]
                ids = rest.split("reqs=")[1].split()[0].split(",")
                if kind == "sync":
                    if rank == 0:
                        sync_steps.setdefault(step, []).extend(ids)
                        sync_wall.extend([ts(m.group(2))] * len(ids))
                    continue
                if kind != "async" or rank != 0:
                    continue
                for rid in ids:
                    if rid not in reqs:
                        reqs[rid] = {"wall": ts(m.group(2)), "sub": t, "sub_step": step,
                                     "pro": {}, "pro_step": {}}
            elif rest.startswith("promote"):
                for rid in rest.split("reqs=")[1].split(","):
                    if rid in reqs and rank not in reqs[rid]["pro"]:
                        reqs[rid]["pro"][rank] = t
                        reqs[rid]["pro_step"][rank] = step
            elif rest.startswith("activate") and rank == 0:
                a = rest.split("reqs=")[1].split(" held=")[0]
                for rid in a.split(",") if a else []:
                    if rid in reqs and "act" not in reqs[rid]:
                        reqs[rid]["act"] = t
                        reqs[rid]["act_step"] = step
                        acts.setdefault(step, []).append(rid)
    for start, end, key in bounds:
        rows = []
        for rid, rq in reqs.items():
            if not (start < rq["wall"] <= end) or "act" not in rq or len(rq["pro"]) != 8:
                continue
            s = rq["act_step"]
            pro_max = max(rq["pro"].values())
            rows.append({
                "rid": rid,
                "N": matched.get(rid, (0, True, 0))[2],
                "tokens": matched.get(rid, (0, True, 0))[0],
                "sub": rq["sub"],
                "sub_step": rq["sub_step"],
                "pro0": rq["pro"][0],
                "pro_max": pro_max,
                "pro_step0": rq["pro_step"][0],
                "pro_step_max": max(rq["pro_step"].values()),
                "act": rq["act"],
                "act_step": s,
                "k": len(acts[s]),
                "wait0": steps[0][s][2],
                "wait_max": max(steps[r][s][2] for r in range(8)),
                "act_end": steps[0][s][1],
                "act_dur": (steps[0][s][1] - steps[0][s][0]) / 1e6,
                "act_sync": len(sync_steps.get(s, [])),
                "sub_step_end": steps[0][rq["sub_step"]][1],
            })
            # Contents of the submit step and of the step between the all-rank
            # promote and the activation (async scheduling: activation is 2 steps
            # after the promote, so exactly one step is in between).
            for tag, ss in (("sub", rq["sub_step"]), ("mid", max(rq["pro_step"].values()) + 1)):
                rows[-1][tag + "_dur"] = (steps[0][ss][1] - steps[0][ss][0]) / 1e6
                rows[-1][tag + "_wait"] = max(steps[r][ss][2] for r in range(8))
                rows[-1][tag + "_k"] = len(acts.get(ss, []))
                rows[-1][tag + "_sync"] = len(sync_steps.get(ss, []))
                rows[-1][tag + "_n"] = steps[0][ss][3]
            # (2) split: all-rank promote -> end of its step, the steps strictly
            # between, and activation step start -> activate log line (host-side
            # submits of the activation step happen before that line).
            pro_s = max(rq["pro_step"].values())
            rows[-1]["between_dur"] = (steps[0][s][0] - steps[0][pro_s][1]) / 1e6
            rows[-1]["act_pre"] = (rq["act"] - steps[0][s][0]) / 1e6
        nsync = sum(1 for w in sync_wall if start < w <= end)
        out[cfg + key] = {"rows": rows, "nsync": nsync}
    print(label, "done", file=sys.stderr)


def collect(run_dir):
    # (scheme, map, input_len, hit, conc) -> {"rows": per async request, "nsync"}
    out = {}
    for serve_log in sorted(glob.glob(f"{run_dir}/*.serve.log")):
        _collect_log(run_dir, serve_log, out)
    return out
