# Usage: python3 tests/async_load_dev/analysis/req_analyze1022.py <metric,...> [run_dir]
# Per (map, hit, scheme) request means of the metrics computed in metrics().
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from req_profile1022 import collect  # noqa: E402

RUN_DIR = sys.argv[2] if len(sys.argv) > 2 else "_data/perf-qwen72b/20261022_101029"
data = collect(RUN_DIR)
SCH = ("prio", "rbl_pr", "rbl_br")
MAPS = ("4/8", "8/16", "16/32")


def m(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def metrics(rows):
    if not rows:
        return None
    r = {}
    r["n"] = len(rows)
    r["p1_r0"] = m([(x["pro0"] - x["sub"]) / 1e6 for x in rows])
    r["p2_r0"] = m([(x["act"] - x["pro0"]) / 1e6 for x in rows])
    r["p3_r0"] = m([x["wait0"] for x in rows])
    r["p1_max"] = m([(x["pro_max"] - x["sub"]) / 1e6 for x in rows])
    r["p2_max"] = m([(x["act"] - x["pro_max"]) / 1e6 for x in rows])
    r["p3_max"] = m([x["wait_max"] for x in rows])
    r["k"] = m([x["k"] for x in rows])
    r["share"] = m([x["wait_max"] / x["k"] for x in rows])
    r["dur"] = m([x["act_dur"] for x in rows])
    r["total"] = m([(x["act_end"] - x["sub"]) / 1e6 for x in rows])
    r["same_step"] = m([1.0 if x["pro_step0"] == x["sub_step"] else 0.0 for x in rows])
    r["skew_steps"] = m([x["pro_step_max"] - x["pro_step0"] for x in rows])
    r["gap"] = m([x["act_step"] - x["pro_step_max"] for x in rows])
    r["sync_in_act"] = m([1.0 if x["act_sync"] else 0.0 for x in rows])
    r["N"] = m([x["N"] for x in rows])
    r["rem"] = m([(x["sub_step_end"] - x["sub"]) / 1e6 for x in rows])
    r["between_dur"] = m([x["between_dur"] for x in rows])
    r["act_pre"] = m([x["act_pre"] for x in rows])
    r["gap_max"] = max(x["act_step"] - x["pro_step_max"] for x in rows)
    r["gap_gt2"] = m([1.0 if x["act_step"] - x["pro_step_max"] > 2 else 0.0 for x in rows])
    # Host idle between two steps (step_end -> next step_start), from requests
    # with exactly one step between promote and activation.
    r["interstep"] = m([(x["between_dur"] - x["mid_dur"]) / 2 for x in rows
                        if x["act_step"] - x["pro_step_max"] == 2])
    # Requests promoted (on all ranks) in the same step as this one.
    by_pro = {}
    for x in rows:
        by_pro[x["pro_step_max"]] = by_pro.get(x["pro_step_max"], 0) + 1
    r["pk"] = m([by_pro[x["pro_step_max"]] for x in rows])
    for tag in ("sub", "mid"):
        for f in ("dur", "wait", "k", "sync", "n"):
            r[f"{tag}_{f}"] = m([x[f"{tag}_{f}"] for x in rows])
        r[f"{tag}_has_act"] = m([1.0 if x[f"{tag}_k"] else 0.0 for x in rows])
        r[f"{tag}_has_sync"] = m([1.0 if x[f"{tag}_sync"] else 0.0 for x in rows])
    return r


def fmt(r, keys):
    return " | ".join(f"{r[k]:.0f}" if abs(r[k]) >= 10 else f"{r[k]:.2f}" for k in keys)


if __name__ == "__main__":
    keys = sys.argv[1].split(",")
    if os.environ.get("PER_CELL"):
        # One row per (map, input len, concurrency) cell of hit PER_CELL.
        hit = int(os.environ["PER_CELL"])
        print("map | input len | conc | " + " | ".join(f"{s} {k}" for k in keys for s in SCH))
        for mp in MAPS:
            for n_in in (8000, 16000, 32000):
                for conc in (4, 8, 16, 32):
                    rs = [metrics(data.get((s, mp, n_in, hit, conc), {"rows": []})["rows"])
                          for s in SCH]
                    if any(r is None for r in rs):
                        continue
                    cols = []
                    for k in keys:
                        for r in rs:
                            cols.append(fmt(r, [k]))
                    print(f"{mp} | {n_in // 1000}k | {conc} | " + " | ".join(cols))
        sys.exit(0)
    print("map | hit | sch | " + " | ".join(keys))
    for mp in MAPS:
        for hit in (80, 98, 100):
            for s in SCH:
                rows = []
                for key, v in data.items():
                    if key[0] == s and key[1] == mp and key[3] == hit:
                        rows += v["rows"]
                r = metrics(rows)
                print(f"{mp} | {hit} | {s} | {fmt(r, keys)}")
