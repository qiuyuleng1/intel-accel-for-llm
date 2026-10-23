# Usage: python3 tests/async_load_dev/analysis/round1022.py
# 10-22 round (20261022_101029) and the fixed-layer vs map comparison (doc 11.16).
# A: per scheme x map ratios vs sync (geomean over cells), by hit rate, paired
#    scheme comparisons, cross-run noise against 20261020_065412.
# B: priority_3level_first_n_per_req at 32k c4 / c8: fixed N (20261021_054543)
#    vs maps (20261020_065412, 20261021_054543, 20261022_101029), each run
#    normalized by its own sync.
import collections
import csv
import math
import statistics

R20 = "_data/perf-qwen72b/20261020_065412"
R21 = "_data/perf-qwen72b/20261021_054543"
R22 = "_data/perf-qwen72b/20261022_101029"
SHORT = {
    "sync": "sync",
    "priority_3level_first_n_per_req": "prio",
    "priority_3level_first_n_per_req_promote_raise": "raise",
    "priority_3level_first_n_per_req_remaining_by_layer": "rbl_pr",
    "priority_3level_first_n_batch_reqs_remaining_by_layer": "rbl_br",
    "batch_reqs_async_load_submit": "batch",
}
MAPS = {
    "0-3:0,4-6:4,7-:8": "4/8",
    "0-3:0,4-6:8,7-:16": "8/16",
    "0-3:0,4-6:16,7-:32": "16/32",
    "0-3:0,4-6:40,7-:79": "40/79",
}
METRICS = ("ttft", "p95", "tpot", "e2e", "dur")


def load(run_dir):
    """(cfg, input_len, hit, conc) -> metric -> value (mean over repeats)."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{run_dir}/summary.csv")):
        scheme = SHORT[r["scheme"]]
        if scheme == "sync":
            cfg = "sync"
        elif r["dynamic_map"] != "none":
            cfg = f"{scheme} {MAPS[r['dynamic_map']]}"
        else:
            n = "all" if r["fixed_layers"] == "-1" else r["fixed_layers"]
            cfg = f"{scheme} N{n}"
        key = (cfg, int(r["input_len"]), int(r["hit_rate"]), int(r["concurrency"]))
        ttft = float(r["mean_ttft_ms"])
        tpot = float(r["mean_tpot_ms"])
        acc[key]["ttft"].append(ttft)
        acc[key]["p95"].append(float(r["p95_ttft_ms"]))
        acc[key]["tpot"].append(tpot)
        acc[key]["e2e"].append(ttft + 127 * tpot)
        acc[key]["dur"].append(float(r["duration"]))
    out = {}
    for key, m in acc.items():
        out[key] = {k: statistics.mean(v) for k, v in m.items()}
    return out


def gm(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) - 1


def pct(x):
    return f"{x * 100:+.0f}%"


def cells_of(data, cfg, flt=None):
    out = []
    for key in data:
        if key[0] == cfg and (flt is None or flt(key)):
            out.append(key[1:])
    return sorted(out)


def ratios(data, cfg, ref, metric, flt=None):
    out = []
    for cell in cells_of(data, cfg, flt):
        if (ref,) + cell in data:
            out.append(data[(cfg,) + cell][metric] / data[(ref,) + cell][metric])
    return out


def part_a():
    d = load(R22)
    cfgs = sorted({k[0] for k in d if k[0] != "sync"}, key=lambda c: (c.split()[1], c))
    print("== A1 10-22: each config vs sync, geomean over its 27 cells")
    print("cfg | TTFT | p95 TTFT | TPOT | e2e | duration | TTFT better cells | worst TTFT cell")
    for cfg in cfgs:
        cols = []
        for m in METRICS:
            cols.append(pct(gm(ratios(d, cfg, "sync", m))))
        worst = None
        better = 0
        total = 0
        for cell in cells_of(d, cfg):
            r = d[(cfg,) + cell]["ttft"] / d[("sync",) + cell]["ttft"]
            total += 1
            better += r < 1
            if worst is None or r > worst[0]:
                worst = (r, cell)
        print(f"{cfg} | {' | '.join(cols)} | {better}/{total} | "
              f"{pct(worst[0] - 1)} {worst[1]}")
    print("\n== A2 10-22: TTFT / e2e vs sync by hit rate (geomean over 9 cells)")
    print("cfg | hit 80 | hit 98 | hit 100")
    for cfg in cfgs:
        cols = []
        for hit in (80, 98, 100):
            f = lambda k, h=hit: k[2] == h
            cols.append(f"{pct(gm(ratios(d, cfg, 'sync', 'ttft', f)))} / "
                        f"{pct(gm(ratios(d, cfg, 'sync', 'e2e', f)))}")
        print(f"{cfg} | {' | '.join(cols)}")
    print("\n== A2b 10-22: TTFT / e2e vs sync by input length")
    print("cfg | 8k | 16k | 32k")
    for cfg in cfgs:
        cols = []
        for n in (8000, 16000, 32000):
            f = lambda k, n=n: k[1] == n
            cols.append(f"{pct(gm(ratios(d, cfg, 'sync', 'ttft', f)))} / "
                        f"{pct(gm(ratios(d, cfg, 'sync', 'e2e', f)))}")
        print(f"{cfg} | {' | '.join(cols)}")
    print("\n== A3 10-22: paired scheme comparison per map (a / b per cell)")
    print("pair | map | TTFT geomean | a better cells | min..max | e2e geomean | TPOT geomean")
    pairs = (("rbl_pr", "prio"), ("rbl_br", "prio"), ("rbl_br", "rbl_pr"),
             ("batch", "prio"), ("rbl_br", "batch"))
    for a, b in pairs:
        all_t = []
        for mp in ("4/8", "8/16", "16/32"):
            t = ratios(d, f"{a} {mp}", f"{b} {mp}", "ttft")
            e = ratios(d, f"{a} {mp}", f"{b} {mp}", "e2e")
            p = ratios(d, f"{a} {mp}", f"{b} {mp}", "tpot")
            all_t += t
            print(f"{a}/{b} | {mp} | {pct(gm(t))} | {sum(1 for x in t if x < 1)}/{len(t)} | "
                  f"{min(t):.2f}..{max(t):.2f} | {pct(gm(e))} | {pct(gm(p))}")
        print(f"{a}/{b} | all | {pct(gm(all_t))} | {sum(1 for x in all_t if x < 1)}/{len(all_t)} | "
              f"{min(all_t):.2f}..{max(all_t):.2f} | |")
    print("\n== A4 noise: same config and cell, 10-22 / 10-20 (mean TTFT)")
    d20 = load(R20)
    print("cfg | cells | median | min..max | within +-10% | within +-20%")
    every = []
    for cfg in ("sync", "prio 8/16", "prio 16/32", "batch 8/16", "batch 16/32"):
        rs = []
        for cell in cells_of(d, cfg):
            if (cfg,) + cell in d20:
                rs.append(d[(cfg,) + cell]["ttft"] / d20[(cfg,) + cell]["ttft"])
        every += rs
        print(f"{cfg} | {len(rs)} | {statistics.median(rs):.2f} | {min(rs):.2f}..{max(rs):.2f} | "
              f"{sum(1 for x in rs if abs(x - 1) <= 0.1)} | {sum(1 for x in rs if abs(x - 1) <= 0.2)}")
    print(f"all | {len(every)} | {statistics.median(every):.2f} | {min(every):.2f}..{max(every):.2f} | "
          f"{sum(1 for x in every if abs(x - 1) <= 0.1)} | {sum(1 for x in every if abs(x - 1) <= 0.2)}")
    print("ratio-vs-sync drift, same config: (10-22 cfg/sync) / (10-20 cfg/sync), TTFT")
    for cfg in ("prio 8/16", "prio 16/32", "batch 8/16", "batch 16/32"):
        rs = []
        for cell in cells_of(d, cfg):
            if (cfg,) + cell in d20:
                r22 = d[(cfg,) + cell]["ttft"] / d[("sync",) + cell]["ttft"]
                r20 = d20[(cfg,) + cell]["ttft"] / d20[("sync",) + cell]["ttft"]
                rs.append(r22 / r20)
        print(f"  {cfg}: geomean {pct(gm(rs))} median {statistics.median(rs):.2f} "
              f"min..max {min(rs):.2f}..{max(rs):.2f}")


def part_b():
    runs = (("10-20", load(R20)), ("10-21", load(R21)), ("10-22", load(R22)))
    print("\n== B prio at 32k c4/c8: each config vs its own run's sync")
    print("run | cfg | cells | TTFT | p95 TTFT | TPOT | e2e | duration | "
          "TTFT h98 | TTFT h100 | e2e h98 | e2e h100 | TTFT c4 | TTFT c8")
    for name, d in runs:
        cfgs = sorted({k[0] for k in d if k[0].startswith("prio ") and k[1] == 32000})
        for cfg in cfgs:
            f = lambda k: k[1] == 32000 and k[2] in (98, 100) and k[3] in (4, 8)
            cols = []
            for m in METRICS:
                cols.append(pct(gm(ratios(d, cfg, "sync", m, f))))
            for m, h in (("ttft", 98), ("ttft", 100), ("e2e", 98), ("e2e", 100)):
                fh = lambda k, h=h: f(k) and k[2] == h
                cols.append(pct(gm(ratios(d, cfg, "sync", m, fh))))
            for c in (4, 8):
                fc = lambda k, c=c: f(k) and k[3] == c
                cols.append(pct(gm(ratios(d, cfg, "sync", "ttft", fc))))
            n = len(ratios(d, cfg, "sync", "ttft", f))
            print(f"{name} | {cfg} | {n} | {' | '.join(cols)}")
    print("\n== B2 hit 95 too (10-20 and 10-21 only), 32k c4/c8, 6 cells")
    print("run | cfg | TTFT | e2e | duration")
    for name, d in runs[:2]:
        cfgs = sorted({k[0] for k in d if k[0].startswith("prio ") and k[1] == 32000})
        for cfg in cfgs:
            f = lambda k: k[1] == 32000 and k[2] in (95, 98, 100) and k[3] in (4, 8)
            print(f"{name} | {cfg} | {pct(gm(ratios(d, cfg, 'sync', 'ttft', f)))} | "
                  f"{pct(gm(ratios(d, cfg, 'sync', 'e2e', f)))} | "
                  f"{pct(gm(ratios(d, cfg, 'sync', 'dur', f)))}")


part_a()
part_b()
