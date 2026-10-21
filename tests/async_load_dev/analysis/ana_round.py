import csv, collections, statistics, sys
O = sys.argv[1]
rows = list(csv.DictReader(open(O + "/summary.csv")))
ab = {"priority_3level_first_n_per_req": "prio", "priority_3level_first_n_per_req_promote_raise": "raise",
      "batch_reqs_async_load_submit": "batch", "sync": "sync"}
mp = {"0-3:0,4-6:8,7-:16": "A", "0-3:0,4-6:16,7-:32": "B", "0-3:0,4-6:40,7-:79": "C", "none": ""}
bad = 0
cell = collections.defaultdict(dict)
for r in rows:
    if r["completed"] != r["num_prompts"]:
        bad += 1
        print("INCOMPLETE", r)
    k = (int(r["input_len"]), int(r["hit_rate"]), int(r["concurrency"]))
    cfg = ab[r["scheme"]] + mp[r["dynamic_map"]]
    cell[k][cfg] = (float(r["mean_ttft_ms"]), float(r["p95_ttft_ms"]), float(r["mean_tpot_ms"]))
print("rows", len(rows), "incomplete", bad)
cfgs = ["prioA", "prioB", "prioC", "raiseA", "raiseB", "raiseC", "batchA", "batchB", "batchC"]
print("\nmean TTFT delta vs sync (%), per cell")
print("in    hit conc  sync_ms | " + " ".join(f"{c:>7s}" for c in cfgs) + " | best")
for k in sorted(cell):
    s = cell[k]["sync"][0]
    ds = []
    for c in cfgs:
        ds.append((cell[k][c][0] / s - 1) * 100)
    bi = min(range(len(cfgs)), key=lambda i: ds[i])
    print(f"{k[0]:5d} {k[1]:4d} {k[2]:4d} {s:8.0f} | " + " ".join(f"{d:+7.0f}" for d in ds) + f" | {cfgs[bi]} {ds[bi]:+.0f}")
