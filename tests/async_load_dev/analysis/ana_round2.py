import csv, collections, math, statistics, sys
O = sys.argv[1]; P = sys.argv[2]
ab = {"priority_3level_first_n_per_req": "prio", "priority_3level_first_n_per_req_promote_raise": "raise",
      "batch_reqs_async_load_submit": "batch", "sync": "sync"}
mp = {"0-3:0,4-6:8,7-:16": "A", "0-3:0,4-6:16,7-:32": "B", "0-3:0,4-6:40,7-:79": "C", "none": "", "0-3:0,4-6:4,7-:8": "Z"}
def load(path):
    cell = collections.defaultdict(dict)
    for r in csv.DictReader(open(path + "/summary.csv")):
        k = (int(r["input_len"]), int(r["hit_rate"]), int(r["concurrency"]))
        cell[k][ab[r["scheme"]] + mp[r["dynamic_map"]]] = (float(r["mean_ttft_ms"]), float(r["p95_ttft_ms"]), float(r["mean_tpot_ms"]))
    return cell
cell = load(O)
print("c4 absolute mean TTFT ms:")
for k in sorted(cell):
    if k[2] == 4:
        print(k, " ".join(f"{c}={v[0]:.0f}" for c, v in sorted(cell[k].items())))
cfgs = ["prioA", "prioB", "prioC", "raiseA", "raiseB", "raiseC", "batchA", "batchB", "batchC"]
print("\nper config over 36 cells: geomean ratio vs sync (mean TTFT), #better than sync, worst, p95 geomean, TPOT geomean")
for c in cfgs:
    lr = []; lp = []; lt = []; win = 0; worst = (-9, None)
    for k in cell:
        r = cell[k][c][0] / cell[k]["sync"][0]
        lr.append(math.log(r)); lp.append(math.log(cell[k][c][1] / cell[k]["sync"][1])); lt.append(math.log(cell[k][c][2] / cell[k]["sync"][2]))
        win += r < 1
        if r - 1 > worst[0]: worst = (r - 1, k)
    g = lambda l: (math.exp(statistics.mean(l)) - 1) * 100
    print(f"{c:7s} ttft {g(lr):+5.1f}%  better {win:2d}/36  worst {worst[0]*100:+4.0f}% {worst[1]}  p95 {g(lp):+5.1f}%  tpot {g(lt):+5.1f}%")
print("\nby hit rate (geomean TTFT vs sync, all lens/concs):")
for h in (80, 95, 98, 100):
    out = []
    for c in cfgs:
        l = [math.log(cell[k][c][0] / cell[k]["sync"][0]) for k in cell if k[1] == h]
        out.append(f"{c}={(math.exp(statistics.mean(l)) - 1) * 100:+.0f}")
    print(h, " ".join(out))
print("\nrepro vs previous round (map 0-3:0,4-6:8,7-:16 = prev 'B'), same cells, ratio new/prev mean TTFT:")
prev = load(P)
for c, pc in (("prioA", "prioA"), ("raiseA", "raiseA"), ("batchA", "batchA"), ("sync", "sync")):
    rs = []
    for k in cell:
        if k in prev and pc in prev[k]:
            rs.append(cell[k][c][0] / prev[k][pc][0])
    print(c, len(rs), "median", round(statistics.median(rs), 3), "min", round(min(rs), 2), "max", round(max(rs), 2))
