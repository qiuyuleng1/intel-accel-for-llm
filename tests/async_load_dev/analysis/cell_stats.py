# Usage: cell_stats.py <OUT> <input> <hit> <conc>
# Per config: E2E = mean TTFT + 127 * mean TPOT, Little's law c / throughput, and the
# vLLM 10 s stats (Running / Waiting / gen tok/s) averaged over the cell window.
import csv, datetime, glob, json, os, re, statistics, sys
O, inp, hit, conc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
stat_re = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")
print("cfg                                   meanTTFT  meanTPOT  E2E_ms   c/X_ms  run_avg  wait_avg  gen_tok/s  samples")
for j in sorted(glob.glob(f"{O}/*_in{inp}_hr{hit}_c{conc}.json")):
    d = json.load(open(j))
    label = os.path.basename(j).split(f"_in{inp}")[0]
    e2e = d["mean_ttft_ms"] + 127 * d["mean_tpot_ms"]
    little = int(conc) / d["request_throughput"] * 1000
    end = datetime.datetime.fromtimestamp(os.stat(j).st_mtime)
    start = end - datetime.timedelta(seconds=d["duration"])
    run = []; wait = []; gen = []
    for line in open(f"{O}/{label}.serve.log", errors="replace"):
        m = stat_re.search(line)
        if not m:
            continue
        t = datetime.datetime.strptime(f"{end.year}-" + m.group(1), "%Y-%m-%d %H:%M:%S")
        if start + datetime.timedelta(seconds=10) <= t <= end:
            gen.append(float(m.group(3))); run.append(int(m.group(4))); wait.append(int(m.group(5)))
    print(f"{label:37s} {d['mean_ttft_ms']:8.0f} {d['mean_tpot_ms']:9.1f} {e2e:8.0f} {little:8.0f} {statistics.mean(run):8.1f} {statistics.mean(wait):9.1f} {statistics.mean(gen):10.1f} {len(run):8d}")
