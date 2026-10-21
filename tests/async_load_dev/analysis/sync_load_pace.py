# Usage: sync_load_pace.py <trace.json.gz> [rank]
# Sync scheme: one unzip task per layer per step-group (label lists the requests).
# Prints per group: nreq, unzip span (layer0 start -> layer79 end), ms/layer, OMP busy.
import gzip, json, re, statistics, sys
d = json.load(gzip.open(sys.argv[1]))
rank = sys.argv[2] if len(sys.argv) > 2 else "0"
ev = d["workers"][rank]["events"]
pat = re.compile(r"^unzip\|([^|]*)\|model\.layers\.(\d+)\.")
groups = {}
for e in ev:
    m = pat.match(e["label"])
    if m:
        groups.setdefault(m.group(1), []).append((e["start_ns"], e["end_ns"], int(m.group(2))))
by = {}
for key, ts in groups.items():
    nreq = key.count(",") + 1
    if "bench" not in key:
        continue
    s0 = min(t[0] for t in ts)
    e1 = max(t[1] for t in ts)
    busy = sum(t[1] - t[0] for t in ts)
    by.setdefault(nreq, []).append(((e1 - s0) / 1e6, busy / 1e6))
print("nreq  count  unzip_span_ms  ms_per_layer  omp_busy_ms")
for n in sorted(by):
    rs = by[n]
    sp = statistics.mean(r[0] for r in rs)
    print(f"{n:4d} {len(rs):6d} {sp:13.1f} {sp / 80:12.2f} {statistics.mean(r[1] for r in rs):12.1f}")
