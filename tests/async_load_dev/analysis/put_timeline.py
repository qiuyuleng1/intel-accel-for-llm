# Usage: put_timeline.py <trace.json.gz> [rank]
# Rebuilds the forward-thread put timeline from zip-task enqueue_ns (one zip per put per layer).
import gzip, json, re, statistics, sys
d = json.load(gzip.open(sys.argv[1]))
rank = sys.argv[2] if len(sys.argv) > 2 else "0"
ev = d["workers"][rank]["events"]
pat = re.compile(r"^zip\|.*model\.layers\.(\d+)\.")
zips = []
for e in ev:
    m = pat.match(e["label"])
    if m:
        zips.append((e["enqueue_ns"], int(m.group(1))))
zips.sort()
steps = []
cur = []
prev_layer = -1
for t, layer in zips:
    if cur and layer < prev_layer:
        steps.append(cur)
        cur = []
    cur.append((t, layer))
    prev_layer = layer
if cur:
    steps.append(cur)
rows = []
for s in steps:
    layers = {}
    for t, layer in s:
        layers.setdefault(layer, []).append(t)
    if len(layers) < 80:
        continue
    nreq = len(layers[0])
    intra = []
    inter = []
    for layer in range(80):
        ts = layers[layer]
        for a, b in zip(ts, ts[1:]):
            intra.append((b - a) / 1e6)
        if layer + 1 in layers:
            inter.append((layers[layer + 1][0] - ts[-1]) / 1e6)
    span = (s[-1][0] - s[0][0]) / 1e6
    rows.append((nreq, span, statistics.mean(intra) if intra else 0.0, statistics.mean(inter), sum(intra), sum(inter)))
print(f"store steps={len(rows)}")
print("nreq  span_ms  intra_put_ms  inter_layer_ms  sum_intra_ms  sum_inter_ms")
by = {}
for r in rows:
    by.setdefault(r[0], []).append(r)
for n in sorted(by):
    rs = by[n]
    print(f"{n:4d} x{len(rs):2d}  {statistics.mean(r[1] for r in rs):7.1f}  {statistics.mean(r[2] for r in rs):11.3f}  "
          f"{statistics.mean(r[3] for r in rs):13.3f}  {statistics.mean(r[4] for r in rs):11.1f}  {statistics.mean(r[5] for r in rs):11.1f}")
