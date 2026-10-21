# Usage: put_vs_unzip.py <trace.json.gz> <min_nreq> [rank]
# For store steps with >= min_nreq stored requests, compare per-layer forward progress
# (first put at layer L) against OMP unzip completion of layer L (loads in the same window).
import gzip, json, re, statistics, sys
d = json.load(gzip.open(sys.argv[1]))
min_nreq = int(sys.argv[2])
rank = sys.argv[3] if len(sys.argv) > 3 else "0"
ev = d["workers"][rank]["events"]
lay = re.compile(r"model\.layers\.(\d+)\.")
zips = []
unz = []
for e in ev:
    m = lay.search(e["label"])
    if not m:
        continue
    if e["label"].startswith("zip|"):
        zips.append((e["enqueue_ns"], int(m.group(1))))
    elif e["label"].startswith("unzip|"):
        unz.append((e["enqueue_ns"], e["start_ns"], e["end_ns"], int(m.group(1))))
zips.sort()
steps = []
cur = []
prev = -1
for t, layer in zips:
    if cur and layer < prev:
        steps.append(cur)
        cur = []
    cur.append((t, layer))
    prev = layer
steps.append(cur)
for s in steps:
    first_put = {}
    nreq = 0
    for t, layer in s:
        if layer not in first_put:
            first_put[layer] = t
        if layer == 0:
            nreq += 1
    if nreq < min_nreq or len(first_put) < 80:
        continue
    t0 = s[0][0]
    t_end = s[-1][0]
    window = []
    for u in unz:
        if t0 - 2_000_000_000 <= u[0] <= t_end:
            window.append(u)
    last_end = {}
    busy = 0
    for enq, st, en, layer in window:
        last_end[layer] = max(last_end.get(layer, 0), en)
        busy += min(en, t_end) - max(st, t0) if en > t0 and st < t_end else 0
    print(f"step nreq={nreq} span={(t_end - t0) / 1e6:.0f} ms  unzip tasks={len(window)} "
          f"unzip busy inside step={busy / 1e6:.0f} ms")
    print("   layer  put_ms  unzip_end_ms  put-unzip_ms")
    for layer in (0, 8, 9, 20, 40, 60, 79):
        pe = (first_put[layer] - t0) / 1e6
        ue = (last_end[layer] - t0) / 1e6 if layer in last_end else float("nan")
        print(f"   {layer:5d} {pe:7.1f} {ue:12.1f} {pe - ue:12.1f}")
    break_after = True
