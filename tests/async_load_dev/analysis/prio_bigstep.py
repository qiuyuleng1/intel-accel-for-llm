# Usage: prio_bigstep.py <trace.json.gz> <nreq> [rank]
# For the first store step with <nreq> stored requests: put time per layer 0..12 and
# OMP activity in 50 ms buckets (unzip by priority, zip), relative to the layer-0 put.
import gzip, json, re, sys
d = json.load(gzip.open(sys.argv[1]))
want = int(sys.argv[2])
rank = sys.argv[3] if len(sys.argv) > 3 else "0"
ev = d["workers"][rank]["events"]
lay = re.compile(r"model\.layers\.(\d+)\.")
zips = []
for e in ev:
    m = lay.search(e["label"])
    if m and e["label"].startswith("zip|"):
        zips.append((e["enqueue_ns"], int(m.group(1))))
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
    if sum(1 for _, l in s if l == 0) == want:
        break
t0 = s[0][0]
first = {}
for t, l in s:
    first.setdefault(l, t)
print("put ms per layer:", " ".join(f"L{l}={(first[l] - t0) / 1e6:.0f}" for l in range(0, 13)), f"L79={(first[79] - t0) / 1e6:.0f}")
B = 50_000_000
print("bucket_ms  P0_unzip  P1_unzip  P2_unzip  zip  idle   (ms busy on OMP-Main per 50 ms)")
for b in range(-4, 16):
    lo = t0 + b * B
    hi = lo + B
    acc = {"u0": 0, "u1": 0, "u2": 0, "zip": 0}
    for e in ev:
        st, en = e["start_ns"], e["end_ns"]
        if en <= lo or st >= hi:
            continue
        ov = min(en, hi) - max(st, lo)
        if e["label"].startswith("zip|"):
            acc["zip"] += ov
        elif e["label"].startswith("unzip|"):
            acc["u" + str(min(e["priority"], 2))] += ov
    busy = sum(acc.values())
    print(f"{b * 50:6d}  {acc['u0'] / 1e6:8.1f}  {acc['u1'] / 1e6:8.1f}  {acc['u2'] / 1e6:8.1f}  {acc['zip'] / 1e6:5.1f}  {(B - busy) / 1e6:5.1f}")
