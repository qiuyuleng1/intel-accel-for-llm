# Usage: sync_step_vs_load.py <trace.json.gz> <serve.log> [rank]
# Sync scheme: for each load group (unzip label request set), find the forward step
# containing its first unzip enqueue and report step duration vs nreq loaded.
import gzip, json, re, statistics, sys
d = json.load(gzip.open(sys.argv[1]))
rank = sys.argv[3] if len(sys.argv) > 3 else "0"
ev = d["workers"][rank]["events"]
pat = re.compile(r"^unzip\|([^|]*)\|model\.layers\.(\d+)\.")
groups = {}
for e in ev:
    m = pat.match(e["label"])
    if m and "bench" in m.group(1):
        groups.setdefault(m.group(1), []).append(e["enqueue_ns"])
step_re = re.compile(r"\(Worker_TP" + rank + r" .*async_load step=(\d+) t=(\d+) (step_start|step_end)")
st = {}
en = {}
with open(sys.argv[2], errors="replace") as f:
    for line in f:
        m = step_re.search(line)
        if m:
            (st if m.group(3) == "step_start" else en)[int(m.group(1))] = int(m.group(2))
steps = []
for k in st:
    if k in en:
        steps.append((st[k], en[k]))
steps.sort()
import bisect
starts = [s for s, _ in steps]
by = {}
for key, enq in groups.items():
    t = min(enq)
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or not (steps[i][0] <= t <= steps[i][1]):
        continue
    by.setdefault(key.count(",") + 1, []).append((steps[i][1] - steps[i][0]) / 1e6)
print("nreq  count  step_ms")
for n in sorted(by):
    print(f"{n:4d} {len(by[n]):6d} {statistics.mean(by[n]):8.1f}")
