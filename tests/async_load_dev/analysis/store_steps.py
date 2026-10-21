# Forward steps with / without KV stores in one traced cell (diagnosis helper).
# usage: python3 store_steps.py <tag>.trace.json.gz <label>.serve.log [rank]
# A step "stores" if a zip task was enqueued during [step_start, step_end]: every
# put() in save_kv_layer() first blocks the CPU on the compute stream
# (xfer_wait_cur_stream(sync_cur_stream=True)), once per layer.
import re
import statistics
import sys

sys.path.insert(0, "tests/async_load_dev")
import trace_timeline as tt  # noqa: E402

STEP_RE = re.compile(r"\(Worker_TP(\d+) .*async_load step=(\d+) t=(\d+) (step_start|step_end|activate)(.*)")
REQS_RE = re.compile(r"reqs=(\S*)")
MS = 1e6

trace_path, log_path = sys.argv[1], sys.argv[2]
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
events = tt.load_trace(trace_path)[rank]
t_min = min(e["enqueue_ns"] for e in events)
t_max = max(e["release_ns"] for e in events)
zip_enqueues = []
for e in events:
    if e["kind"].startswith("zip"):
        zip_enqueues.append((e["enqueue_ns"], tuple(e["reqs"])))
zip_enqueues.sort()

steps = {}
activated = {}
with open(log_path, errors="replace") as f:
    for line in f:
        m = STEP_RE.search(line)
        if not m or int(m.group(1)) != rank:
            continue
        step, t, kind, rest = int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        if t < t_min or t > t_max:
            continue
        if kind == "step_start":
            steps[step] = [t, None]
        elif kind == "step_end" and step in steps:
            steps[step][1] = t
        elif kind == "activate":
            reqs = REQS_RE.search(rest).group(1)
            if reqs:
                activated[step] = len(reqs.split(","))

groups = {"store+act": [], "store": [], "act": [], "plain": []}
fit_points = []
stored_reqs = []
j = 0
for step in sorted(steps):
    s0, s1 = steps[step]
    if s1 is None:
        continue
    while j < len(zip_enqueues) and zip_enqueues[j][0] < s0:
        j += 1
    k = j
    zips = 0
    while k < len(zip_enqueues) and zip_enqueues[k][0] <= s1:
        zips += 1
        k += 1
    store = zips > 0
    act = step in activated
    if store:
        stored_reqs.append(zips)
    key = ("store+act" if act else "store") if store else ("act" if act else "plain")
    groups[key].append((s1 - s0) / MS)
    if store:
        fit_points.append((zips / 80.0, (s1 - s0) / MS))

total = 0.0
for key in groups:
    total += sum(groups[key])
parts = []
for key in ("store+act", "store", "act", "plain"):
    values = groups[key]
    if values:
        parts.append(f"{key}: n={len(values)} mean={statistics.fmean(values):.0f} "
                     f"sum={sum(values) / 1000:.1f}s")
    else:
        parts.append(f"{key}: n=0")
extra = ""
if stored_reqs:
    extra = f" | zip tasks per store step mean={statistics.fmean(stored_reqs):.0f}"
print(f"rank {rank} steps total {total / 1000:.1f}s | " + " | ".join(parts) + extra)
if len(fit_points) >= 3:
    xs = []
    ys = []
    for x, y in fit_points:
        xs.append(x)
        ys.append(y)
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = 0.0
    sxy = 0.0
    for x, y in fit_points:
        sxx += (x - mx) ** 2
        sxy += (x - mx) * (y - my)
    if sxx > 0:
        b = sxy / sxx
        print(f"  store-step duration ~= {my - b * mx:.0f} ms + {b:.0f} ms x stored requests"
              f" (stored requests per step {min(xs):.1f}..{max(xs):.1f})")
