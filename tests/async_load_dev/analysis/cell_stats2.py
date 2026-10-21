# Usage: cell_stats2.py <run_dir> <input_len> <hit> <conc>
# Log-only analysis of one cell for every config: bench duration / TTFT / TPOT (json),
# vLLM 10 s stats inside the cell window (prompt + generation throughput, running,
# waiting) and connector decisions (async share, selected layers).
import datetime
import glob
import json
import os
import re
import statistics
import sys

run_dir, inp, hit, conc = sys.argv[1:5]
stat_re = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs"
)
conn_re = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*get_num_new_matched_tokens, req-(\S+), "
    r"externally-cached tokens: (\d+), .*async=(\w+), selected_load_layers=(-?\d+)"
)
short = {
    "priority_3level_first_n_per_req_promote_raise": "raise",
    "priority_3level_first_n_per_req": "prio",
    "batch_reqs_async_load_submit": "batch",
    "sync": "sync",
}
maps = {"L0-8-16": "A", "L0-16-32": "B", "L0-40-79": "C"}
print(f"cell {inp} h{hit} c{conc}")
print("cfg     dur_s  meanTTFT  p95TTFT  TPOT  prompt_tok/s  gen_tok/s  running  waiting  async%  layers")
for path in sorted(glob.glob(f"{run_dir}/*_in{inp}_hr{hit}_c{conc}.json")):
    label = os.path.basename(path).split(f"_in{inp}")[0]
    scheme = label
    tag = ""
    for key, value in maps.items():
        if label.endswith("_" + key):
            scheme = label[: -len(key) - 1]
            tag = value
    cfg = short[scheme] + tag
    result = json.load(open(path))
    end = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
    start = end - datetime.timedelta(seconds=result["duration"])
    prompt_tp = []
    gen_tp = []
    running = []
    waiting = []
    n_async = 0
    n_req = 0
    layers = {}
    for line in open(f"{run_dir}/{label}.serve.log", errors="replace"):
        m = stat_re.search(line)
        if m:
            t = datetime.datetime.strptime(f"{end.year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
            if start + datetime.timedelta(seconds=10) <= t <= end:
                prompt_tp.append(float(m.group(2)))
                gen_tp.append(float(m.group(3)))
                running.append(int(m.group(4)))
                waiting.append(int(m.group(5)))
            continue
        m = conn_re.search(line)
        if m:
            t = datetime.datetime.strptime(f"{end.year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
            if start <= t <= end and int(m.group(3)) > 0:
                n_req += 1
                if m.group(4) == "True":
                    n_async += 1
                    layers[m.group(5)] = layers.get(m.group(5), 0) + 1
    print(
        f"{cfg:7s} {result['duration']:6.0f} {result['mean_ttft_ms']:9.0f} {result['p95_ttft_ms']:8.0f} "
        f"{result['mean_tpot_ms']:5.0f} {statistics.mean(prompt_tp):13.0f} {statistics.mean(gen_tp):10.0f} "
        f"{statistics.mean(running):8.1f} {statistics.mean(waiting):8.1f} "
        f"{100 * n_async / max(n_req, 1):6.0f}  {layers}"
    )
