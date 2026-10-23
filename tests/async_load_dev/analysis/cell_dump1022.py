# Usage: python3 tests/async_load_dev/analysis/cell_dump1022.py <label> <input_len> <hit> <conc> [run_dir]
# Dump the rank-0 async_load events of one cell, one line per event, with t in ms
# relative to the cell start. step lines are folded: "S<step> dur=<ms> n=<reqs>
# wait=<ms>" printed at step end.
import datetime
import glob
import os
import re
import sys

label, n_in, hit, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
run_dir = sys.argv[5] if len(sys.argv) > 5 else "_data/perf-qwen72b/20261022_101029"
cell_re = re.compile(r"_in(\d+)_hr(\d+)_c(\d+)\.json$")
ends = []
for js in glob.glob(f"{run_dir}/{label}_in*_hr*_c*.json"):
    m = cell_re.search(js)
    ends.append((os.path.getmtime(js), int(m.group(1)), int(m.group(2)), int(m.group(3))))
ends.sort()
prev = 0.0
prev_len = None
win = None
for end, n, h, c in ends:
    start = prev
    if n != prev_len:
        start = max(start, os.path.getmtime(f"{run_dir}/{label}.in{n}.prewarm.log"))
    if (n, h, c) == (n_in, hit, conc):
        win = (start, end)
    prev, prev_len = end, n


def ts(s):
    return datetime.datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


line_re = re.compile(
    r"^\(Worker_TP0 pid=\d+\) DEBUG (\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*async_load step=(\d+) "
    r"t=(\d+) (.*)$"
)
short = {}


def sid(rid):
    if rid not in short:
        short[rid] = f"r{len(short)}"
    return short[rid]


def shorten(s):
    return re.sub(r"cmpl-bench-[0-9a-f]+-(\d+)-0-[0-9a-f]+", lambda m: sid(m.group(0)), s)


t0 = None
step_t = {}
with open(f"{run_dir}/{label}.serve.log", errors="replace") as f:
    for line in f:
        if not line.startswith("(Worker_TP0") or "async_load step=" not in line:
            continue
        m = line_re.search(line)
        if not m:
            continue
        wall = ts(m.group(1))
        if not (win[0] < wall <= win[1]):
            continue
        step = int(m.group(2))
        t = int(m.group(3)) / 1e6
        if t0 is None:
            t0 = t
        rest = m.group(4)
        if rest.startswith("step_start"):
            step_t[step] = (t, rest.split("reqs=")[1])
            continue
        if rest.startswith("step_end"):
            st, nreq = step_t.get(step, (t, "?"))
            w = re.search(r"wait_ms=([\d.]+)", rest).group(1)
            print(f"{t - t0:10.1f} S{step} end dur={t - st:.0f} n={nreq} wait={w}")
            continue
        print(f"{t - t0:10.1f} S{step} {shorten(rest)}")
