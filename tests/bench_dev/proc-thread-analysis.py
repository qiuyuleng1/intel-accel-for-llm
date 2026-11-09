# SPDX-License-Identifier: Apache-2.0
"""Per-thread CPU and run-queue time of the TP workers during each run.

Usage: python3 tests/bench_dev/proc-thread-analysis.py <sampler.csv> <results_root>

<sampler.csv> is tests/bench_dev/proc-thread-sampler.py's output, recorded
while test-qwen3-32B-async-load.sh wrote <results_root> with
KVSHRINK_THREAD_MAP=1 (the serve logs then name each Python thread). For
every run window [end - duration, end] and TP rank it prints, in seconds over
the window:
  main CPU / runq     the worker's main thread (tid == pid): time on a CPU,
                      time runnable but waiting for a core
  python threads      every other Python thread with its CPU time; a Python
                      thread can only run Python code while holding the GIL,
                      so CPU time here bounds how long it can have held it
  native threads      CPU time of all non-Python threads, and the busiest ones
                      by name (comm)
"""

import csv
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# utime/stime in /proc are in clock ticks of 1/100 s on this kernel.
TICKS_PER_SECOND = 100
THREAD_MAP_PATTERN = re.compile(r"thread-map pid=(\d+) threads=(.*)$", re.MULTILINE)
BUSIEST_NATIVE_SHOWN = 4


@dataclass
class ThreadUsage:
    """One thread's usage over a run window."""
    tid: int
    label: str
    cpu_s: float
    runq_s: float


def load_step_trace_analysis():
    # Reuses read_runs() of the sibling script (its file name has dashes).
    path = Path(__file__).with_name("step-trace-analysis.py")
    spec = importlib.util.spec_from_file_location("step_trace_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_thread_names(serve_log: Path) -> dict[int, str]:
    """Linux thread id -> Python thread name, from the thread-map lines."""
    names = {}
    text = serve_log.read_text(errors="replace").replace("\r", "\n")
    for match in THREAD_MAP_PATTERN.finditer(text):
        for entry in match.group(2).split(","):
            name, _, tid = entry.rpartition(":")
            # Thread names may contain ',' or ':' themselves; such pieces
            # carry no thread id.
            if tid.isdigit():
                names[int(tid)] = name
    return names


def read_samples(sampler_csv: Path) -> list[dict[str, str]]:
    with open(sampler_csv) as sampler_file:
        return list(csv.DictReader(sampler_file))


def thread_usage(samples: list[dict[str, str]], rank: int, start: float, end: float,
                 thread_names: dict[int, str]) -> list[ThreadUsage]:
    first: dict[int, dict[str, str]] = {}
    last: dict[int, dict[str, str]] = {}
    for sample in samples:
        if int(sample["rank"]) != rank:
            continue
        epoch = float(sample["epoch"])
        if epoch < start or epoch > end:
            continue
        tid = int(sample["tid"])
        if tid not in first:
            first[tid] = sample
        last[tid] = sample
    usage = []
    for tid, first_sample in first.items():
        last_sample = last[tid]
        cpu_s = (int(last_sample["cpu_ticks"]) - int(first_sample["cpu_ticks"])) / TICKS_PER_SECOND
        runq_s = (int(last_sample["runq_ns"]) - int(first_sample["runq_ns"])) / 1e9
        if tid == int(first_sample["pid"]):
            label = "main"
        elif tid in thread_names:
            label = f"py:{thread_names[tid]}"
        else:
            label = f"native:{first_sample['comm']}"
        usage.append(ThreadUsage(tid=tid, label=label, cpu_s=cpu_s, runq_s=runq_s))
    return usage


def print_rank(rank: int, usage: list[ThreadUsage]) -> None:
    main_cpu = 0.0
    main_runq = 0.0
    python_threads = []
    native_cpu = 0.0
    native_threads = []
    for thread in usage:
        if thread.label == "main":
            main_cpu = thread.cpu_s
            main_runq = thread.runq_s
        elif thread.label.startswith("py:"):
            python_threads.append(thread)
        else:
            native_cpu += thread.cpu_s
            native_threads.append(thread)
    python_parts = []
    for thread in python_threads:
        python_parts.append(f"{thread.label[3:]}={thread.cpu_s:.2f}")
    native_threads.sort(key=lambda thread: thread.cpu_s, reverse=True)
    busiest_parts = []
    for thread in native_threads[:BUSIEST_NATIVE_SHOWN]:
        busiest_parts.append(f"{thread.label[7:]}/{thread.tid}={thread.cpu_s:.2f}")
    print(f"  rank {rank}: main CPU {main_cpu:6.2f} runq {main_runq:6.3f} | "
          f"python threads {' '.join(python_parts)} | "
          f"native threads {len(native_threads)} CPU {native_cpu:7.2f} "
          f"(busiest {' '.join(busiest_parts)})")


def main() -> None:
    sampler_csv = Path(sys.argv[1])
    results_root = Path(sys.argv[2])
    step_trace_analysis = load_step_trace_analysis()
    samples = read_samples(sampler_csv)
    for setting_dir in sorted(results_root.iterdir()):
        if not (setting_dir / "summary.csv").is_file():
            continue
        config, runs = step_trace_analysis.read_runs(setting_dir)
        thread_names = read_thread_names(setting_dir / config / "serve.log")
        print(f"===== {setting_dir.name} ({config}): {len(thread_names)} named Python threads")
        for run in runs:
            print(f"--- {run.tag} ({run.end_epoch - run.start_epoch:.1f} s)")
            for rank in range(8):
                usage = thread_usage(samples, rank, run.start_epoch, run.end_epoch, thread_names)
                if usage:
                    print_rank(rank, usage)


if __name__ == "__main__":
    main()
