"""Per-variant summary of one test-qwen38-27B-frontend.sh result directory.

For every variant directory (rust, python, ...) with one `vllm bench serve` run
(vllm-prefix/in<len>/*.json), the run's time window is [json write time -
duration_s, json write time]. Inside it:
  - engine lines of serve.log (every 10 s; Python "Engine 000: ... Running: R
    reqs, Waiting: W reqs", Rust "... Reqs Running: R, Waiting: W"), averaged
    without leaving any out (a run is only ~4-7 lines long);
  - cpu_threads.csv (tests/bench_dev/cpu_thread_sampler.py, one sample per
    second): per process label the mean and max of its per-second CPU %
    (100 = one core), and the busiest threads by mean CPU % over the window.
From summary.csv: request rate, TTFT, TPOT and the GPU prefix cache counters,
which count every request the engine got in the run: num_prompts plus the bench
client's 1 test request and NUM_WARMUPS warm-up requests.

Usage: python3 tests/bench_dev/frontend_run_summary.py <result dir> [variant ...]
"""

import csv
import datetime
import glob
import os
import re
import sys
from dataclasses import dataclass

ENGINE_LINES = (
    re.compile(
        r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine 000: .*?"
        r"Running: (\d+) reqs, Waiting: (\d+) reqs"
    ),
    re.compile(
        r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[log_stats\.rs:\d+\] .*?"
        r"Reqs Running: (\d+), Waiting: (\d+)"
    ),
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
YEAR = 2026
EXTRA_REQUESTS = 1 + 5
DEFAULT_VARIANTS = ("python", "python-workers8", "rust", "rust-threads8")
LABELS = ("api_server", "rust_frontend", "engine_core", "bench_client")
TOP_THREADS = 4


@dataclass
class EngineSample:
    """One engine stats line of serve.log."""
    time: float
    running: int
    waiting: int


def read_engine_samples(serve_log_path):
    with open(serve_log_path, "rb") as log_file:
        text = log_file.read().decode("utf-8", errors="replace")
    # vllm-rs colors its log fields with ANSI escape sequences.
    text = ANSI_ESCAPE.sub("", text)
    samples = []
    for line in text.replace("\r", "\n").split("\n"):
        for pattern in ENGINE_LINES:
            match = pattern.search(line)
            if not match:
                continue
            when = datetime.datetime(
                YEAR, int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4)), int(match.group(5)),
                tzinfo=datetime.timezone.utc,
            )
            samples.append(EngineSample(when.timestamp(), int(match.group(6)), int(match.group(7))))
    return samples


def average(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def print_cpu(cpu_path, start, end):
    """Mean / max CPU % per process label and the busiest threads in [start, end]."""
    seconds = set()
    label_totals = {}
    thread_sums = {}
    with open(cpu_path) as cpu_file:
        for row in csv.DictReader(cpu_file):
            when = float(row["unix_time_s"])
            if when < start or when > end:
                continue
            seconds.add(when)
            label = row["label"]
            cpu_pct = float(row["cpu_pct"])
            if row["tid"] == "total":
                per_second = label_totals.setdefault(label, {})
                per_second[when] = per_second.get(when, 0.0) + cpu_pct
                continue
            name = row["thread_name"]
            # The main thread of a process has tid == pid.
            if row["tid"] == row["pid"]:
                name += " (main)"
            key = (label, row["tid"], name)
            thread_sums[key] = thread_sums.get(key, 0.0) + cpu_pct
    second_count = len(seconds)
    print(f"  CPU % of one core, {second_count} one-second samples:")
    for label in LABELS:
        if label not in label_totals:
            continue
        totals = list(label_totals[label].values())
        # A second without a row for the label means it used < 1 %.
        mean = sum(totals) / second_count
        threads = []
        for (thread_label, tid, name), cpu_sum in thread_sums.items():
            if thread_label == label:
                threads.append((cpu_sum / second_count, tid, name))
        threads.sort(reverse=True)
        busy_threads = 0
        for thread_mean, _, _ in threads:
            if thread_mean >= 10:
                busy_threads += 1
        top = []
        for thread_mean, tid, name in threads[:TOP_THREADS]:
            top.append(f"{name}/{tid} {thread_mean:.0f}")
        print(
            f"    {label:13s} mean {mean:5.0f} max {max(totals):5.0f} | "
            f"threads >= 10%: {busy_threads:2d} | top: {', '.join(top)}"
        )


def main():
    result_dir = sys.argv[1]
    variants = sys.argv[2:]
    if not variants:
        variants = DEFAULT_VARIANTS
    for variant in variants:
        variant_dir = f"{result_dir}/{variant}"
        with open(f"{variant_dir}/summary.csv") as summary_file:
            row = next(csv.DictReader(summary_file))
        input_len = int(row["input_len_tokens"])
        concurrency = int(row["max_concurrency"])
        json_paths = glob.glob(f"{variant_dir}/vllm-prefix/in{input_len}/*.json")
        end = os.path.getmtime(json_paths[0])
        start = end - float(row["duration_s"])

        engine_requests = int(row["num_prompts"]) + EXTRA_REQUESTS
        query_tokens = int(row["gpu_prefix_cache_query_tokens"])
        hit_tokens = int(row["gpu_prefix_cache_hit_tokens"])
        running = []
        waiting = []
        for sample in read_engine_samples(f"{variant_dir}/vllm-prefix/serve.log"):
            if start <= sample.time <= end:
                running.append(sample.running)
                waiting.append(sample.waiting)
        in_engine = average(running) + average(waiting)

        print(f"== {variant}: input {input_len}, concurrency {concurrency}, {row['num_prompts']} requests")
        print(
            f"  duration {float(row['duration_s']):.1f} s, {float(row['request_throughput_req_s']):.2f} req/s, "
            f"TTFT mean/median/p95 {float(row['mean_ttft_ms']):.0f}/{float(row['median_ttft_ms']):.0f}/"
            f"{float(row['p95_ttft_ms']):.0f} ms, TPOT mean/median/p95 {float(row['mean_tpot_ms']):.1f}/"
            f"{float(row['median_tpot_ms']):.1f}/{float(row['p95_tpot_ms']):.1f} ms"
        )
        print(
            f"  engine per request ({engine_requests} requests): prompt tokens {query_tokens / engine_requests:.0f}, "
            f"prefix-cache hit {hit_tokens / engine_requests:.0f}, computed {(query_tokens - hit_tokens) / engine_requests:.0f}"
        )
        print(
            f"  engine lines ({len(running)}): running {average(running):.0f}, waiting {average(waiting):.0f}, "
            f"outside the engine {concurrency - in_engine:.0f}"
        )
        print_cpu(f"{variant_dir}/cpu_threads.csv", start, end)


if __name__ == "__main__":
    main()
