"""Where the time of a request goes, per run of one test-kimi-k3.sh config, from
the logs only (no GPU needed).

For every run (one `vllm bench serve`) of the config in summary.csv:
  window = [json "date" - duration, json "date"]; the json "date" is written by
  the bench client in the same container clock as serve.log, so no time zone
  conversion is needed.
  samples = the serve.log lines vLLM prints every 10 s inside the window:
    Engine 000: Avg prompt throughput: P tokens/s, ..., Running: R reqs,
    Waiting: W reqs[, Deferred: D reqs], ...
  Waiting includes Deferred (requests waiting for the connector's KV load).

Little's law over the window (average count = arrival rate x average time),
arrival rate = completed requests / duration:
  deferred_s = avg D / rate        waiting for the connector's KV load
  queued_s   = avg (W - D) / rate  loaded (or nothing to load), waiting to be scheduled
  running_s  = avg R / rate        scheduled: prefill and decode
  outside_s  = (concurrency - avg (R + W)) / rate
               sent by the client but not yet in the engine (API server)
Every sample is one instant, so a run of 30-50 s gives only 3-5 samples; the
sample count is printed next to the values.

LMCache only, when <config dir>/lmcache_server.log exists: mean of each kind of
line inside the window (LMCache prints one line per TP rank):
  prefetch_ms  "Prefetch request completed ... in X ms"
  retrieve_ms  "Retrieved N tokens in X seconds"
  store_ms     "Stored N of M requested tokens in X seconds"

Usage:
  python3 tests/bench_dev/analysis/kimi_ttft_breakdown.py <summary.csv> <config> <config dir>
  <config> is the summary.csv "config" value (kvshrink, lmcache, ...); <config dir>
  holds serve.log and in<input>/<config>_dp<dp>tp<tp>_in<input>_con<c>_hit<h>.json.
"""

import csv
import datetime
import json
import os
import re
import sys
from dataclasses import dataclass

ENGINE_LINE = re.compile(
    r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine 000: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?"
)
LMCACHE_TIME = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]"
PREFETCH_LINE = re.compile(LMCACHE_TIME + r".*Prefetch request completed .* in ([\d.]+) ms")
RETRIEVE_LINE = re.compile(LMCACHE_TIME + r".*Retrieved \d+ tokens in ([\d.]+) seconds")
STORE_LINE = re.compile(LMCACHE_TIME + r".*Stored \d+ of \d+ requested tokens in ([\d.]+) seconds")
OUTPUT_LEN = 128


@dataclass
class EngineSample:
    """One 10-second engine stats line."""
    time: datetime.datetime
    prompt_tokens_per_s: float
    running: int
    waiting: int
    deferred: int


@dataclass
class TimedValue:
    """One LMCache log line: when it was printed and the duration it reports."""
    time: datetime.datetime
    milliseconds: float


def read_text(path):
    with open(path, "rb") as log_file:
        return log_file.read().decode("utf-8", errors="replace").replace("\r", "\n")


def read_engine_samples(serve_log_path, year):
    samples = []
    for line in read_text(serve_log_path).split("\n"):
        match = ENGINE_LINE.search(line)
        if not match:
            continue
        deferred = 0
        if match.group(10) is not None:
            deferred = int(match.group(10))
        samples.append(
            EngineSample(
                time=datetime.datetime(
                    year, int(match.group(1)), int(match.group(2)),
                    int(match.group(3)), int(match.group(4)), int(match.group(5)),
                ),
                prompt_tokens_per_s=float(match.group(6)),
                running=int(match.group(8)),
                waiting=int(match.group(9)),
                deferred=deferred,
            )
        )
    return samples


def read_lmcache_values(lmcache_log_path, pattern, scale_to_ms):
    values = []
    for line in read_text(lmcache_log_path).split("\n"):
        match = pattern.search(line)
        if not match:
            continue
        values.append(
            TimedValue(
                time=datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"),
                milliseconds=float(match.group(2)) * scale_to_ms,
            )
        )
    return values


def run_json_path(config_dir, config, row):
    input_len = row["input_len_tokens"]
    return (
        f"{config_dir}/in{input_len}/{config}_dp{row['dp_size']}tp{row['tp_size']}"
        f"_in{input_len}_con{row['max_concurrency']}_hit{row['hit_rate_pct']}.json"
    )


def average(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def mean_in_window(timed_values, start, end):
    inside = []
    for value in timed_values:
        if start <= value.time <= end:
            inside.append(value.milliseconds)
    return average(inside)


def main():
    summary_path, config, config_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = []
    with open(summary_path) as summary_file:
        for row in csv.DictReader(summary_file):
            if row["config"] == config:
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows with config={config} in {summary_path}")

    lmcache_log_path = f"{config_dir}/lmcache_server.log"
    has_lmcache_log = os.path.exists(lmcache_log_path)
    prefetch = []
    retrieve = []
    store = []
    if has_lmcache_log:
        prefetch = read_lmcache_values(lmcache_log_path, PREFETCH_LINE, 1.0)
        retrieve = read_lmcache_values(lmcache_log_path, RETRIEVE_LINE, 1000.0)
        store = read_lmcache_values(lmcache_log_path, STORE_LINE, 1000.0)

    # serve.log lines carry no year; the bench json dates do.
    first_json_path = run_json_path(config_dir, config, rows[0])
    with open(first_json_path) as json_file:
        year = int(json.load(json_file)["date"][:4])
    samples = read_engine_samples(f"{config_dir}/serve.log", year)

    header = (
        "input hit con n | ttft_ms e2e_s little_s | deferred_s queued_s running_s outside_s "
        "| prompt_tok/s | samples"
    )
    if has_lmcache_log:
        header += " | prefetch_ms retrieve_ms store_ms"
    print(header)
    for row in rows:
        input_len = int(row["input_len_tokens"])
        concurrency = int(row["max_concurrency"])
        with open(run_json_path(config_dir, config, row)) as json_file:
            bench = json.load(json_file)
        end = datetime.datetime.strptime(bench["date"], "%Y%m%d-%H%M%S")
        duration_s = float(bench["duration"])
        start = end - datetime.timedelta(seconds=duration_s)

        window = []
        for sample in samples:
            if start <= sample.time <= end:
                window.append(sample)
        running = []
        waiting_not_deferred = []
        deferred = []
        in_engine = []
        prompt = []
        for sample in window:
            running.append(sample.running)
            waiting_not_deferred.append(sample.waiting - sample.deferred)
            deferred.append(sample.deferred)
            in_engine.append(sample.running + sample.waiting)
            prompt.append(sample.prompt_tokens_per_s)

        completed = int(bench["completed"])
        rate = completed / duration_s
        ttft_ms = float(row["mean_ttft_ms"])
        tpot_ms = float(row["mean_tpot_ms"])
        line = (
            f"{input_len // 1000}k {row['hit_rate_pct']} {concurrency} {completed} | "
            f"{ttft_ms:.0f} {(ttft_ms + (OUTPUT_LEN - 1) * tpot_ms) / 1000:.2f} "
            f"{concurrency / rate:.2f} | "
            f"{average(deferred) / rate:.2f} {average(waiting_not_deferred) / rate:.2f} "
            f"{average(running) / rate:.2f} {(concurrency - average(in_engine)) / rate:.2f} | "
            f"{average(prompt):.0f} | {len(window)}"
        )
        if has_lmcache_log:
            line += (
                f" | {mean_in_window(prefetch, start, end):.1f} "
                f"{mean_in_window(retrieve, start, end):.1f} "
                f"{mean_in_window(store, start, end):.1f}"
            )
        print(line)


if __name__ == "__main__":
    main()
