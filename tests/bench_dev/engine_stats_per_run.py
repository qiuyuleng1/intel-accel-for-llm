"""vLLM engine stats per run of one test-qwen38-27B.sh result directory.

For every run (one `vllm bench serve`) of one config: the run's time window is
[json write time - duration_s, json write time] (rsync -a keeps the write time), and
the serve.log lines vLLM prints every 10 s inside it
  Engine 000: Avg prompt throughput: P tokens/s, Avg generation throughput: G tokens/s,
  Running: R reqs, Waiting: W reqs[, Deferred: D reqs], GPU KV cache usage: K%, ...
are averaged, leaving out the first 3 and last 2 lines (ramp-up and tail). Waiting
includes Deferred (requests the scheduler skipped, e.g. still loading KV from a
connector). From summary.csv: request_throughput (requests/s), TTFT and TPOT means.

Columns printed:
  in_engine = Running + Waiting; outside = concurrency - in_engine, the requests the
  client has sent that are not yet in the engine;
  in_tok/s = request_throughput x input length (prompt tokens entering per second);
  little_s = concurrency / request_throughput (Little's law time in flight) next to
  e2e_s = TTFT + 127 x TPOT (measured time in flight).

Usage: python3 tests/bench_dev/engine_stats_per_run.py bench-results/<run dir> <config>
"""

import csv
import datetime
import os
import re
import sys
from dataclasses import dataclass

ENGINE_LINE = re.compile(
    r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine 000: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?.*?"
    r"GPU KV cache usage: ([\d.]+)%"
)
YEAR = 2026
OUTPUT_LEN = 128
SKIP_HEAD = 3
SKIP_TAIL = 2


@dataclass
class EngineSample:
    """One 10-second engine stats line."""
    time: float
    prompt_tokens_per_s: float
    generation_tokens_per_s: float
    running: int
    waiting: int
    deferred: int
    kv_usage_pct: float


def read_samples(serve_log_path):
    with open(serve_log_path, "rb") as log_file:
        text = log_file.read().decode("utf-8", errors="replace")
    samples = []
    for line in text.replace("\r", "\n").split("\n"):
        match = ENGINE_LINE.search(line)
        if not match:
            continue
        when = datetime.datetime(
            YEAR, int(match.group(1)), int(match.group(2)),
            int(match.group(3)), int(match.group(4)), int(match.group(5)),
        )
        deferred = 0
        if match.group(10) is not None:
            deferred = int(match.group(10))
        samples.append(
            EngineSample(
                time=when.timestamp(),
                prompt_tokens_per_s=float(match.group(6)),
                generation_tokens_per_s=float(match.group(7)),
                running=int(match.group(8)),
                waiting=int(match.group(9)),
                deferred=deferred,
                kv_usage_pct=float(match.group(11)),
            )
        )
    return samples


def average(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def main():
    result_dir = sys.argv[1]
    config = sys.argv[2]
    samples = read_samples(f"{result_dir}/{config}/serve.log")
    print(
        "input hit con | ttft_ms tpot_ms e2e_s little_s | req/s in_tok/s | "
        "running waiting deferred in_engine outside | prompt_tok/s gen_tok/s kv% | samples"
    )
    with open(f"{result_dir}/summary.csv") as summary_file:
        for row in csv.DictReader(summary_file):
            if row["config"] != config:
                continue
            input_len = int(row["input_len_tokens"])
            concurrency = int(row["max_concurrency"])
            json_path = (
                f"{result_dir}/{config}/in{input_len}/{config}_tp4_in{input_len}"
                f"_con{concurrency}_hit{row['hit_rate_pct']}.json"
            )
            end = os.path.getmtime(json_path)
            start = end - float(row["duration_s"])
            window = []
            for sample in samples:
                if start <= sample.time <= end:
                    window.append(sample)
            steady = window[SKIP_HEAD:len(window) - SKIP_TAIL]
            running = []
            waiting = []
            deferred = []
            prompt = []
            generation = []
            kv_usage = []
            for sample in steady:
                running.append(sample.running)
                waiting.append(sample.waiting)
                deferred.append(sample.deferred)
                prompt.append(sample.prompt_tokens_per_s)
                generation.append(sample.generation_tokens_per_s)
                kv_usage.append(sample.kv_usage_pct)
            requests_per_s = float(row["request_throughput_req_s"])
            ttft_ms = float(row["mean_ttft_ms"])
            tpot_ms = float(row["mean_tpot_ms"])
            in_engine = average(running) + average(waiting)
            print(
                f"{input_len} {row['hit_rate_pct']} {concurrency} | "
                f"{ttft_ms:.0f} {tpot_ms:.1f} {(ttft_ms + (OUTPUT_LEN - 1) * tpot_ms) / 1000:.1f} "
                f"{concurrency / requests_per_s:.1f} | {requests_per_s:.2f} {requests_per_s * input_len:.0f} | "
                f"{average(running):.0f} {average(waiting):.0f} {average(deferred):.0f} "
                f"{in_engine:.0f} {concurrency - in_engine:.0f} | "
                f"{average(prompt):.0f} {average(generation):.0f} {average(kv_usage):.1f} | {len(steady)}"
            )


if __name__ == "__main__":
    main()
