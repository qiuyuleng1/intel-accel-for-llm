# SPDX-License-Identifier: Apache-2.0
"""Print vLLM's Running / Waiting / Deferred request counts during every run.

Usage: python3 tests/bench_dev/running-reqs-per-run.py <results_dir> <config>

vLLM's API server logs one stats line every VLLM_LOG_STATS_INTERVAL seconds
(default 10): "Running: N reqs" are the requests in the running queue
(prefilling or decoding) at the last scheduler step before the line,
"Waiting: N reqs" the requests not yet scheduled, and "Deferred: N reqs" the
part of Waiting that the scheduler skipped (e.g. still loading external KV).
A run's window is [end - duration, end] from its `vllm bench serve` result
JSON, so warm-ups and dataset generation are left out.
"""

import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

STATS_PATTERN = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine \d+: "
    r"Avg prompt throughput: [\d.]+ tokens/s, "
    r"Avg generation throughput: [\d.]+ tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?"
)


@dataclass
class StatsLine:
    """One vLLM stats line; the log has no year, so it is kept as text."""
    month_day_time: str
    running: int
    waiting: int
    deferred: int


def read_stats_lines(serve_log: Path) -> list[StatsLine]:
    text = serve_log.read_text(errors="replace").replace("\r", "\n")
    stats_lines = []
    for line in text.split("\n"):
        match = STATS_PATTERN.search(line)
        if match is None:
            continue
        deferred = 0
        # vLLM omits "Deferred" when no waiting request was skipped.
        if match.group(4) is not None:
            deferred = int(match.group(4))
        stats_lines.append(
            StatsLine(
                month_day_time=match.group(1),
                running=int(match.group(2)),
                waiting=int(match.group(3)),
                deferred=deferred,
            )
        )
    return stats_lines


def describe(label: str, values: list[int]) -> str:
    return (
        f"{label} mean {statistics.mean(values):.1f} "
        f"median {statistics.median(values):.1f} "
        f"min {min(values)} max {max(values)}"
    )


def main() -> None:
    results_dir = Path(sys.argv[1])
    config = sys.argv[2]
    stats_lines = read_stats_lines(results_dir / config / "serve.log")

    with open(results_dir / "summary.csv") as summary_file:
        for run in csv.DictReader(summary_file):
            if run["config"] != config:
                continue
            tag = (
                f"{config}_tp{run['tp_size']}_in{run['input_len_tokens']}"
                f"_con{run['max_concurrency']}_hit{run['hit_rate_pct']}"
            )
            result_json = results_dir / config / f"in{run['input_len_tokens']}" / f"{tag}.json"
            with open(result_json) as result_file:
                result = json.load(result_file)
            run_end = datetime.strptime(result["date"], "%Y%m%d-%H%M%S")
            run_start = run_end - timedelta(seconds=result["duration"])

            running = []
            waiting = []
            deferred = []
            for stats_line in stats_lines:
                logged_at = datetime.strptime(
                    f"{run_end.year}-{stats_line.month_day_time}", "%Y-%m-%d %H:%M:%S"
                )
                if run_start <= logged_at <= run_end:
                    running.append(stats_line.running)
                    waiting.append(stats_line.waiting)
                    deferred.append(stats_line.deferred)

            print(f"{tag}: {len(running)} stats lines")
            if len(running) == 0:
                continue
            print(f"    {describe('Running', running)}")
            print(f"    {describe('Waiting', waiting)}")
            print(f"    {describe('Deferred', deferred)}")


if __name__ == "__main__":
    main()
