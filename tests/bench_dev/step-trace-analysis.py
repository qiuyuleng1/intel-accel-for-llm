# SPDX-License-Identifier: Apache-2.0
"""Summarize kvshrink/step_trace.py lines of a test-qwen3-32B-async-load.sh run.

Usage: python3 tests/bench_dev/step-trace-analysis.py <results_root>

<results_root> holds one directory per setting (summary.csv and
<config>/serve.log). For every `vllm bench serve` run of a setting it keeps
the step-trace lines whose step started inside the run's window
[end - duration, end] (from the run's result JSON, so warm-ups and dataset
generation are left out) and prints:
  1. totals over the run, rank 0: where the wall time went
  2. full-batch steps (at least concurrency - 2 requests), split into steps
     with prefill, decode-only steps with KVShrink load activity, and without
  3. KV block layout of the batch (rank 0 samples every 20 steps)
  4. per rank: main-thread run-queue wait, other threads' CPU, and the GPU
     time of full-batch decode steps
"""

import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

TRACE_PATTERN = re.compile(r"step-trace (rank=\d+ step=\d+ .*)$")
FIELD_PATTERN = re.compile(r"(\w+)=(-?[\d.]+)")
CONNECTOR_FIELDS = [
    "start_load_kv_ms",
    "wait_for_layer_load_ms",
    "wait_for_layer_load_blocked_ms",
    "save_kv_layer_ms",
    "wait_for_save_ms",
    "get_finished_ms",
]
STEP_FIELDS = [
    "period_ms",
    "exec_ms",
    "gpu_ms",
    "main_cpu_ms",
    "main_runq_ms",
    "other_threads_cpu_ms",
    "connector_ms",
]


@dataclass
class RunWindow:
    """One `vllm bench serve` run of a setting."""
    tag: str
    concurrency: int
    start_epoch: float
    end_epoch: float


def read_trace(serve_log: Path) -> list[dict[str, float]]:
    text = serve_log.read_text(errors="replace").replace("\r", "\n")
    steps = []
    for line in text.split("\n"):
        match = TRACE_PATTERN.search(line)
        if match is None:
            continue
        step = {}
        for name, value in FIELD_PATTERN.findall(match.group(1)):
            step[name] = float(value)
        connector_ms = 0.0
        for name in CONNECTOR_FIELDS:
            # The blocked wait is part of wait_for_layer_load_ms already.
            if name != "wait_for_layer_load_blocked_ms":
                connector_ms += step[name]
        step["connector_ms"] = connector_ms
        steps.append(step)
    return steps


def read_runs(setting_dir: Path) -> tuple[str, list[RunWindow]]:
    config = ""
    runs = []
    with open(setting_dir / "summary.csv") as summary_file:
        for row in csv.DictReader(summary_file):
            config = row["config"]
            tag = (
                f"{config}_tp{row['tp_size']}_in{row['input_len_tokens']}"
                f"_con{row['max_concurrency']}_hit{row['hit_rate_pct']}"
            )
            result_json = setting_dir / config / f"in{row['input_len_tokens']}" / f"{tag}.json"
            with open(result_json) as result_file:
                result = json.load(result_file)
            # The result's date is the container's local time, which is UTC
            # like the host this script runs on.
            end = datetime.strptime(result["date"], "%Y%m%d-%H%M%S")
            start = end - timedelta(seconds=result["duration"])
            runs.append(
                RunWindow(
                    tag=tag,
                    concurrency=int(row["max_concurrency"]),
                    start_epoch=start.timestamp(),
                    end_epoch=end.timestamp(),
                )
            )
    return config, runs


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"mean {statistics.mean(values):8.2f}  median {statistics.median(values):8.2f}"
        f"  n={len(values)}"
    )


def print_totals(steps: list[dict[str, float]]) -> None:
    period_total = 0.0
    totals = {}
    for name in CONNECTOR_FIELDS + ["main_cpu_ms", "main_runq_ms", "other_threads_cpu_ms"]:
        totals[name] = 0.0
    for step in steps:
        period_total += step["period_ms"]
        for name in totals:
            totals[name] += step[name]
    print(f"  1. totals, rank 0: {len(steps)} steps, {period_total / 1e3:.2f} s of steps")
    for name in totals:
        share = 100.0 * totals[name] / period_total if period_total > 0 else 0.0
        print(f"       {name:32s} {totals[name] / 1e3:9.3f} s  {share:5.1f}% of step time")


def step_kind(step: dict[str, float]) -> str:
    # A load submitted in a step runs beside that step's forward, so a step
    # counts as loading if it started a load, found one unfinished, or ran a
    # promoted request whose remaining layers were still arriving.
    if step["prefill_tokens"] > 0:
        return "with prefill"
    if step["loads_started"] > 0 or step["loads_unfinished"] > 0 or step["promoted"] > 0:
        return "load activity"
    return "no load activity"


def print_step_kinds(steps: list[dict[str, float]]) -> None:
    period_total = 0.0
    periods_by_kind: dict[str, list[float]] = {}
    for step in steps:
        period_total += step["period_ms"]
        periods_by_kind.setdefault(step_kind(step), []).append(step["period_ms"])
    print("  1b. all steps by kind, rank 0: share of step time, period_ms")
    for kind, periods in periods_by_kind.items():
        share = 100.0 * sum(periods) / period_total
        print(f"       {kind:18s} {share:5.1f}%  {describe(periods)}")


def print_decode_groups(steps: list[dict[str, float]], concurrency: int) -> None:
    groups = {"no load activity": [], "load activity": [], "with prefill": []}
    for step in steps:
        if step["reqs"] < concurrency - 2:
            continue
        groups[step_kind(step)].append(step)
    print(f"  2. full-batch steps (>= {concurrency - 2} requests), rank 0")
    for group_name, group_steps in groups.items():
        print(f"     {group_name}: {len(group_steps)} steps")
        for name in STEP_FIELDS:
            values = []
            for step in group_steps:
                if name == "gpu_ms" and step[name] < 0:
                    continue
                values.append(step[name])
            print(f"       {name:24s} {describe(values)}")


def print_block_layout(steps: list[dict[str, float]]) -> None:
    refs = []
    unique = []
    runs = []
    for step in steps:
        if step["block_refs"] < 0:
            continue
        refs.append(step["block_refs"])
        unique.append(step["block_unique"])
        runs.append(step["block_runs"])
    print("  3. KV block layout of the batch, rank 0")
    print(f"       block_refs   {describe(refs)}")
    print(f"       block_unique {describe(unique)}")
    print(f"       block_runs   {describe(runs)}")


def print_ranks(steps_by_rank: dict[int, list[dict[str, float]]], concurrency: int) -> None:
    print("  4. per rank: main-thread run-queue wait (s), other threads' CPU (s),"
          " GPU ms of full-batch decode steps")
    for rank in sorted(steps_by_rank):
        runq_total = 0.0
        other_total = 0.0
        decode_gpu = []
        for step in steps_by_rank[rank]:
            runq_total += step["main_runq_ms"]
            other_total += step["other_threads_cpu_ms"]
            if step["prefill_tokens"] == 0 and step["reqs"] >= concurrency - 2 and step["gpu_ms"] >= 0:
                decode_gpu.append(step["gpu_ms"])
        print(f"       rank {rank}: runq {runq_total / 1e3:7.3f}  other CPU {other_total / 1e3:8.3f}"
              f"  decode gpu_ms {describe(decode_gpu)}")


def main() -> None:
    results_root = Path(sys.argv[1])
    for setting_dir in sorted(results_root.iterdir()):
        if not (setting_dir / "summary.csv").is_file():
            continue
        config, runs = read_runs(setting_dir)
        all_steps = read_trace(setting_dir / config / "serve.log")
        print(f"===== {setting_dir.name} ({config}): {len(all_steps)} step-trace lines")
        for run in runs:
            steps_by_rank: dict[int, list[dict[str, float]]] = {}
            for step in all_steps:
                if run.start_epoch <= step["t"] <= run.end_epoch:
                    steps_by_rank.setdefault(int(step["rank"]), []).append(step)
            print(f"--- {run.tag}")
            rank0_steps = steps_by_rank.get(0, [])
            if not rank0_steps:
                print("  no rank 0 steps in the window")
                continue
            print_totals(rank0_steps)
            print_step_kinds(rank0_steps)
            print_decode_groups(rank0_steps, run.concurrency)
            print_block_layout(rank0_steps)
            print_ranks(steps_by_rank, run.concurrency)


if __name__ == "__main__":
    main()
