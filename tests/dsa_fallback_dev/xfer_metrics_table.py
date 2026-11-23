# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tables from two kvshrink-benchmark.sh result dirs (IAXL_DSA_AUTO_FALLBACK_ENABLE=1 and =0):
per-rank KVStore H2D / D2H transfer counters read after each bench run, and the bench metrics.
Usage: python3 tests/dsa_fallback_dev/xfer_metrics_table.py <fallback1 dir> <fallback0 dir>"""

import glob
import json
import os
import sys

GB = 1e9
NS_PER_S = 1e9


def read_runs(results_dir: str) -> list:
    runs = []
    pattern = os.path.join(results_dir, "kvshrink", "in*", "*_xfer_after_rank0.json")
    for rank0_file in sorted(glob.glob(pattern)):
        prefix = rank0_file[: -len("_xfer_after_rank0.json")]
        with open(prefix + ".json") as bench_file:
            bench = json.load(bench_file)
        ranks = []
        rank = 0
        while os.path.exists(f"{prefix}_xfer_after_rank{rank}.json"):
            with open(f"{prefix}_xfer_after_rank{rank}.json") as metrics_file:
                ranks.append(json.load(metrics_file))
            rank += 1
        runs.append({"tag": os.path.basename(prefix), "bench": bench, "ranks": ranks})
    return runs


def print_xfer_table(direction: str, settings: list) -> None:
    print(f"\n{direction.upper()}")
    print("| setting | run | rank | bytes (GB) | in flight (s) | GB/s | waiting in xfer_wait (s) | in flight / run duration |")
    print("|---|---|---|---|---|---|---|---|")
    for setting_name, runs in settings:
        for run in runs:
            duration_s = run["bench"]["duration"]
            for metrics in run["ranks"]:
                in_flight_s = metrics[f"{direction}_ns"] / NS_PER_S
                print(
                    f"| {setting_name} | {run['tag']} | {metrics['rank']} | "
                    f"{metrics[f'{direction}_bytes'] / GB:.2f} | {in_flight_s:.2f} | "
                    f"{metrics[f'{direction}_gbps']:.2f} | {metrics[f'{direction}_wait_ns'] / NS_PER_S:.2f} | "
                    f"{in_flight_s / duration_s * 100:.1f}% |"
                )


def print_bench_table(settings: list) -> None:
    print("\nBENCH")
    print("| setting | run | duration (s) | req/s | output tok/s | TTFT mean/p50/p95 (ms) | TPOT mean/p50/p95 (ms) |")
    print("|---|---|---|---|---|---|---|")
    for setting_name, runs in settings:
        for run in runs:
            bench = run["bench"]
            print(
                f"| {setting_name} | {run['tag']} | {bench['duration']:.1f} | "
                f"{bench['request_throughput']:.3f} | {bench['output_throughput']:.1f} | "
                f"{bench['mean_ttft_ms']:.1f} / {bench['median_ttft_ms']:.1f} / {bench['p95_ttft_ms']:.1f} | "
                f"{bench['mean_tpot_ms']:.2f} / {bench['median_tpot_ms']:.2f} / {bench['p95_tpot_ms']:.2f} |"
            )


def main() -> int:
    settings = [
        ("IAXL_DSA_AUTO_FALLBACK_ENABLE=1", read_runs(sys.argv[1])),
        ("IAXL_DSA_AUTO_FALLBACK_ENABLE=0", read_runs(sys.argv[2])),
    ]
    print_xfer_table("h2d", settings)
    print_xfer_table("d2h", settings)
    print_bench_table(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
