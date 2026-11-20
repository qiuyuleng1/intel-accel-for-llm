# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Print tables 1 and 2 of doc/design/dsa-auto-fallback.zh-CN.md from stability_runs.sh logs.

Usage: python3 tests/dsa_fallback_dev/stability_tables.py <RESULT_DIR>
Runs whose log is still incomplete are skipped, so it also works while the experiment runs.
"""

import os
import re
import statistics
import sys
from dataclasses import dataclass

SETTING_DESCRIPTIONS = {
    "A": "64 MiB，热身 1 + 计时 3",
    "B": "64 MiB，热身 2 + 计时 10",
    "C": "128 MiB，热身 1 + 计时 3",
    "D": "128 MiB，热身 2 + 计时 10",
}
DIRECTIONS = ("H2D", "D2H")
FRAGMENT_KIBS = (256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048)
DSA_METHOD = "IAXL DSA"
BATCH_METHOD = "CUDA Copy Batch"
RESULT_LINE = re.compile(
    r"^(H2D|D2H)\s+(\d+)K\s+(CUDA Copy Batch|IAXL DSA)\s+[\d.]+\s+([\d.]+)\s+(PASS|FAIL)$"
)
RUN_LOG_NAME = re.compile(r"^run(\d+)\.log$")
NO_CROSSOVER = "无"


@dataclass
class RunResult:
    """GB/s of one benchmark run: gbps[direction][fragment KiB][method]."""
    round_number: int
    gbps: dict[str, dict[int, dict[str, float]]]


def fragment_label(fragment_kib: int) -> str:
    if fragment_kib % 1024 == 0:
        return f"{fragment_kib // 1024}M"
    return f"{fragment_kib}K"


def parse_run_log(path: str) -> dict[str, dict[int, dict[str, float]]]:
    gbps: dict[str, dict[int, dict[str, float]]] = {}
    with open(path, encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = RESULT_LINE.match(line.strip())
            if match is None:
                continue
            direction, fragment_text, method, gbps_text, check = match.groups()
            if check == "FAIL":
                raise ValueError(f"{path}: copy check FAIL: {line.strip()}")
            if direction not in gbps:
                gbps[direction] = {}
            fragment_kib = int(fragment_text)
            if fragment_kib not in gbps[direction]:
                gbps[direction][fragment_kib] = {}
            gbps[direction][fragment_kib][method] = float(gbps_text)
    return gbps


def is_complete(gbps: dict[str, dict[int, dict[str, float]]]) -> bool:
    for direction in DIRECTIONS:
        if direction not in gbps:
            return False
        for fragment_kib in FRAGMENT_KIBS:
            if fragment_kib not in gbps[direction]:
                return False
            methods = gbps[direction][fragment_kib]
            if DSA_METHOD not in methods or BATCH_METHOD not in methods:
                return False
    return True


def load_setting_runs(result_dir: str, setting: str) -> list[RunResult]:
    setting_dir = os.path.join(result_dir, setting)
    runs: list[RunResult] = []
    if not os.path.isdir(setting_dir):
        return runs
    for file_name in os.listdir(setting_dir):
        match = RUN_LOG_NAME.match(file_name)
        if match is None:
            continue
        path = os.path.join(setting_dir, file_name)
        gbps = parse_run_log(path)
        if not is_complete(gbps):
            print(f"（跳过未完成的运行：{setting}/{file_name}）")
            continue
        runs.append(RunResult(round_number=int(match.group(1)), gbps=gbps))
    runs.sort(key=lambda run: run.round_number)
    return runs


def spread_percent(values: list[float]) -> float:
    return (max(values) - min(values)) / statistics.mean(values) * 100


def crossover_kib(run: RunResult, direction: str) -> int | None:
    """First fragment size (ascending) where CUDA batch GB/s beats DSA, None if it never does."""
    for fragment_kib in FRAGMENT_KIBS:
        methods = run.gbps[direction][fragment_kib]
        if methods[BATCH_METHOD] > methods[DSA_METHOD]:
            return fragment_kib
    return None


def print_bandwidth_table(setting: str, direction: str, runs: list[RunResult]) -> None:
    print(f"### {direction}，设置 {setting}（{SETTING_DESCRIPTIONS[setting]}），{len(runs)} 次运行")
    print()
    print(
        "| 片段大小 | DSA 平均 | DSA 中位数 | DSA 波动 % | CUDA batch 平均 "
        "| CUDA batch 中位数 | CUDA batch 波动 % | CUDA batch 比 DSA 快的运行次数 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for fragment_kib in FRAGMENT_KIBS:
        dsa_values: list[float] = []
        batch_values: list[float] = []
        batch_faster_runs = 0
        for run in runs:
            methods = run.gbps[direction][fragment_kib]
            dsa_values.append(methods[DSA_METHOD])
            batch_values.append(methods[BATCH_METHOD])
            if methods[BATCH_METHOD] > methods[DSA_METHOD]:
                batch_faster_runs += 1
        print(
            f"| {fragment_label(fragment_kib)} "
            f"| {statistics.mean(dsa_values):.2f} | {statistics.median(dsa_values):.2f} "
            f"| {spread_percent(dsa_values):.1f} "
            f"| {statistics.mean(batch_values):.2f} | {statistics.median(batch_values):.2f} "
            f"| {spread_percent(batch_values):.1f} "
            f"| {batch_faster_runs}/{len(runs)} |"
        )
    print()


def format_crossover_counts(runs: list[RunResult], direction: str) -> str:
    counts: dict[int | None, int] = {}
    for run in runs:
        crossover = crossover_kib(run, direction)
        counts[crossover] = counts.get(crossover, 0) + 1
    parts: list[str] = []
    for fragment_kib in FRAGMENT_KIBS:
        if fragment_kib in counts:
            parts.append(f"{fragment_label(fragment_kib)}：{counts[fragment_kib]} 次")
    if None in counts:
        parts.append(f"{NO_CROSSOVER}：{counts[None]} 次")
    return "，".join(parts)


def print_crossover_table(runs_by_setting: dict[str, list[RunResult]]) -> None:
    print("### 交界点：每个值出现的次数")
    print()
    print("| 设置 | 运行次数 | H2D 交界点：每个值出现的次数 | D2H 交界点：每个值出现的次数 |")
    print("|---|---|---|---|")
    for setting, runs in runs_by_setting.items():
        print(
            f"| {setting}（{SETTING_DESCRIPTIONS[setting]}） | {len(runs)} "
            f"| {format_crossover_counts(runs, 'H2D')} | {format_crossover_counts(runs, 'D2H')} |"
        )
    print()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <RESULT_DIR>")
    result_dir = sys.argv[1]
    runs_by_setting: dict[str, list[RunResult]] = {}
    for setting in SETTING_DESCRIPTIONS:
        runs = load_setting_runs(result_dir, setting)
        if runs:
            runs_by_setting[setting] = runs

    print(f"结果目录：{result_dir}")
    print()
    print("- 平均 / 中位数：该片段大小在所有完成的运行里的 GB/s 平均值 / 中位数；每次运行的 GB/s 是计时几次里最快一次。")
    print("- 波动 %：(这些 GB/s 的最大值 − 最小值) ÷ 平均值 × 100。")
    print("- CUDA batch 比 DSA 快的运行次数：同一次运行里 CUDA batch 的 GB/s 大于 DSA 的运行次数 / 完成的运行次数。")
    print(f"- 交界点：每次运行从 256K 往上第一个 CUDA batch 比 DSA 快的片段大小；到 2M 都没有记为\"{NO_CROSSOVER}\"。")
    print()
    for setting, runs in runs_by_setting.items():
        for direction in DIRECTIONS:
            print_bandwidth_table(setting, direction, runs)
    print_crossover_table(runs_by_setting)


if __name__ == "__main__":
    main()
