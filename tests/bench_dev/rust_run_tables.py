"""Tables of one test-qwen38-27B-rust.sh run, laid out like
intel-accel-for-llm-kimi-k3/doc/design/b300-bench-results.zh-CN.md: one row per
(input, concurrency, hit rate), one column per setting.

Reads <run dir>/<setting>/summary.csv of every setting, from the first given run
directory that has the setting; settings found in none are left out. Settings
are split by attention backend, and every metric gets one table per backend:
  FlashInfer: vllm-wo-prefix, vllm-prefix, kvshrink-zip-dsa, kvshrink-nozip-dsa,
              kvshrink-nozip-nodsa, kvshrink-nozip-dsa-omp12,
              kvshrink-nozip-nodsa-omp12
  FLASH_ATTN: kvshrink-zip-dsa-flash-attn, lmcache
Tables: TTFT (ms, mean/p50/p95), TPOT (ms, mean/p50/p95), output tok/s, actual
hit rate; then one ScratchPool table for the kvshrink settings (waits:
count / over 10 ms / longest ms / total ms, all TP ranks added) and one line of
checks per run.
Actual hit rate = hit tokens / prompt tokens of the run's requests plus the
bench's 1 test and 5 warm-up requests, from vLLM's counters: GPU prefix cache
hits for vllm-prefix, external (connector) hits for kvshrink and lmcache. The
prompt tokens are vllm-prefix's GPU prefix cache query tokens of the same run:
every setting sends the same prompts (same seed) through the same Rust
tokenizer, and only vllm-prefix counts queries.
vllm-wo-prefix runs hit rate 80 only; its hit 95 rows repeat the hit 80 values
(without KV reuse both hit rates do the same work).

Usage: python3 tests/bench_dev/rust_run_tables.py bench-results/<run dir> [bench-results/<run dir> ...]
"""

import csv
import os
import sys
from functools import partial

BACKEND_GROUPS = (
    (
        "FlashInfer",
        (
            "vllm-wo-prefix",
            "vllm-prefix",
            "kvshrink-zip-dsa",
            "kvshrink-nozip-dsa",
            "kvshrink-nozip-nodsa",
            "kvshrink-nozip-dsa-omp12",
            "kvshrink-nozip-nodsa-omp12",
        ),
    ),
    ("FLASH_ATTN", ("kvshrink-zip-dsa-flash-attn", "lmcache")),
)
KVSHRINK_SETTINGS = (
    "kvshrink-zip-dsa",
    "kvshrink-nozip-dsa",
    "kvshrink-nozip-nodsa",
    "kvshrink-nozip-dsa-omp12",
    "kvshrink-nozip-nodsa-omp12",
    "kvshrink-zip-dsa-flash-attn",
)
LATENCY_TABLES = (
    ("TTFT", "ttft", 0),
    ("TPOT", "tpot", 1),
)


def read_rows(run_dir, setting):
    """{(input, concurrency, hit rate): summary row} of one setting."""
    rows = {}
    with open(f"{run_dir}/{setting}/summary.csv") as summary_file:
        for row in csv.DictReader(summary_file):
            key = (int(row["input_len_tokens"]), int(row["max_concurrency"]), int(row["hit_rate_pct"]))
            rows[key] = row
    return rows


def row_for(rows, setting, key):
    """The run of `key`; vllm-wo-prefix's hit 95 row is its hit 80 run."""
    input_len, concurrency, hit_rate = key
    if setting == "vllm-wo-prefix" and hit_rate == 95:
        return rows.get((input_len, concurrency, 80))
    return rows.get(key)


def latency_cell(row, metric, decimals):
    mean = float(row[f"mean_{metric}_ms"])
    median = float(row[f"median_{metric}_ms"])
    p95 = float(row[f"p95_{metric}_ms"])
    return f"{mean:.{decimals}f}/{median:.{decimals}f}/{p95:.{decimals}f}"


def throughput_cell(row):
    return f"{float(row['output_throughput_tok_s']):.0f}"


def scratch_cell(row):
    longest = float(row["scratch_wait_max_ms"])
    total = float(row["scratch_wait_total_ms"])
    return f"{row['scratch_wait_count']} / {row['scratch_wait_over_10ms_count']} / {longest:.1f} / {total:.0f}"


def print_table(title, settings, rows_by_setting, keys, format_cell):
    print(f"#### {title}")
    print("")
    print("| 输入 | 并发 | hit | " + " | ".join(settings) + " |")
    print("|---|---|---|" + "---|" * len(settings))
    for key in keys:
        input_len, concurrency, hit_rate = key
        cells = []
        for setting in settings:
            row = row_for(rows_by_setting[setting], setting, key)
            if row is None:
                cells.append("")
            else:
                cells.append(format_cell(row))
        print(f"| {input_len // 1000}k | {concurrency} | {hit_rate} | " + " | ".join(cells) + " |")
    print("")


def print_hit_rate_table(title, settings, rows_by_setting, keys):
    print(f"#### {title}")
    print("")
    print("| 输入 | 并发 | hit | " + " | ".join(settings) + " |")
    print("|---|---|---|" + "---|" * len(settings))
    for key in keys:
        input_len, concurrency, hit_rate = key
        prompt_tokens = int(rows_by_setting["vllm-prefix"][key]["gpu_prefix_cache_query_tokens"])
        cells = []
        for setting in settings:
            row = row_for(rows_by_setting[setting], setting, key)
            hit_tokens = int(row["gpu_prefix_cache_hit_tokens"]) + int(row["external_prefix_cache_hit_tokens"])
            cells.append(f"{100.0 * hit_tokens / prompt_tokens:.1f}%")
        print(f"| {input_len // 1000}k | {concurrency} | {hit_rate} | " + " | ".join(cells) + " |")
    print("")


def print_checks(rows_by_setting, keys):
    print("#### checks per run: completed/prompts, preemptions, evict count / >10 ms / longest ms, "
          "DSA batch failures, block size, GPU prefix cache hit/query (warm-ups included)")
    print("")
    for setting in rows_by_setting:
        for key in keys:
            row = rows_by_setting[setting].get(key)
            if row is None:
                continue
            prefix_share = ""
            hit_tokens = int(row["gpu_prefix_cache_hit_tokens"])
            query_tokens = int(row["gpu_prefix_cache_query_tokens"])
            if hit_tokens > 0 and query_tokens > 0:
                prefix_share = f", GPU prefix {100.0 * hit_tokens / query_tokens:.1f}%"
            input_len, concurrency, hit_rate = key
            print(
                f"- {setting} {input_len // 1000}k con {concurrency} hit {hit_rate}: "
                f"{row['completed']}/{row['num_prompts']}, preempt {row['preemptions']}, "
                f"evict {row['evict_count']} / {row['evict_over_10ms_count']} / {row['evict_max_ms']}, "
                f"DSA failed {row['dsa_batch_failed_count']}, block {row['block_size_tokens']}{prefix_share}"
            )


def main():
    run_dirs = sys.argv[1:]
    rows_by_setting = {}
    for _, settings in BACKEND_GROUPS:
        for setting in settings:
            for run_dir in run_dirs:
                if os.path.exists(f"{run_dir}/{setting}/summary.csv"):
                    rows_by_setting[setting] = read_rows(run_dir, setting)
                    break
    present_groups = []
    for backend, settings in BACKEND_GROUPS:
        present_settings = []
        for setting in settings:
            if setting in rows_by_setting:
                present_settings.append(setting)
        present_groups.append((backend, tuple(present_settings)))
    present_kvshrink_settings = []
    for setting in KVSHRINK_SETTINGS:
        if setting in rows_by_setting:
            present_kvshrink_settings.append(setting)
    key_set = set()
    for rows in rows_by_setting.values():
        for key in rows:
            key_set.add(key)
    keys = sorted(key_set)

    for name, metric, decimals in LATENCY_TABLES:
        for backend, settings in present_groups:
            cell = partial(latency_cell, metric=metric, decimals=decimals)
            print_table(f"{name}（ms，mean/p50/p95），{backend}", settings, rows_by_setting, keys, cell)
    for backend, settings in present_groups:
        print_table(f"出 tok/s，{backend}", settings, rows_by_setting, keys, throughput_cell)
    for backend, settings in present_groups:
        print_hit_rate_table(f"实际命中率，{backend}", settings, rows_by_setting, keys)
    print_table(
        "ScratchPool 等待（次数 / >10 ms 次数 / 最长 ms / 合计 ms，所有 TP rank 相加）",
        present_kvshrink_settings, rows_by_setting, keys, scratch_cell,
    )
    print_checks(rows_by_setting, keys)


if __name__ == "__main__":
    main()
