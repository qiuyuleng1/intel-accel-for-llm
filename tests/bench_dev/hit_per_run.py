"""Actual KV cache hits of one test-qwen38-27B.sh result directory, per run.

kvshrink: every request's "externally-cached tokens" from the connector log, as
distinct values with request counts (the last line of a request counts; the
requests of one `vllm bench serve` share the id prefix req-cmpl-bench-<8 hex>,
the 5 warm-up requests have other ids and are left out).
vllm-prefix: vLLM logs no per-request prefix cache hit, only the counters
summary.csv keeps per run (gpu_prefix_cache_hit_tokens / _query_tokens, warm-ups
included).

Usage: python3 tests/bench_dev/hit_per_run.py bench-results/<run dir>
"""

import csv
import os
import re
import sys
from dataclasses import dataclass, field

CONNECTOR_LINE = re.compile(
    r"get_num_new_matched_tokens, req-cmpl-bench-([0-9a-f]+)-(\d+)(?:-0)?-[0-9a-f]+, "
    r"externally-cached tokens: (\d+)"
)
# "### <config> (<time>)" in test-qwen38-27B.log, "### <config> serve=..." in
# kvshrink-benchmark.sh's run.log.
CONFIG_LINE = re.compile(r"^### (\S+) (?:\(|serve=)")
PREWARM_LINE = re.compile(r"^\s+prewarm: hit 100, (\d+) prompts")
RUN_LINE = re.compile(r"^--- (\S+)_tp\d+_in(\d+)_con(\d+)_hit(\d+) prompts=(\d+)")


@dataclass
class PlannedBench:
    """One `vllm bench serve` of the kvshrink config, in the order it ran."""
    name: str
    num_prompts: int
    input_len: int = 0
    concurrency: int = 0
    hit_rate: int = 0


@dataclass
class BenchRequests:
    """Connector lines of one `vllm bench serve`, by request index."""
    hits_by_request: dict = field(default_factory=dict)


def read_lines(path):
    with open(path, "rb") as log_file:
        text = log_file.read().decode("utf-8", errors="replace")
    return text.replace("\r", "\n").split("\n")


def read_kvshrink_plan(test_log_path):
    plan = []
    config = ""
    for line in read_lines(test_log_path):
        config_match = CONFIG_LINE.match(line)
        if config_match:
            config = config_match.group(1)
            continue
        if config != "kvshrink":
            continue
        prewarm_match = PREWARM_LINE.match(line)
        if prewarm_match:
            plan.append(PlannedBench("prewarm", int(prewarm_match.group(1))))
            continue
        run_match = RUN_LINE.match(line)
        if run_match:
            plan.append(
                PlannedBench(
                    name="run",
                    num_prompts=int(run_match.group(5)),
                    input_len=int(run_match.group(2)),
                    concurrency=int(run_match.group(3)),
                    hit_rate=int(run_match.group(4)),
                )
            )
    return plan


def read_kvshrink_benches(serve_log_path):
    """Bench id -> BenchRequests, in the order the bench ids first appear."""
    benches = {}
    for line in read_lines(serve_log_path):
        match = CONNECTOR_LINE.search(line)
        if not match:
            continue
        bench_id = match.group(1)
        request_index = int(match.group(2))
        hit_tokens = int(match.group(3))
        if bench_id not in benches:
            benches[bench_id] = BenchRequests()
        hits_by_request = benches[bench_id].hits_by_request
        if request_index not in hits_by_request:
            hits_by_request[request_index] = []
        hits_by_request[request_index].append(hit_tokens)
    return benches


def describe_hits(bench, input_len):
    """Distinct hit values (last line per request), most tokens first."""
    count_by_hit = {}
    requests_with_differing_lines = 0
    for request_index in bench.hits_by_request:
        hits = bench.hits_by_request[request_index]
        if len(set(hits)) > 1:
            requests_with_differing_lines += 1
        last_hit = hits[-1]
        if last_hit not in count_by_hit:
            count_by_hit[last_hit] = 0
        count_by_hit[last_hit] += 1
    parts = []
    for hit_tokens in sorted(count_by_hit, reverse=True):
        percent = 100.0 * hit_tokens / input_len
        parts.append(f"{hit_tokens} ({percent:.1f}%) x {count_by_hit[hit_tokens]}")
    return "; ".join(parts), requests_with_differing_lines


def main():
    result_dir = sys.argv[1]
    # test-qwen38-27B-rust.sh gives every setting its own directory with
    # kvshrink-benchmark.sh's run.log and no test-qwen38-27B.log.
    plan_log = f"{result_dir}/test-qwen38-27B.log"
    if not os.path.exists(plan_log):
        plan_log = f"{result_dir}/run.log"
    plan = read_kvshrink_plan(plan_log)
    benches = read_kvshrink_benches(f"{result_dir}/kvshrink/serve.log")
    bench_ids = list(benches)
    if len(bench_ids) != len(plan):
        raise ValueError(
            f"{len(bench_ids)} bench ids in the serve log, {len(plan)} benches planned"
        )

    block_size = 0
    vllm_prefix_rows = []
    with open(f"{result_dir}/summary.csv") as summary_file:
        for row in csv.DictReader(summary_file):
            block_size = int(row["block_size_tokens"])
            if row["config"] == "vllm-prefix":
                vllm_prefix_rows.append(row)

    print("kvshrink: externally-cached tokens per request (last line per request)")
    print("input hit con prompts requests_logged max_shared_prefix_hit | hit tokens (hit/input) x requests")
    input_len_of_prewarm = 0
    for position in range(len(plan)):
        planned = plan[position]
        bench = benches[bench_ids[position]]
        requests_logged = len(bench.hits_by_request)
        if requests_logged != planned.num_prompts:
            raise ValueError(
                f"bench {bench_ids[position]}: {requests_logged} requests logged, "
                f"{planned.num_prompts} planned"
            )
        if planned.name == "prewarm":
            # The prewarm's input length is the one of the runs that follow it.
            input_len_of_prewarm = plan[position + 1].input_len
            hits, differing = describe_hits(bench, input_len_of_prewarm)
            print(f"{input_len_of_prewarm} prewarm(100) prompts={planned.num_prompts} | {hits}")
            continue
        shared_prefix_len = planned.input_len * planned.hit_rate // 100
        max_shared_prefix_hit = shared_prefix_len // block_size * block_size
        hits, differing = describe_hits(bench, planned.input_len)
        print(
            f"{planned.input_len} {planned.hit_rate} {planned.concurrency} "
            f"{planned.num_prompts} {requests_logged} {max_shared_prefix_hit} | {hits}"
            f" | requests whose lines differ: {differing}"
        )

    print("")
    print("vllm-prefix: vLLM counters per run (warm-ups included)")
    print("input hit con prompts hit_tokens query_tokens hit/query")
    for row in vllm_prefix_rows:
        hit_tokens = int(row["gpu_prefix_cache_hit_tokens"])
        query_tokens = int(row["gpu_prefix_cache_query_tokens"])
        percent = 100.0 * hit_tokens / query_tokens
        print(
            f"{row['input_len_tokens']} {row['hit_rate_pct']} {row['max_concurrency']} "
            f"{row['num_prompts']} {hit_tokens} {query_tokens} {percent:.1f}%"
        )


if __name__ == "__main__":
    main()
