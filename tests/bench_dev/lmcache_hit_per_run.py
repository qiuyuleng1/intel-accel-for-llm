"""lmcache hits per request, per run, of one test-qwen38-27B.sh result directory.

LMCache logs no hit token count per request. Its server logs one line per request:
  Prefetch request completed (L1+L2): <retained>/<requested> retained keys ...
  (external_request_id=cmpl-bench-<8 hex>-<index>-0-<suffix>, ...)
On Qwen3.8-27B at TP=4 every chunk has 4 attention keys (one per rank) and the hit
ends at one chunk with 4 mamba keys, so hit chunks = (retained - 4) / 4. A request
with no such line found no key and counts as a 0-token hit. The check:
per run, the hits of its requests plus those of its 5 warm-up requests (ids
cmpl-<16 hex>) must equal vLLM's counter external_prefix_cache_hit_tokens in
summary.csv.

Usage: python3 tests/bench_dev/lmcache_hit_per_run.py bench-results/<run dir>
"""

import csv
import os
import re
import sys
from dataclasses import dataclass, field

PREFETCH_LINE = re.compile(
    r"(\d+)/(\d+) retained keys .*external_request_id=(cmpl-(?:bench-[0-9a-f]+-\d+|[0-9a-f]+))-"
)
# "### <config> (<time>)" in test-qwen38-27B.log, "### <config> serve=..." in
# kvshrink-benchmark.sh's run.log.
CONFIG_LINE = re.compile(r"^### (\S+) (?:\(|serve=)")
PREWARM_LINE = re.compile(r"^\s+prewarm: hit 100, (\d+) prompts")
RUN_LINE = re.compile(r"^--- (\S+)_tp\d+_in(\d+)_con(\d+)_hit(\d+) prompts=(\d+)")
KEYS_PER_CHUNK = 4
MAMBA_KEYS = 4


@dataclass
class PlannedBench:
    """One `vllm bench serve` of the lmcache config, in the order it ran."""
    name: str
    num_prompts: int
    input_len: int = 0
    concurrency: int = 0
    hit_rate: int = 0


@dataclass
class BenchRequests:
    """Hit chunks of one bench's requests and of the warm-ups sent before it."""
    hit_chunks_by_request: dict = field(default_factory=dict)
    warmup_hit_chunks: dict = field(default_factory=dict)


def read_lines(path):
    with open(path, "rb") as log_file:
        text = log_file.read().decode("utf-8", errors="replace")
    return text.replace("\r", "\n").split("\n")


def read_plan(test_log_path):
    plan = []
    config = ""
    for line in read_lines(test_log_path):
        config_match = CONFIG_LINE.match(line)
        if config_match:
            config = config_match.group(1)
            continue
        if config != "lmcache":
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


def hit_chunks(retained):
    if retained == 0:
        return 0
    if retained < MAMBA_KEYS or (retained - MAMBA_KEYS) % KEYS_PER_CHUNK != 0:
        raise ValueError(f"retained keys {retained} do not fit 4 per chunk + 4 mamba")
    return (retained - MAMBA_KEYS) // KEYS_PER_CHUNK


def read_benches(server_log_path):
    """Bench ids in first-appearance order -> BenchRequests; the last line of a request counts."""
    benches = {}
    pending_warmups = {}
    for line in read_lines(server_log_path):
        match = PREFETCH_LINE.search(line)
        if not match:
            continue
        chunks = hit_chunks(int(match.group(1)))
        request_id = match.group(3)
        if not request_id.startswith("cmpl-bench-"):
            pending_warmups[request_id] = chunks
            continue
        bench_id = request_id.split("-")[2]
        if bench_id not in benches:
            benches[bench_id] = BenchRequests(warmup_hit_chunks=pending_warmups)
            pending_warmups = {}
        benches[bench_id].hit_chunks_by_request[request_id] = chunks
    return benches


def main():
    result_dir = sys.argv[1]
    # test-qwen38-27B-rust.sh gives every setting its own directory with
    # kvshrink-benchmark.sh's run.log and no test-qwen38-27B.log.
    plan_log = f"{result_dir}/test-qwen38-27B.log"
    if not os.path.exists(plan_log):
        plan_log = f"{result_dir}/run.log"
    plan = read_plan(plan_log)
    benches = read_benches(f"{result_dir}/lmcache/lmcache_server.log")
    bench_ids = list(benches)
    if len(bench_ids) != len(plan):
        raise ValueError(f"{len(bench_ids)} bench ids in the log, {len(plan)} benches planned")

    block_size = 0
    counter_by_run = {}
    with open(f"{result_dir}/summary.csv") as summary_file:
        for row in csv.DictReader(summary_file):
            if row["config"] != "lmcache":
                continue
            block_size = int(row["block_size_tokens"])
            key = (int(row["input_len_tokens"]), int(row["hit_rate_pct"]), int(row["max_concurrency"]))
            counter_by_run[key] = int(row["external_prefix_cache_hit_tokens"])

    print("input hit con prompts requests_logged max_shared_prefix_hit | hit tokens (hit/input) x requests | counter check")
    input_len_of_prewarm = 0
    for position in range(len(plan)):
        planned = plan[position]
        bench = benches[bench_ids[position]]
        if len(bench.hit_chunks_by_request) > planned.num_prompts:
            raise ValueError(
                f"bench {bench_ids[position]}: {len(bench.hit_chunks_by_request)} requests logged, "
                f"{planned.num_prompts} planned"
            )
        unlogged_requests = planned.num_prompts - len(bench.hit_chunks_by_request)
        if planned.name == "prewarm":
            input_len = plan[position + 1].input_len
        else:
            input_len = planned.input_len
        count_by_hit = {}
        total_hit_tokens = 0
        for chunks in bench.hit_chunks_by_request.values():
            hit_tokens = chunks * block_size
            total_hit_tokens += hit_tokens
            if hit_tokens not in count_by_hit:
                count_by_hit[hit_tokens] = 0
            count_by_hit[hit_tokens] += 1
        if unlogged_requests > 0:
            if 0 not in count_by_hit:
                count_by_hit[0] = 0
            count_by_hit[0] += unlogged_requests
        parts = []
        for hit_tokens in sorted(count_by_hit, reverse=True):
            parts.append(f"{hit_tokens} ({100.0 * hit_tokens / input_len:.1f}%) x {count_by_hit[hit_tokens]}")
        if planned.name == "prewarm":
            print(f"{input_len} prewarm(100) prompts={planned.num_prompts} | {'; '.join(parts)}")
            continue
        for chunks in bench.warmup_hit_chunks.values():
            total_hit_tokens += chunks * block_size
        counter = counter_by_run[(planned.input_len, planned.hit_rate, planned.concurrency)]
        check = "match" if counter == total_hit_tokens else f"MISMATCH log {total_hit_tokens} counter {counter}"
        max_shared_prefix_hit = planned.input_len * planned.hit_rate // 100 // block_size * block_size
        print(
            f"{planned.input_len} {planned.hit_rate} {planned.concurrency} {planned.num_prompts} "
            f"{len(bench.hit_chunks_by_request)} {max_shared_prefix_hit} | {'; '.join(parts)} | "
            f"warm-ups {len(bench.warmup_hit_chunks)}, {check}"
        )


if __name__ == "__main__":
    main()
