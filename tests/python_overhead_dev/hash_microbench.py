# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only timing of the DDR keys kvshrink builds per request, and of Record.has on them.
# No vLLM server, no GPU: run in the dev container with CUDA_VISIBLE_DEVICES= (empty).
#   current_keys: kvshrink today, kvshrink_connector.py L282-L287 at 78aaef4 (xxhash of repr)
#   hex64/hex32/hex16_keys: the proposal, request.block_hashes (computed by vLLM anyway), the first
#                 32/16/8 bytes of each -> .hex()
#   has_<format>: Record.has on those keys, all present (all-hit scenario)
#
#   CUDA_VISIBLE_DEVICES= taskset -c 64 python3 tests/python_overhead_dev/hash_microbench.py \
#       --output-dir tests/python_overhead_dev/results/hash_microbench/<dir>

import argparse
import functools
import json
import os
import platform
import random
import statistics
import time

import torch
import vllm
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.utils import ConstantList

from iaxl.torch_ext import Record
from iaxl.utils.hash import generate_block_hashs

BLOCK_SIZE = 16
# KVStore.LABEL
RECORD_LABEL = "kv"
# Qwen3 special tokens start at 151643, so random ids below it have as many digits as real prompts.
TOKEN_ID_UPPER_BOUND = 151643
# Prompt tokens per request and prompts in DDR, as in run_lineprofiler_bench.sh: the DDR held the
# warmup prompts (2 x concurrency) plus the one measured prompt.
WORKLOADS = {
    "32k": {"prompt_tokens": 32768, "prompts_in_ddr": 3},
    "2k": {"prompt_tokens": 2048, "prompts_in_ddr": 33},
}
VLLM_HASH_ALGO = "sha256"
# Bytes of vLLM's 32-byte sha256 kept in the key; each byte becomes 2 hex characters.
HEX_KEY_BYTES = {"hex64": 32, "hex32": 16, "hex16": 8}


def make_prompt(prompt_tokens: int, seed: int) -> list[int]:
    generator = random.Random(seed)
    token_ids = []
    for _ in range(prompt_tokens):
        token_ids.append(generator.randrange(TOKEN_ID_UPPER_BOUND))
    return token_ids


def compute_vllm_block_hashes(token_ids: list[int]) -> list[bytes]:
    """request.block_hashes as vLLM 0.29 fills it for a text-only prompt (no extra keys)."""
    hash_function = get_hash_fn_by_name(VLLM_HASH_ALGO)
    block_hashes = []
    parent_block_hash = None
    for start in range(0, len(token_ids) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block_hash = kv_cache_utils.hash_block_tokens(
            hash_function, parent_block_hash, token_ids[start : start + BLOCK_SIZE], None
        )
        block_hashes.append(block_hash)
        parent_block_hash = block_hash
    return block_hashes


def build_current_keys(all_token_ids: ConstantList, block_size: int) -> list[str]:
    # Copied unchanged from kvshrink_connector.py L282-L287 (78aaef4) so the timing matches.
    block_hashes = [
        str(block_hash)
        for block_hash in generate_block_hashs(
            all_token_ids[:-1], block_size
        )
    ]
    return block_hashes


def build_hex_keys(
    all_token_ids: ConstantList, vllm_block_hashes: list[bytes], block_size: int, key_bytes: int
) -> list[str]:
    # kvshrink leaves out the last prompt token, so when the prompt fills whole blocks it uses
    # one block fewer than vLLM hashed.
    num_blocks = (len(all_token_ids) - 1) // block_size
    keys = []
    for index in range(num_blocks):
        # Slicing all 32 bytes returns the same bytes object, so hex64 pays no copy.
        keys.append(vllm_block_hashes[index][:key_bytes].hex())
    return keys


def summarize(durations_ms: list[float]) -> dict:
    return {
        "samples": len(durations_ms),
        "mean_ms": statistics.mean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
    }


def time_once_ms(function) -> float:
    start = time.perf_counter_ns()
    function()
    return (time.perf_counter_ns() - start) / 1e6


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_workload(name: str, workload: dict, output_dir: str, warmup: int, repeats: int) -> dict:
    prompt_tokens = workload["prompt_tokens"]
    measured_prompt = make_prompt(prompt_tokens, seed=0)
    all_token_ids = ConstantList(measured_prompt)
    vllm_block_hashes = compute_vllm_block_hashes(measured_prompt)

    expected_blocks = (prompt_tokens - 1) // BLOCK_SIZE
    current_keys = build_current_keys(all_token_ids, BLOCK_SIZE)
    check(len(current_keys) == expected_blocks, f"{name}: current keys {len(current_keys)}")
    check(len(set(current_keys)) == expected_blocks, f"{name}: duplicate current keys")
    keys_by_format = {"current": current_keys}
    for key_format, key_bytes in HEX_KEY_BYTES.items():
        keys = build_hex_keys(all_token_ids, vllm_block_hashes, BLOCK_SIZE, key_bytes)
        check(len(keys) == expected_blocks, f"{name}: {key_format} keys {len(keys)}")
        check(len(set(keys)) == expected_blocks, f"{name}: duplicate {key_format} keys")
        for key in keys:
            check(len(key) == 2 * key_bytes, f"{name}: {key_format} key length {len(key)}")
        keys_by_format[key_format] = keys

    # Fill one DDR record per key format with the measured prompt plus the other prompts.
    keys_in_ddr_by_format = {}
    for key_format, keys in keys_by_format.items():
        keys_in_ddr_by_format[key_format] = list(keys)
    for seed in range(1, workload["prompts_in_ddr"]):
        other_prompt = make_prompt(prompt_tokens, seed=seed)
        other_token_ids = ConstantList(other_prompt)
        other_block_hashes = compute_vllm_block_hashes(other_prompt)
        keys_in_ddr_by_format["current"].extend(build_current_keys(other_token_ids, BLOCK_SIZE))
        for key_format, key_bytes in HEX_KEY_BYTES.items():
            keys_in_ddr_by_format[key_format].extend(
                build_hex_keys(other_token_ids, other_block_hashes, BLOCK_SIZE, key_bytes)
            )

    records = {}
    for key_format, keys_in_ddr in keys_in_ddr_by_format.items():
        record = Record(os.path.join(output_dir, f"chunks_{name}_{key_format}.db"))
        record.submit(RECORD_LABEL, keys_in_ddr)
        record.sync()
        all_found = all(record.has(RECORD_LABEL, keys_by_format[key_format]))
        check(all_found, f"{name}: {key_format} miss")
        records[key_format] = record

    # functools.partial, not lambda: a lambda made in a loop would see the last loop values.
    cases = {"current_keys": functools.partial(build_current_keys, all_token_ids, BLOCK_SIZE)}
    for key_format, key_bytes in HEX_KEY_BYTES.items():
        cases[f"{key_format}_keys"] = functools.partial(
            build_hex_keys, all_token_ids, vllm_block_hashes, BLOCK_SIZE, key_bytes
        )
    for key_format, record in records.items():
        cases[f"has_{key_format}"] = functools.partial(
            record.has, RECORD_LABEL, keys_by_format[key_format]
        )
    durations_ms = {}
    for case_name in cases:
        durations_ms[case_name] = []
    # Cases take turns in every round so that drift (CPU frequency, other load) hits all of them.
    for round_index in range(warmup + repeats):
        for case_name, function in cases.items():
            duration_ms = time_once_ms(function)
            if round_index >= warmup:
                durations_ms[case_name].append(duration_ms)

    for record in records.values():
        record.shutdown()

    key_mean_chars = {}
    for key_format, keys in keys_by_format.items():
        total_chars = 0
        for key in keys:
            total_chars += len(key)
        key_mean_chars[key_format] = total_chars / len(keys)
    result = {
        "prompt_tokens": prompt_tokens,
        "blocks": expected_blocks,
        "rows_in_ddr": len(keys_in_ddr_by_format["current"]),
        "key_mean_chars": key_mean_chars,
        "durations_ms": durations_ms,
        "summary": {},
    }
    for case_name in cases:
        result["summary"][case_name] = summarize(durations_ms[case_name])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    kv_cache_utils.init_none_hash(get_hash_fn_by_name(VLLM_HASH_ALGO))
    environment = {
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "vllm_file": vllm.__file__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "vllm_hash_algo": VLLM_HASH_ALGO,
        "warmup_rounds": args.warmup,
        "measured_rounds": args.repeats,
    }
    results = {"environment": environment, "workloads": {}}
    for name, workload in WORKLOADS.items():
        results["workloads"][name] = run_workload(
            name, workload, args.output_dir, args.warmup, args.repeats
        )

    with open(os.path.join(args.output_dir, "results.json"), "w") as result_file:
        json.dump(results, result_file, indent=1)

    print(json.dumps(environment))
    print(f"{'workload':8} {'case':13} {'n':>3} {'mean ms':>9} {'median ms':>9} "
          f"{'min ms':>8} {'max ms':>8}")
    for name, workload_result in results["workloads"].items():
        print(f"# {name}: {workload_result['prompt_tokens']} tokens, {workload_result['blocks']} "
              f"blocks, {workload_result['rows_in_ddr']} rows in DDR record, key mean chars "
              f"{json.dumps(workload_result['key_mean_chars'])}")
        for case_name, summary in workload_result["summary"].items():
            print(f"{name:8} {case_name:13} {summary['samples']:3d} {summary['mean_ms']:9.3f} "
                  f"{summary['median_ms']:9.3f} {summary['min_ms']:8.3f} "
                  f"{summary['max_ms']:8.3f}")


if __name__ == "__main__":
    main()
