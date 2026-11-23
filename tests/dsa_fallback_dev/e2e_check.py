# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 4 check: read <model_dir>/fallback1 and <model_dir>/fallback0 (serve.log + outputs.json
written by run_tests.sh) and print the calibration lines, copy backend lines, per-request
KVShrink hits and whether the replies of IAXL_DSA_AUTO_FALLBACK_ENABLE=1 and =0 are identical."""

import json
import os
import re
import sys

CROSSOVER_LINE = re.compile(r"\[rank(\d+)\] DSA crossover measured by rank (\d+): H2D (.+?), D2H (.+?)(?: \|.*)?$")
CALIBRATE_LINE = re.compile(r"\[kv_xfer\] dsa calibrate GPU(\d+) (H2D|D2H): (.*)")
BLOCK_SIZE_LINE = re.compile(r"copy_chunks_batch: contiguous block size = ([\d.]+) KB")
BACKEND_LINE = re.compile(r"copy_chunks_batch (H2D|D2H): using (Intel DSA|CUDA) backend")
HIT_LINE = re.compile(r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+)")


def read_run(run_dir: str) -> dict:
    crossover_rows = []
    calibrate_rows = []
    block_sizes = {}
    backends = {}
    hits_by_log_id = {}
    traceback_count = 0
    with open(os.path.join(run_dir, "serve.log"), errors="replace") as log_file:
        for line in log_file:
            if "Traceback" in line:
                traceback_count += 1
            match = CROSSOVER_LINE.search(line)
            if match:
                crossover_rows.append(match.groups())
            match = CALIBRATE_LINE.search(line)
            if match:
                calibrate_rows.append(match.groups())
            match = BLOCK_SIZE_LINE.search(line)
            if match:
                block_sizes[match.group(1)] = block_sizes.get(match.group(1), 0) + 1
            match = BACKEND_LINE.search(line)
            if match:
                key = f"{match.group(1)} {match.group(2)}"
                backends[key] = backends.get(key, 0) + 1
            match = HIT_LINE.search(line)
            if match:
                hits_by_log_id[match.group(1)] = int(match.group(2))
    with open(os.path.join(run_dir, "outputs.json")) as outputs_file:
        replies = json.load(outputs_file)
    for reply in replies:
        reply["hit_tokens"] = None
        for log_id, hit_tokens in hits_by_log_id.items():
            if log_id.startswith(reply["response_id"]):
                reply["hit_tokens"] = hit_tokens
    return {
        "crossover_rows": crossover_rows,
        "calibrate_rows": calibrate_rows,
        "block_sizes": block_sizes,
        "backends": backends,
        "replies": replies,
        "traceback_count": traceback_count,
    }


def print_run(name: str, run: dict) -> None:
    print(f"== {name}")
    print(f"KVStore calibration lines: {len(run['crossover_rows'])}")
    for rank, first_rank, h2d, d2h in run["crossover_rows"]:
        print(f"  rank {rank}: measured by rank {first_rank}, H2D {h2d}, D2H {d2h}")
    print(f"C++ calibration lines: {len(run['calibrate_rows'])}")
    for gpu, direction, text in run["calibrate_rows"]:
        print(f"  GPU{gpu} {direction}: {text}")
    print(f"contiguous block size lines (KB: processes): {run['block_sizes']}")
    print(f"backend lines (direction backend: processes): {run['backends']}")
    print(f"Traceback lines: {run['traceback_count']}")
    print("round prompt prompt_tokens hit_tokens text")
    for reply in run["replies"]:
        print(
            f"  {reply['round']} {reply['prompt_index']} {reply['prompt_tokens']} "
            f"{reply['hit_tokens']} {reply['text']!r}"
        )


def main() -> int:
    model_dir = sys.argv[1]
    with_fallback = read_run(os.path.join(model_dir, "fallback1"))
    without_fallback = read_run(os.path.join(model_dir, "fallback0"))
    print_run("IAXL_DSA_AUTO_FALLBACK_ENABLE=1", with_fallback)
    print_run("IAXL_DSA_AUTO_FALLBACK_ENABLE=0", without_fallback)

    checks = []
    distinct_values = set()
    for row in with_fallback["crossover_rows"]:
        distinct_values.add((row[2], row[3]))
    tp_size = 8
    if len(sys.argv) > 2:
        tp_size = int(sys.argv[2])
    checks.append((f"=1: {tp_size} ranks log the crossover", len(with_fallback["crossover_rows"]) == tp_size))
    checks.append(("=1: all ranks use the same crossover", len(distinct_values) == 1))
    checks.append(("=0: no rank calibrates", len(without_fallback["crossover_rows"]) == 0))
    last_round = with_fallback["replies"][-1]["round"]
    for run_name, run in (("=1", with_fallback), ("=0", without_fallback)):
        every_prompt_hit = True
        for reply in run["replies"]:
            if reply["round"] == last_round and not reply["hit_tokens"]:
                every_prompt_hit = False
        checks.append((f"{run_name}: every prompt hits in round {last_round}", every_prompt_hit))
        checks.append((f"{run_name}: no Traceback", run["traceback_count"] == 0))
        # Corrupted KV shows up as garbled text; whether the reply is reasonable is read by hand.
        readable = True
        for reply in run["replies"]:
            text = reply["text"].replace("\n", " ").replace("\t", " ")
            if not text.strip() or not text.isascii() or not text.isprintable():
                readable = False
        checks.append((f"{run_name}: every reply is non-empty printable ASCII", readable))
        # e2e_requests.py asks for item 17 of report <prompt_index>; its value follows this formula.
        correct = True
        for reply in run["replies"]:
            expected_value = (reply["prompt_index"] * 7919 + 17 * 104729) % 100000
            if str(expected_value) not in reply["text"]:
                correct = False
        checks.append((f"{run_name}: every reply contains the correct value", correct))
        checks.append((f"{run_name}: no C++ calibration lines (IAXL_DEBUG_LOG off)", len(run["calibrate_rows"]) == 0))
    for index in range(min(len(with_fallback["replies"]), len(without_fallback["replies"]))):
        if with_fallback["replies"][index]["text"] != without_fallback["replies"][index]["text"]:
            print(f"text differs between =1 and =0 (not a failure): reply {index}")

    all_passed = True
    print("== checks")
    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {description}")
        if not passed:
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
