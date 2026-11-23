# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 2 check of one vLLM run: read <run_dir>/serve.log, log.kvshrink-vllm and outputs.json (written by
run_tests.sh) and check the calibration line, the copy backend of every rank and the replies.
Usage: e2e_check.py <run_dir> <IAXL_DSA_GD_ENABLE 0|1> <TP size>"""

import json
import os
import re
import sys

SUMMARY_LINE = re.compile(r"\[dsa_calibration\] crossover H2D (\d+) KB, D2H (\d+) KB")
BLOCK_SIZE_LINE = re.compile(r"copy_chunks_batch: contiguous block size = ([\d.]+) KB")
BACKEND_LINE = re.compile(r"copy_chunks_batch (H2D|D2H): using (Intel DSA|CUDA) backend")
HIT_LINE = re.compile(r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+)")


def read_lines(path: str) -> list[str]:
    with open(path, errors="replace") as log_file:
        return log_file.read().splitlines()


def main() -> int:
    run_dir = sys.argv[1]
    dsa_enabled = sys.argv[2] == "1"
    tp_size = int(sys.argv[3])
    serve_lines = read_lines(os.path.join(run_dir, "serve.log"))
    vllm_log_lines = read_lines(os.path.join(run_dir, "log.kvshrink-vllm"))

    # Lines up to the summary come from the calibration workers, later ones from vLLM.
    summary_index = -1
    summary_count = 0
    calibration_lines = []
    for index in range(len(serve_lines)):
        if "[dsa_calibration]" in serve_lines[index]:
            calibration_lines.append(serve_lines[index])
        if SUMMARY_LINE.search(serve_lines[index]):
            summary_index = index
            summary_count += 1
    vllm_lines = serve_lines[summary_index + 1:]

    block_sizes_kb = []
    backends = {"H2D": [], "D2H": []}
    calibration_dsa_lines = {"H2D": 0, "D2H": 0}
    for index in range(len(serve_lines)):
        match = BACKEND_LINE.search(serve_lines[index])
        if match and index < summary_index and match.group(2) == "Intel DSA":
            calibration_dsa_lines[match.group(1)] += 1
    for line in vllm_lines:
        match = BLOCK_SIZE_LINE.search(line)
        if match:
            block_sizes_kb.append(float(match.group(1)))
        match = BACKEND_LINE.search(line)
        if match:
            backends[match.group(1)].append(match.group(2))

    hits_by_log_id = {}
    traceback_count = 0
    for line in serve_lines:
        if "Traceback" in line:
            traceback_count += 1
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

    print(f"calibration lines: {calibration_lines}")
    print(f"calibration DSA backend lines (direction: workers): {calibration_dsa_lines}")
    print(f"vLLM contiguous block sizes (KB, one per process): {block_sizes_kb}")
    print(f"vLLM backends (direction: one per process): {backends}")
    print(f"Traceback lines: {traceback_count}")
    print("round prompt prompt_tokens hit_tokens text")
    for reply in replies:
        print(f"  {reply['round']} {reply['prompt_index']} {reply['prompt_tokens']} "
              f"{reply['hit_tokens']} {reply['text']!r}")

    checks = []
    if dsa_enabled:
        checks.append(("one calibration summary line", summary_count == 1))
        summary_in_vllm_log = False
        for line in vllm_log_lines:
            if SUMMARY_LINE.search(line):
                summary_in_vllm_log = True
        checks.append(("summary line also in log.kvshrink-vllm", summary_in_vllm_log))
        checks.append((f"{tp_size} calibration workers used DSA in both directions",
                       calibration_dsa_lines["H2D"] == tp_size and calibration_dsa_lines["D2H"] == tp_size))
        crossover_kb = {"H2D": 0, "D2H": 0}
        if summary_count == 1:
            match = SUMMARY_LINE.search(serve_lines[summary_index])
            crossover_kb["H2D"] = int(match.group(1))
            crossover_kb["D2H"] = int(match.group(2))
        block_kb = 0.0
        if block_sizes_kb:
            block_kb = block_sizes_kb[0]
        for direction in ("H2D", "D2H"):
            expected = "Intel DSA"
            if crossover_kb[direction] > 0 and block_kb >= crossover_kb[direction]:
                expected = "CUDA"
            all_expected = len(backends[direction]) == tp_size
            for backend in backends[direction]:
                if backend != expected:
                    all_expected = False
            checks.append((f"{direction}: {tp_size} ranks use {expected} (block {block_kb} KB, "
                           f"crossover {crossover_kb[direction]} KB)", all_expected))
    else:
        checks.append(("no calibration lines", len(calibration_lines) == 0))
        for direction in ("H2D", "D2H"):
            all_cuda = len(backends[direction]) == tp_size
            for backend in backends[direction]:
                if backend != "CUDA":
                    all_cuda = False
            checks.append((f"{direction}: {tp_size} ranks use CUDA", all_cuda))
    last_round = replies[-1]["round"]
    every_prompt_hit = True
    correct = True
    for reply in replies:
        if reply["round"] == last_round and not reply["hit_tokens"]:
            every_prompt_hit = False
        # e2e_requests.py asks for item 17 of report <prompt_index>; its value follows this formula.
        expected_value = (reply["prompt_index"] * 7919 + 17 * 104729) % 100000
        if str(expected_value) not in reply["text"]:
            correct = False
    checks.append((f"every prompt hits in round {last_round}", every_prompt_hit))
    checks.append(("every reply contains the correct value", correct))
    checks.append(("no Traceback", traceback_count == 0))

    all_passed = True
    print("== checks")
    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {description}")
        if not passed:
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
