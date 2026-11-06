# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Checks that kvshrink hits the DDR with keys taken from vLLM's block hashes (dev helper).

first:   send a fresh long prompt, which must miss (externally-cached tokens 0); wait for its KV
         to be stored, then send the same prompt --hits more times, one at a time.
restart: after vLLM was restarted, send the prompt that `first` saved --hits times; the hits must
         load what the first server persisted to disk.
A hit must load every full block except the one holding the last prompt token (the connector
never loads that token); on hybrid models it must load at least one block.
Every hit output is compared with the miss output of `first`:
  --compare bytes   byte-identical (the server must run with VLLM_BATCH_INVARIANT=1)
  --compare answer  the first line answers the prompt's question (sensor and value of Record
                    0042, which lies in the loaded prefix); for GDN hybrid models, where vLLM
                    rejects VLLM_BATCH_INVARIANT, so outputs may legitimately differ
No output may contain U+FFFD or control characters. Greedy decoding (temperature 0, seed 0).
Writes request.json, responses.json, connector_log.txt and report.json to --out; exit code 0
only if every check passes.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

MAX_TOKENS = 64


@dataclass
class ConnectorConfig:
    block_size: int
    has_mamba: bool


@dataclass
class OutputCheck:
    response_id: str
    loaded_tokens: list[int]
    same_as_miss: bool
    first_difference: int
    answer_correct: bool
    bad_characters: list[str]
    text: str


def build_prompt(records: int, seed: int) -> str:
    generator = random.Random(seed)
    lines = ["Sensor log."]
    for index in range(records):
        sensor = f"{generator.choice('ABCDEFGH')}{generator.randint(10, 99)}"
        value = generator.randint(1000, 9999)
        lines.append(f"Record {index:04d}: sensor {sensor} measured {value} units.")
    lines.append("Question: Which sensor appears in Record 0042 and what did it measure?")
    lines.append("Answer:")
    return "\n".join(lines)


def expected_answer(prompt: str) -> str:
    record = re.search(r"Record 0042: sensor (\S+) measured (\d+) units\.", prompt)
    return f"{record.group(1)} measured {record.group(2)} units"


def post(api: str, body: dict) -> dict:
    request = urllib.request.Request(
        api, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def bad_characters(text: str) -> list[str]:
    found = []
    for character in text:
        if character == "\ufffd" or (ord(character) < 32 and character not in "\n\t\r"):
            found.append(hex(ord(character)))
    return found


def first_difference(reference: str, text: str) -> int:
    """Character offset of the first difference, -1 if the texts are equal."""
    limit = min(len(reference), len(text))
    for index in range(limit):
        if reference[index] != text[index]:
            return index
    if len(reference) != len(text):
        return limit
    return -1


def read_log(path: str, start_offset: int) -> str:
    # Byte offset, not line count: the log contains '\r' progress output.
    with open(path, "rb") as log_file:
        log_file.seek(start_offset)
        return log_file.read().decode(errors="replace")


def read_connector_config(vllm_log: str) -> ConnectorConfig:
    match = re.search(
        r"KVShrinkConnector\(SCHEDULER\): block_size=(\d+), has_mamba=(True|False)",
        read_log(vllm_log, 0),
    )
    if match is None:
        raise RuntimeError(f"no KVShrinkConnector(SCHEDULER) line in {vllm_log}")
    return ConnectorConfig(block_size=int(match.group(1)), has_mamba=match.group(2) == "True")


def loaded_tokens_by_request(connector_lines: list[str]) -> dict[str, list[int]]:
    # A request asked twice (not scheduled the first time) has two lines.
    loaded = {}
    for line in connector_lines:
        match = re.search(r"req-(\S+), externally-cached tokens: (\d+)", line)
        if match is None:
            continue
        loaded.setdefault(match.group(1), []).append(int(match.group(2)))
    return loaded


def loaded_tokens_of(response_id: str, loaded: dict[str, list[int]]) -> list[int]:
    # vLLM's internal request id starts with the id of the API response.
    values = []
    for request_id, tokens in loaded.items():
        if request_id.startswith(response_id):
            values.extend(tokens)
    return values


def check_output(
    response: dict, miss_text: str, answer: str, loaded: dict[str, list[int]]
) -> OutputCheck:
    text = response["choices"][0]["text"]
    first_line = text.strip().split("\n")[0]
    return OutputCheck(
        response_id=response["id"],
        loaded_tokens=loaded_tokens_of(response["id"], loaded),
        same_as_miss=text.encode() == miss_text.encode(),
        first_difference=first_difference(miss_text, text),
        answer_correct=answer in first_line,
        bad_characters=bad_characters(text),
        text=text,
    )


def hit_passes(
    hit: OutputCheck, compare: str, full_hit_tokens: int, config: ConnectorConfig
) -> bool:
    if not hit.loaded_tokens:
        return False
    for tokens in hit.loaded_tokens:
        if config.has_mamba:
            # A hybrid hit must end at a block whose mamba state was saved, which can be
            # earlier than the last full block.
            if tokens <= 0 or tokens % config.block_size != 0 or tokens > full_hit_tokens:
                return False
        elif tokens != full_hit_tokens:
            return False
    if hit.bad_characters:
        return False
    if compare == "bytes":
        return hit.same_as_miss
    return hit.answer_correct


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=["first", "restart"])
    parser.add_argument("--api", default="http://localhost:8020/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", choices=["bytes", "answer"], required=True)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--hits", type=int, default=3)
    parser.add_argument("--first-dir", help="result dir of the first phase (restart only)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = read_connector_config(args.vllm_log)
    log_start = os.path.getsize(args.vllm_log)

    if args.phase == "first":
        # A fresh prompt per run, so that the first request is a real miss.
        body = {
            "model": args.model,
            "prompt": build_prompt(args.records, int(time.time())),
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "seed": 0,
        }
        miss = post(args.api, body)
        time.sleep(5)  # let the miss's KV be stored
    else:
        if args.first_dir is None:
            parser.error("restart needs --first-dir")
        with open(os.path.join(args.first_dir, "responses.json")) as first_file:
            first_responses = json.load(first_file)
        body = first_responses["request"]
        miss = first_responses["miss"]

    hits = []
    for _ in range(args.hits):
        hits.append(post(args.api, body))
        time.sleep(1)
    time.sleep(2)

    connector_lines = []
    for line in read_log(args.vllm_log, log_start).split("\n"):
        if "get_num_new_matched_tokens" in line:
            connector_lines.append(line + "\n")
    loaded = loaded_tokens_by_request(connector_lines)

    prompt_tokens = miss["usage"]["prompt_tokens"]
    full_hit_tokens = (prompt_tokens - 1) // config.block_size * config.block_size
    miss_text = miss["choices"][0]["text"]
    answer = expected_answer(body["prompt"])
    miss_check = check_output(miss, miss_text, answer, loaded)
    hit_checks = []
    for hit in hits:
        hit_checks.append(check_output(hit, miss_text, answer, loaded))

    passed = True
    if args.phase == "first":
        if miss_check.loaded_tokens != [0] or miss_check.bad_characters:
            passed = False
    # Without a correct miss answer the answer comparison proves nothing.
    if args.compare == "answer" and not miss_check.answer_correct:
        passed = False
    for hit_check in hit_checks:
        if not hit_passes(hit_check, args.compare, full_hit_tokens, config):
            passed = False

    hit_reports = []
    for hit_check in hit_checks:
        hit_reports.append(asdict(hit_check))
    report = {
        "phase": args.phase,
        "verdict": "PASS" if passed else "FAIL",
        "compare": args.compare,
        "connector": asdict(config),
        "prompt_tokens": prompt_tokens,
        "full_hit_tokens": full_hit_tokens,
        "expected_answer": answer,
        "miss": asdict(miss_check),
        "hits": hit_reports,
    }
    with open(os.path.join(args.out, "request.json"), "w") as request_file:
        json.dump(body, request_file, indent=1)
    with open(os.path.join(args.out, "responses.json"), "w") as responses_file:
        json.dump({"request": body, "miss": miss, "hits": hits}, responses_file, indent=1)
    with open(os.path.join(args.out, "connector_log.txt"), "w") as connector_file:
        connector_file.writelines(connector_lines)
    with open(os.path.join(args.out, "report.json"), "w") as report_file:
        json.dump(report, report_file, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(
        f"SUMMARY {args.phase} {report['verdict']}: prompt_tokens={prompt_tokens} "
        f"block_size={config.block_size} has_mamba={config.has_mamba} "
        f"full_hit_tokens={full_hit_tokens} expected_answer={answer!r}"
    )
    print(
        f"SUMMARY miss: loaded_tokens={miss_check.loaded_tokens} "
        f"answer_correct={miss_check.answer_correct} "
        f"bad_characters={len(miss_check.bad_characters)} text={miss_text[:80]!r}"
    )
    for index in range(len(hit_checks)):
        hit_check = hit_checks[index]
        print(
            f"SUMMARY hit {index + 1}: loaded_tokens={hit_check.loaded_tokens} "
            f"same_as_miss={hit_check.same_as_miss} "
            f"first_difference={hit_check.first_difference} "
            f"answer_correct={hit_check.answer_correct} "
            f"bad_characters={len(hit_check.bad_characters)} text={hit_check.text[:80]!r}"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
