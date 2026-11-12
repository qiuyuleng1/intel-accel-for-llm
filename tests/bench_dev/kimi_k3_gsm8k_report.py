# SPDX-License-Identifier: Apache-2.0
"""Report of one kimi_k3_gsm8k.sh result directory.

Per pass (config, few-shot count, pass1/pass2):
- accuracy (evalscope review score), garbled outputs (U+FFFD or a control character
  other than newline, tab, carriage return), prompt token range;
- vLLM counters over the pass (/metrics after minus before, warm-up free since
  evalscope sends none): GPU prefix cache hit / query tokens, external hit tokens;
- kvshrink only: every request's external hit from the connector log (last line per
  request), its source, and whether the answer is correct. Source "junction" means
  the hit length equals a shared_prefix_boundary an earlier request logged on the same
  vLLM (the hit uses a mamba state saved where a chunk was stopped at that boundary);
  "chunk_end" is any other hit; "none" is no hit.

Usage: python3 gsm8k_correctness_report.py <results dir>
"""

import glob
import json
import re
import sys
from dataclasses import dataclass

CONNECTOR_LINE = re.compile(
    r"get_num_new_matched_tokens, req-(\S+?), externally-cached tokens: (\d+),"
)
BOUNDARY = re.compile(r"shared_prefix_boundary=(\d+)")
BASE_ID = re.compile(r"((?:chat)?cmpl-[0-9a-f]+)")
COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)
ALLOWED_CONTROL_CHARS = "\n\t\r"


@dataclass
class Answer:
    """One question's evalscope result."""
    index: str
    correct: bool
    extracted: str
    target: str
    garbled: bool
    response_id: str = ""
    input_tokens: int = 0


@dataclass
class Hit:
    """One request's external hit from the connector log."""
    tokens: int
    source: str


def is_garbled(text: str) -> bool:
    if "\ufffd" in text:
        return True
    for char in text:
        if ord(char) < 32 and char not in ALLOWED_CONTROL_CHARS:
            return True
    return False


def read_jsonl(pattern: str) -> list[dict]:
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise ValueError(f"expected one file for {pattern}, found {paths}")
    records = []
    with open(paths[0]) as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_answers(pass_dir: str) -> dict[str, Answer]:
    answers = {}
    for review in read_jsonl(f"{pass_dir}/*/reviews/*/gsm8k*.jsonl"):
        score = review["sample_score"]["score"]
        index = str(review["index"])
        # evalscope 1.14.1: value is keyed by main_score_name ("accuracy"), target is a list.
        answers[index] = Answer(
            index=index,
            correct=float(score["value"][score["main_score_name"]]) == 1.0,
            extracted=str(score["extracted_prediction"]),
            target=",".join(review["target"]),
            garbled=is_garbled(str(score["prediction"])),
        )
    for prediction in read_jsonl(f"{pass_dir}/*/predictions/*/gsm8k*.jsonl"):
        index = str(prediction["index"])
        output = prediction["model_output"]
        answers[index].response_id = output["id"]
        answers[index].input_tokens = int(output["usage"]["input_tokens"])
    return answers


def read_counters(path: str) -> dict[str, float]:
    totals = {}
    for name in COUNTERS:
        totals[name] = 0.0
    with open(path) as metrics_file:
        for line in metrics_file:
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("{")[0].split(" ")[0]
            if name in totals:
                totals[name] += float(line.split()[-1])
    return totals


def read_hits(pass_dir: str, boundaries_seen: set[int]) -> dict[str, Hit]:
    """Response id -> Hit. Adds the boundaries this pass logged to boundaries_seen."""
    hits = {}
    with open(f"{pass_dir}/connector_log.txt") as log_file:
        for line in log_file:
            match = CONNECTOR_LINE.search(line)
            if not match:
                continue
            base = BASE_ID.search(match.group(1))
            if not base:
                raise ValueError(f"no response id in connector line: {line.strip()}")
            tokens = int(match.group(2))
            if tokens > 0 and tokens in boundaries_seen:
                source = "junction"
            elif tokens > 0:
                source = "chunk_end"
            else:
                source = "none"
            # A request asked again before it is scheduled logs again; the last line counts.
            hits[base.group(1)] = Hit(tokens, source)
            boundary = BOUNDARY.search(line)
            if boundary and int(boundary.group(1)) > 0:
                boundaries_seen.add(int(boundary.group(1)))
    return hits


def percent(part: int, whole: int) -> str:
    if whole == 0:
        return "-"
    return f"{100.0 * part / whole:.1f}%"


def main() -> None:
    results_dir = sys.argv[1]
    run_order = []
    with open(f"{results_dir}/run_order.txt") as order_file:
        for line in order_file:
            if line.strip():
                run_order.append(line.split())

    summary_rows = []
    hit_rows = []
    wrong_with_hit = []
    boundaries_by_config = {}
    for config, shots, pass_name in run_order:
        pass_dir = f"{results_dir}/{config}/{shots}/{pass_name}"
        if not glob.glob(f"{pass_dir}/*/reviews/*/gsm8k*.jsonl"):
            summary_rows.append(f"{config:12} {shots:8} {pass_name:5}  no evalscope review (pass failed)")
            continue
        answers = read_answers(pass_dir)
        before = read_counters(f"{pass_dir}/metrics_before.txt")
        after = read_counters(f"{pass_dir}/metrics_after.txt")
        gpu_hits = after["vllm:prefix_cache_hits_total"] - before["vllm:prefix_cache_hits_total"]
        gpu_queries = after["vllm:prefix_cache_queries_total"] - before["vllm:prefix_cache_queries_total"]
        external_hits = (
            after["vllm:external_prefix_cache_hits_total"]
            - before["vllm:external_prefix_cache_hits_total"]
        )
        correct = 0
        garbled = 0
        min_tokens = 0
        max_tokens = 0
        for answer in answers.values():
            if answer.correct:
                correct += 1
            if answer.garbled:
                garbled += 1
            if min_tokens == 0 or answer.input_tokens < min_tokens:
                min_tokens = answer.input_tokens
            if answer.input_tokens > max_tokens:
                max_tokens = answer.input_tokens
        gpu_ratio = f"{gpu_hits:.0f}/{gpu_queries:.0f} ({percent(int(gpu_hits), int(gpu_queries))})"
        summary_rows.append(
            f"{config:12} {shots:8} {pass_name:5} {len(answers):5} "
            f"{min_tokens:>6}-{max_tokens:<6} {percent(correct, len(answers)):>7} "
            f"{garbled:7} {gpu_ratio:>28} {external_hits:14.0f}"
        )

        if config != "kvshrink":
            continue
        if config not in boundaries_by_config:
            boundaries_by_config[config] = set()
        hits = read_hits(pass_dir, boundaries_by_config[config])
        # (tokens, source) -> [requests, correct]
        groups = {}
        for answer in answers.values():
            if answer.response_id not in hits:
                raise ValueError(f"{pass_dir}: no connector line for {answer.response_id}")
            hit = hits[answer.response_id]
            key = (hit.tokens, hit.source)
            if key not in groups:
                groups[key] = [0, 0]
            groups[key][0] += 1
            if answer.correct:
                groups[key][1] += 1
            elif hit.tokens > 0:
                wrong_with_hit.append(
                    f"{shots} {pass_name} question {answer.index}: hit {hit.tokens} "
                    f"({hit.source}), answer {answer.extracted}, expected {answer.target}"
                )
        for key in sorted(groups, reverse=True):
            requests, right = groups[key]
            hit_rows.append(
                f"{shots:8} {pass_name:5} {key[0]:>7} {key[1]:10} {requests:8} "
                f"{right:7} {percent(right, requests):>8}"
            )

    print("== Per pass")
    print(
        f"{'config':12} {'shots':8} {'pass':5} {'qs':>5} {'prompt tokens':13} "
        f"{'acc':>7} {'garbled':>7} {'GPU prefix hit/query (vLLM)':>28} {'external hit':>14}"
    )
    for row in summary_rows:
        print(row)
    print("")
    print("== kvshrink: requests by external hit (connector log, last line per request)")
    print(f"{'shots':8} {'pass':5} {'hit':>7} {'source':10} {'requests':>8} {'correct':>7} {'acc':>8}")
    for row in hit_rows:
        print(row)
    print("")
    print(f"== kvshrink: wrong answers with an external hit ({len(wrong_with_hit)})")
    for row in wrong_with_hit:
        print(row)


if __name__ == "__main__":
    main()
