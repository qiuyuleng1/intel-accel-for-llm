# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compares the two gsm8k passes of gsm8k_two_passes.sh for one model (dev helper).

Per pass: accuracy (evalscope's mean acc over the questions) and how much of each question's
prompt kvshrink loaded, read from the connector log lines of that pass
("externally-cached tokens") and matched to the question by the chat completion id.
Across passes: questions whose correctness flipped, and questions whose output text is
byte-identical.

usage: python3 gsm8k_compare.py <model result dir> <kvshrink block size>
"""

import glob
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass


@dataclass
class Question:
    index: int
    correct: bool
    text: str
    prompt_tokens: int
    loaded_tokens: int  # -1: no connector line found for this question


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as jsonl_file:
        for line in jsonl_file:
            rows.append(json.loads(line))
    return rows


def only_file(pattern: str) -> str:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError(f"expected one file for {pattern}, found {matches}")
    return matches[0]


def loaded_tokens_by_request(connector_log: str) -> dict[str, int]:
    # A request asked twice (not scheduled the first time) logs twice; the last one counts.
    loaded = {}
    with open(connector_log) as log_file:
        for line in log_file:
            match = re.search(r"req-(\S+), externally-cached tokens: (\d+)", line)
            if match is not None:
                loaded[match.group(1)] = int(match.group(2))
    return loaded


def read_pass(pass_dir: str) -> dict[int, Question]:
    reviews = read_jsonl(only_file(os.path.join(pass_dir, "*", "reviews", "*", "gsm8k_main.jsonl")))
    predictions = read_jsonl(
        only_file(os.path.join(pass_dir, "*", "predictions", "*", "gsm8k_main.jsonl"))
    )
    loaded = loaded_tokens_by_request(os.path.join(pass_dir, "connector_log.txt"))

    output_by_index = {}
    for prediction in predictions:
        output_by_index[int(prediction["index"])] = prediction["model_output"]

    questions = {}
    for review in reviews:
        index = int(review["index"])
        output = output_by_index[index]
        loaded_tokens = -1
        for request_id, tokens in loaded.items():
            # vLLM's request id starts with the id of the chat completion.
            if request_id.startswith(output["id"]):
                loaded_tokens = tokens
        questions[index] = Question(
            index=index,
            correct=review["sample_score"]["score"]["value"]["acc"] == 1.0,
            text=output["choices"][0]["message"]["content"],
            prompt_tokens=output["usage"]["input_tokens"],
            loaded_tokens=loaded_tokens,
        )
    return questions


def describe_loading(questions: dict[int, Question], block_size: int) -> dict:
    counts = {"whole prompt": 0, "part": 0, "none": 0, "no log line": 0}
    loaded_values = []
    for question in questions.values():
        # The connector never loads the last prompt token, nor a block it does not fill.
        whole = (question.prompt_tokens - 1) // block_size * block_size
        if question.loaded_tokens < 0:
            counts["no log line"] += 1
            continue
        loaded_values.append(question.loaded_tokens)
        if question.loaded_tokens == 0:
            counts["none"] += 1
        elif question.loaded_tokens == whole:
            counts["whole prompt"] += 1
        else:
            counts["part"] += 1
    summary = {"questions by loaded part of the prompt": counts}
    if loaded_values:
        summary["loaded tokens min/median/max"] = [
            min(loaded_values),
            statistics.median(loaded_values),
            max(loaded_values),
        ]
    return summary


def main() -> int:
    model_dir = sys.argv[1]
    block_size = int(sys.argv[2])
    first = read_pass(os.path.join(model_dir, "pass1"))
    second = read_pass(os.path.join(model_dir, "pass2"))
    if sorted(first) != sorted(second):
        raise RuntimeError("the two passes answered different questions")

    correct_to_wrong = []
    wrong_to_correct = []
    identical_outputs = 0
    for index in sorted(first):
        if first[index].correct and not second[index].correct:
            correct_to_wrong.append(index)
        if not first[index].correct and second[index].correct:
            wrong_to_correct.append(index)
        if first[index].text.encode() == second[index].text.encode():
            identical_outputs += 1

    prompt_tokens = []
    for question in first.values():
        prompt_tokens.append(question.prompt_tokens)
    summary = {
        "questions": len(first),
        "prompt tokens min/median/max": [
            min(prompt_tokens),
            statistics.median(prompt_tokens),
            max(prompt_tokens),
        ],
        "block size": block_size,
    }
    for name, questions in (("pass1", first), ("pass2", second)):
        correct = 0
        thinking_outputs = 0
        for question in questions.values():
            if question.correct:
                correct += 1
            # Thinking left on shows up as a closing think tag in the answer.
            if "</think>" in question.text:
                thinking_outputs += 1
        summary[name] = {
            "accuracy": correct / len(questions),
            "correct": correct,
            "outputs containing </think>": thinking_outputs,
            **describe_loading(questions, block_size),
        }
    summary["correct in pass1, wrong in pass2"] = correct_to_wrong
    summary["wrong in pass1, correct in pass2"] = wrong_to_correct
    summary["byte-identical outputs"] = identical_outputs

    with open(os.path.join(model_dir, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
