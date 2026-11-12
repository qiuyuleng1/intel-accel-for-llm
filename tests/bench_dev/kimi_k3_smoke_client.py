# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 + kvshrink smoke: does it answer, and are answers right after a kvshrink hit.

Requests, one at a time (temperature 0, max_tokens 2048, default chat template):
  short-1  one-sentence fact question (correct if the answer names Paris)
  short-2  gsm8k test question 1 without examples
  long-A   SHOTS gsm8k train examples + test question 2 (stores the KV)
  long-A2  the same prompt again (hits up to its last full block)
  long-B   the same examples + test question 3 (hits the shared examples)
  long-C   the same examples + test question 4
  xlong-A  LONG_SHOTS examples (longer than one 16384-token chunk) + test question 5
  xlong-A2 the same prompt again
  xlong-B  the same LONG_SHOTS examples + test question 6
Answers are checked against gsm8k targets and for garbled text (U+FFFD or a control
character other than newline, tab, carriage return). The report joins each request
with its kvshrink connector line (externally-cached tokens).

  run:    python3 kimi_k3_smoke_client.py run --port 8000 --model M --dataset-dir D --out F
  report: python3 kimi_k3_smoke_client.py report --requests F --connector-log L --block-size B
"""

import argparse
import glob
import json
import re
import urllib.request
from dataclasses import asdict, dataclass

from datasets import load_from_disk

SHOTS = 60
LONG_SHOTS = 120
MAX_TOKENS = 2048
INSTRUCTION = (
    "Solve the last question. Reason step by step like the examples and end your "
    "answer with '#### <number>'."
)
FINAL_ANSWER = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
ANY_NUMBER = re.compile(r"-?[\d,]*\.?\d+")
CONNECTOR_LINE = re.compile(
    r"get_num_new_matched_tokens, req-(\S+?), externally-cached tokens: (\d+),"
)
BASE_ID = re.compile(r"((?:chat)?cmpl-[0-9a-f]+)")
ALLOWED_CONTROL_CHARS = "\n\t\r"


@dataclass
class SmokeResult:
    """One request and its answer."""
    name: str
    response_id: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    target: str
    extracted: str
    correct: bool
    garbled: bool
    text: str


def find_split(dataset_dir: str, split: str):
    """The gsm8k split saved under dataset_dir/datasets (evalscope's cache)."""
    for state_path in glob.glob(f"{dataset_dir}/datasets/*/state.json"):
        with open(state_path) as state_file:
            if json.load(state_file)["_split"] == split:
                return load_from_disk(state_path.rsplit("/", 1)[0])
    raise ValueError(f"no gsm8k {split} split under {dataset_dir}")


def final_number(answer_text: str) -> str:
    matches = FINAL_ANSWER.findall(answer_text)
    if not matches:
        matches = ANY_NUMBER.findall(answer_text)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def same_number(left: str, right: str) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-6
    except ValueError:
        return False


def is_garbled(text: str) -> bool:
    if "\ufffd" in text:
        return True
    for char in text:
        if ord(char) < 32 and char not in ALLOWED_CONTROL_CHARS:
            return True
    return False


def ask(port: int, model: str, prompt: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def build_examples_text(train, shots: int) -> str:
    examples = []
    for train_index in range(shots):
        record = train[train_index]
        examples.append(f"Question: {record['question']}\nAnswer: {record['answer']}")
    return INSTRUCTION + "\n\n" + "\n\n".join(examples)


def run(args: argparse.Namespace) -> None:
    train = find_split(args.dataset_dir, "train")
    test = find_split(args.dataset_dir, "test")
    examples_text = build_examples_text(train, SHOTS)
    long_examples_text = build_examples_text(train, LONG_SHOTS)

    # (name, prompt, target): an empty target means the answer must name Paris.
    plan = [
        ("short-1", "What is the capital of France? Answer in one short sentence.", ""),
        (
            "short-2",
            f"{test[0]['question']}\nEnd your answer with '#### <number>'.",
            final_number(test[0]["answer"]),
        ),
    ]
    for name, test_index in (("long-A", 1), ("long-A2", 1), ("long-B", 2), ("long-C", 3)):
        record = test[test_index]
        prompt = f"{examples_text}\n\nQuestion: {record['question']}\nAnswer:"
        plan.append((name, prompt, final_number(record["answer"])))
    for name, test_index in (("xlong-A", 4), ("xlong-A2", 4), ("xlong-B", 5)):
        record = test[test_index]
        prompt = f"{long_examples_text}\n\nQuestion: {record['question']}\nAnswer:"
        plan.append((name, prompt, final_number(record["answer"])))

    results = []
    for name, prompt, target in plan:
        response = ask(args.port, args.model, prompt)
        choice = response["choices"][0]
        text = choice["message"]["content"] or ""
        if target:
            extracted = final_number(text)
            correct = same_number(extracted, target)
        else:
            extracted = text.strip()[:40]
            correct = "Paris" in text
        result = SmokeResult(
            name=name,
            response_id=response["id"],
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            finish_reason=choice["finish_reason"],
            target=target,
            extracted=extracted,
            correct=correct,
            garbled=is_garbled(text),
            text=text,
        )
        results.append(result)
        print(
            f"{name}: prompt={result.prompt_tokens} completion={result.completion_tokens} "
            f"finish={result.finish_reason} answer={extracted!r} target={target!r} "
            f"correct={correct} garbled={result.garbled}",
            flush=True,
        )
    with open(args.out, "w") as out_file:
        for result in results:
            out_file.write(json.dumps(asdict(result)) + "\n")


def report(args: argparse.Namespace) -> None:
    hit_by_id = {}
    with open(args.connector_log) as log_file:
        for line in log_file:
            match = CONNECTOR_LINE.search(line)
            if match:
                hit_by_id[BASE_ID.search(match.group(1)).group(1)] = int(match.group(2))
    print(f"{'request':8} {'prompt':>6} {'hit':>6} {'max':>6} {'compl':>5} {'finish':6} {'correct':7} garbled  answer / target")
    with open(args.requests) as requests_file:
        for line in requests_file:
            result = SmokeResult(**json.loads(line))
            hit = hit_by_id.get(result.response_id, -1)
            # A hit covers whole blocks and never the last prompt token.
            max_hit = (result.prompt_tokens - 1) // args.block_size * args.block_size
            print(
                f"{result.name:8} {result.prompt_tokens:>6} {hit:>6} {max_hit:>6} "
                f"{result.completion_tokens:>5} "
                f"{result.finish_reason:6} {str(result.correct):7} {str(result.garbled):7}  "
                f"{result.extracted!r} / {result.target!r}"
            )
    print("(hit -1: no connector line for the request)")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--port", type=int, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--dataset-dir", required=True)
    run_parser.add_argument("--out", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--requests", required=True)
    report_parser.add_argument("--connector-log", required=True)
    report_parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
