# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 4 client: send the same long prompts to a vLLM server for several rounds, one request at
a time, temperature 0, and save each reply. Round 1 stores the KV cache through KVShrink; later
rounds of the same prompt load it back. Standard library only, so it runs on the host."""

import argparse
import json
import urllib.request

SENTENCES_PER_PROMPT = 600


def build_prompt(prompt_index: int) -> str:
    lines = [f"Inventory report {prompt_index}."]
    for item in range(SENTENCES_PER_PROMPT):
        value = (prompt_index * 7919 + item * 104729) % 100000
        lines.append(f"Record {prompt_index}-{item}: item {item} of report {prompt_index} has value {value}.")
    lines.append(f"Question: what is the value of item 17 of report {prompt_index}? Answer:")
    return " ".join(lines)


def send(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    replies = []
    for round_index in range(1, args.rounds + 1):
        for prompt_index in range(args.prompts):
            reply = send(args.port, args.model, build_prompt(prompt_index), args.max_tokens)
            record = {
                "round": round_index,
                "prompt_index": prompt_index,
                "response_id": reply["id"],
                "prompt_tokens": reply["usage"]["prompt_tokens"],
                "text": reply["choices"][0]["text"],
            }
            replies.append(record)
            print(
                f"round {round_index} prompt {prompt_index} id {record['response_id']} "
                f"prompt_tokens {record['prompt_tokens']} text {record['text']!r}",
                flush=True,
            )
    with open(args.output, "w") as output_file:
        json.dump(replies, output_file, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
