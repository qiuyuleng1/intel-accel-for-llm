# SPDX-License-Identifier: Apache-2.0
"""Send one ~4000-token completion request twice to a running vLLM and check it.

The prompt is RECORDS lines "Record N: the warehouse in city N stores 7N boxes
of apples and 3N boxes of pears." followed by a question about one record, so a
correct answer is known (7 * ASKED_RECORD). Both requests: temperature 0, seed
0, MAX_TOKENS tokens. Saved in <out dir>: request.json, response_1.json,
response_2.json. Printed per response: prompt tokens (as the server counted
them), finish reason, whether the text has U+FFFD or a control character other
than newline/tab/carriage return, whether it contains the right number, and the
text; then whether both texts are byte-identical.

Usage: python3 rust_kvshrink_smoke_client.py <out dir> [port]
"""

import json
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "/home/intel/model-space/Qwen3.8-27B"
TARGET_PROMPT_TOKENS = 4000
ASKED_RECORD = 42
MAX_TOKENS = 48
ALLOWED_CONTROL_CHARS = "\n\t\r"


def build_prompt(tokenizer):
    question = (
        f"\nQuestion: How many boxes of apples does the warehouse in city {ASKED_RECORD} "
        "store? Answer with the number only.\nAnswer:"
    )
    lines = []
    record = 1
    while True:
        line = (
            f"Record {record}: the warehouse in city {record} stores {7 * record} boxes "
            f"of apples and {3 * record} boxes of pears.\n"
        )
        candidate = "".join(lines) + line + question
        if len(tokenizer(candidate)["input_ids"]) > TARGET_PROMPT_TOKENS:
            break
        lines.append(line)
        record += 1
    return "".join(lines) + question


def is_garbled(text):
    if "\ufffd" in text:
        return True
    for char in text:
        if ord(char) < 32 and char not in ALLOWED_CONTROL_CHARS:
            return True
    return False


def send(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main():
    out_dir = sys.argv[1]
    port = 8000
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = build_prompt(tokenizer)
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
    }
    with open(f"{out_dir}/request.json", "w") as request_file:
        json.dump(body, request_file)
    print(f"prompt: {len(tokenizer(prompt)['input_ids'])} tokens (HF tokenizer), "
          f"expected answer {7 * ASKED_RECORD}")
    texts = []
    for attempt in (1, 2):
        start = time.perf_counter()
        response = send(port, body)
        seconds = time.perf_counter() - start
        with open(f"{out_dir}/response_{attempt}.json", "w") as response_file:
            json.dump(response, response_file)
        text = response["choices"][0]["text"]
        texts.append(text)
        print(
            f"response {attempt}: id {response['id']}, {seconds:.2f} s, "
            f"prompt tokens {response['usage']['prompt_tokens']}, "
            f"finish {response['choices'][0]['finish_reason']}, "
            f"garbled {is_garbled(text)}, has {7 * ASKED_RECORD}: {str(7 * ASKED_RECORD) in text}"
        )
        print(f"  text: {text!r}")
        # Let the first request's KV reach kvshrink's DDR pool before the second.
        time.sleep(3)
    print(f"byte-identical: {texts[0] == texts[1]}")


if __name__ == "__main__":
    main()
