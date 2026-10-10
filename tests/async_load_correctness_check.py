# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Output-correctness check for async KV load (doc/design/async-load-priority.zh-CN.md).

1. Send one long prompt (cache miss) -> reference output.
2. Send the same prompt CONCURRENCY times at once (cache hits, loaded asynchronously).
3. Every hit output must be byte-identical to the reference and contain no U+FFFD /
   illegal control characters; the connector log must show externally-cached > 0
   for every hit request.

Greedy decoding (temperature 0, fixed seed). The server MUST run with
VLLM_BATCH_INVARIANT=1: otherwise the reference (batch of 1) and the hits (mixed
batches) use different kernel reductions and greedy outputs may diverge without any
load-path bug. Writes request, responses, connector log
lines and report.json to --out. Exit code 0 only if all checks pass.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request


def build_prompt(records: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = [
        f"Record {i:04d}: sensor {rng.choice('ABCDEFGH')}{rng.randint(10, 99)} "
        f"measured {rng.randint(1000, 9999)} units."
        for i in range(records)
    ]
    return ("Sensor log.\n" + "\n".join(lines)
            + "\nQuestion: Which sensor appears in Record 0042 and what did it measure?"
            "\nAnswer:")


def post(api: str, body: dict) -> dict:
    req = urllib.request.Request(api, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def bad_chars(text: str) -> list[str]:
    return [hex(ord(c)) for c in text
            if c == "\ufffd" or (ord(c) < 32 and c not in "\n\t\r")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8010/v1/completions")
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--vllm-log", default="log.kvshrink-vllm")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--records", type=int, default=260)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Byte offset, not line count: the log contains '\r' progress output.
    log_start = os.path.getsize(args.vllm_log)

    # Fresh prompt per run so the reference is a real cache miss.
    body = {"model": args.model, "prompt": build_prompt(args.records, int(time.time())),
            "max_tokens": 64, "temperature": 0, "seed": 0}
    ref = post(args.api, body)
    time.sleep(5)  # let the reference's KV be stored
    hits: list[dict] = [{} for _ in range(args.concurrency)]

    def run(i: int) -> None:
        hits[i] = post(args.api, body)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(2)

    with open(args.vllm_log, "rb") as f:
        f.seek(log_start)
        text = f.read().decode(errors="replace")
    log_lines = [l + "\n" for l in text.split("\n")
                 if "get_num_new_matched_tokens" in l or "async_load step" in l]
    ext = {}
    for line in log_lines:
        m = re.search(r"req-(\S+), externally-cached tokens: (\d+)", line)
        if m:
            ext[m.group(1)] = int(m.group(2))

    def ext_of(resp_id: str) -> int:
        vals = [v for k, v in ext.items() if k.startswith(resp_id + "-")]
        return vals[0] if len(vals) == 1 else -1

    ref_text = ref["choices"][0]["text"]
    report = {
        "prompt_tokens": ref["usage"]["prompt_tokens"],
        "reference": {"id": ref["id"], "externally_cached": ext_of(ref["id"]),
                      "bad_chars": bad_chars(ref_text), "text": ref_text},
        "hits": [{"id": h["id"], "externally_cached": ext_of(h["id"]),
                  "identical": h["choices"][0]["text"].encode() == ref_text.encode(),
                  "bad_chars": bad_chars(h["choices"][0]["text"])} for h in hits],
    }
    ok = (report["reference"]["externally_cached"] == 0 and not report["reference"]["bad_chars"]
          and all(h["externally_cached"] > 0 and h["identical"]
                  and not h["bad_chars"] for h in report["hits"]))
    report["verdict"] = "PASS" if ok else "FAIL"

    with open(os.path.join(args.out, "responses.json"), "w") as f:
        json.dump({"request": body, "reference": ref, "hits": hits}, f, indent=1)
    with open(os.path.join(args.out, "connector_log.txt"), "w") as f:
        f.writelines(log_lines)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
