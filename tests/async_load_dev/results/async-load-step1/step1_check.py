"""Step-1 end-to-end check: TaskQueue trace + DDR hit correctness.

Run inside the container while kvshrink-vllm-serve.sh is up:
    python3 step1_check.py --out _data/async-load-step1
"""

import argparse
import json
import os
import random
import statistics
import time
import urllib.request

def http_get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def http_post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def build_prompt(n_records, seed):
    rng = random.Random(seed)
    lines = [
        f"Record {i:04d}: sensor {rng.choice('ABCDEFGH')}{rng.randint(10, 99)} "
        f"measured {rng.randint(1000, 9999)} units at hour {rng.randint(0, 23)}."
        for i in range(n_records)
    ]
    return (
        "Below is a sensor log.\n"
        + "\n".join(lines)
        + "\nQuestion: Which sensor appears in Record 0007 and what did it measure? "
        "Answer in one sentence.\nAnswer:"
    )


def drain(ctrl, enable=None):
    url = f"{ctrl}/v1/cache/queue_trace?queue=OMP-Main"
    if enable is not None:
        url += f"&enable={int(enable)}"
    return http_get(url)["workers"]


def summarize(events):
    """Group events by kind; report count, distinct desc/layers, wait/run ms."""
    out = {}
    for kind in sorted({e["label"].split("|")[0] for e in events}):
        evs = [e for e in events if e["label"].split("|")[0] == kind]
        waits = [(e["start_ns"] - e["enqueue_ns"]) / 1e6 for e in evs]
        runs = [(e["end_ns"] - e["start_ns"]) / 1e6 for e in evs]
        out[kind] = {
            "count": len(evs),
            "descs": sorted({e["label"].split("|")[1] for e in evs}),
            "num_layers": len({e["label"].split("|")[2] for e in evs}),
            "priorities": sorted({e["priority"] for e in evs}),
            "wait_ms_min_med_max": [min(waits), statistics.median(waits), max(waits)],
            "run_ms_min_med_max": [min(runs), statistics.median(runs), max(runs)],
            "max_high_depth": max(e["high_depth"] for e in evs),
            "max_low_depth": max(e["low_depth"] for e in evs),
            "ts_monotonic": all(
                e["enqueue_ns"] <= e["start_ns"] <= e["end_ns"] for e in evs
            ),
        }
    return out


def bad_chars(text):
    return [
        hex(ord(c))
        for c in text
        if c == "\ufffd" or (ord(c) < 0x20 and c not in "\n\t\r")
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8010")
    ap.add_argument("--ctrl", default="http://localhost:18710")
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--records", type=int, default=220)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def save(name, obj):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    body = {
        "model": args.model,
        "prompt": build_prompt(args.records, seed=1234),
        "max_tokens": 64,
        "temperature": 0,
        "seed": 0,
    }
    save("request.json", body)

    drain(args.ctrl, enable=True)  # discard stale events, start tracing

    t0 = time.monotonic()
    r1 = http_post(f"{args.api}/v1/completions", body)
    t1 = time.monotonic()
    save("response_1_miss.json", r1)
    time.sleep(5)  # let async zip_to_mem (put) finish
    trace1 = drain(args.ctrl)
    save("trace_1_miss.json", trace1)

    t2 = time.monotonic()
    r2 = http_post(f"{args.api}/v1/completions", body)
    t3 = time.monotonic()
    save("response_2_hit.json", r2)
    time.sleep(2)
    trace2 = drain(args.ctrl, enable=False)
    save("trace_2_hit.json", trace2)

    text1 = r1["choices"][0]["text"]
    text2 = r2["choices"][0]["text"]
    report = {
        "prompt_tokens": r1["usage"]["prompt_tokens"],
        "completion_tokens": [r1["usage"]["completion_tokens"], r2["usage"]["completion_tokens"]],
        "wall_s": [round(t1 - t0, 3), round(t3 - t2, 3)],
        "text_identical": text1.encode() == text2.encode(),
        "bad_chars": [bad_chars(text1), bad_chars(text2)],
        "text_1": text1,
        "trace_1_miss": {
            rank: {"dropped": w["dropped"], "summary": summarize(w["events"])}
            for rank, w in trace1.items()
        },
        "trace_2_hit": {
            rank: {"dropped": w["dropped"], "summary": summarize(w["events"])}
            for rank, w in trace2.items()
        },
    }
    save("report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
