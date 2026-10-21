"""T2 (baseline): async A then async B; measure how long B's head waits behind A.

Run inside the container while kvshrink-vllm-serve.sh is up (fixed N=4):
    python3 t2_ab_check.py --out _data/async-load-baseline-T2
"""

import argparse
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

LAYER_RE = re.compile(r"layers\.(\d+)\.")


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
        f"Sensor log #{seed}.\n"
        + "\n".join(lines)
        + "\nQuestion: Which sensor appears in Record 0007 and what did it measure? "
        "Answer in one sentence.\nAnswer:"
    )


def drain(ctrl, enable=None):
    url = f"{ctrl}/v1/cache/queue_trace?queue=OMP-Main"
    if enable is not None:
        url += f"&enable={int(enable)}"
    workers = http_get(url)["workers"]
    if len(workers) != 1:
        raise RuntimeError(f"expected 1 worker, got {list(workers)}")
    w = next(iter(workers.values()))
    if w["dropped"]:
        raise RuntimeError(f"trace dropped {w['dropped']} events")
    return w["events"]


def unzip_by_req(events):
    """{req_id: {layer: event}} for unzip events."""
    out = {}
    for e in events:
        kind, desc, key = e["label"].split("|")
        if kind != "unzip":
            continue
        layer = int(LAYER_RE.search(key).group(1))
        out.setdefault(desc, {})[layer] = e
    return out


def head_metrics(by_req, req, other, n_head):
    """B-head metrics; times in ms relative to B layer-0 enqueue."""
    b = by_req[req]
    t0 = b[0]["enqueue_ns"]
    head = [b[i] for i in range(n_head)]
    m = {
        "b0_wait_ms": (b[0]["start_ns"] - b[0]["enqueue_ns"]) / 1e6,
        "head_ready_ms": (max(e["end_ns"] for e in head) - t0) / 1e6,
        "all_ready_ms": (max(e["end_ns"] for e in b.values()) - t0) / 1e6,
        "b0_high_depth": b[0]["high_depth"],
    }
    if other is not None:
        a = by_req[other]
        ahead = [
            layer for layer, e in a.items()
            if e["start_ns"] < b[0]["start_ns"] and e["end_ns"] > t0
        ]
        m["a_tasks_run_between_b0_enqueue_and_start"] = len(ahead)
        m["a_layers_ahead"] = [min(ahead), max(ahead)] if ahead else []
        m["a_tasks_queued_before_b0"] = sum(
            1 for e in a.values() if e["enqueue_ns"] < t0 and e["start_ns"] > t0
        )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8010")
    ap.add_argument("--ctrl", default="http://localhost:18710")
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--records", type=int, default=220)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--delays-ms", default="0,20,50")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def save(name, obj):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def body(seed):
        return {
            "model": args.model,
            "prompt": build_prompt(args.records, seed),
            "max_tokens": 64,
            "temperature": 0,
            "seed": 0,
        }

    bodies = {"A": body(11), "B": body(22)}
    save("requests.json", bodies)

    def internal_id(resp, by_req):
        ids = [d for d in by_req if d.startswith(resp["id"] + "-")]
        if len(ids) != 1:
            raise RuntimeError(f"cannot map {resp['id']} to trace ids {list(by_req)}")
        return ids[0]

    # 1) warm-up: both miss -> stored; outputs are the correctness reference.
    ref = {k: http_post(f"{args.api}/v1/completions", b) for k, b in bodies.items()}
    save("ref_responses.json", ref)
    time.sleep(5)

    drain(args.ctrl, enable=True)
    results = {"n_head": args.n_head, "trials": []}

    # 2) B alone (hit): baseline for B's head latency without contention.
    rb = http_post(f"{args.api}/v1/completions", bodies["B"])
    time.sleep(2)
    ev = drain(args.ctrl)
    by_req = unzip_by_req(ev)
    b_id = internal_id(rb, by_req)
    results["solo_B"] = {
        "req_id": b_id,
        "metrics": head_metrics(by_req, b_id, None, args.n_head),
        "text_identical": rb["choices"][0]["text"] == ref["B"]["choices"][0]["text"],
    }
    save("trace_solo_B.json", ev)

    # 3) A then B after a delay, both hit.
    for d in [int(x) for x in args.delays_ms.split(",")]:
        resp = {}

        def send(k):
            resp[k] = http_post(f"{args.api}/v1/completions", bodies[k])

        ta = threading.Thread(target=send, args=("A",))
        tb = threading.Thread(target=send, args=("B",))
        ta.start()
        time.sleep(d / 1000)
        tb.start()
        ta.join()
        tb.join()
        time.sleep(2)
        ev = drain(args.ctrl)
        save(f"trace_AB_delay{d}ms.json", ev)
        by_req = unzip_by_req(ev)
        a_id, b_id = internal_id(resp["A"], by_req), internal_id(resp["B"], by_req)
        if by_req[a_id][0]["enqueue_ns"] > by_req[b_id][0]["enqueue_ns"]:
            a_id, b_id = b_id, a_id  # name by queue arrival, not send order
        results["trials"].append({
            "delay_ms": d,
            "first_in_queue": a_id,
            "second_in_queue": b_id,
            "same_submit_batch": abs(
                by_req[b_id][0]["enqueue_ns"] - by_req[a_id][63]["enqueue_ns"]
            ) < 5e6,
            "metrics_second": head_metrics(by_req, b_id, a_id, args.n_head),
            "text_identical": {
                k: resp[k]["choices"][0]["text"] == ref[k]["choices"][0]["text"]
                for k in "AB"
            },
        })
        time.sleep(1)

    drain(args.ctrl, enable=False)
    save("report.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
