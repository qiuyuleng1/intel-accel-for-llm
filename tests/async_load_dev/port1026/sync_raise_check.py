"""Targeted check of the 1026 PR branch: an async request's remaining layers raised to P0 in the
same step as a sync batch (dev helper, not part of the PR).

Server: map 0-1:16,2-:0 (one request in flight: async with N=16; two or more: sync) and
VLLM_BATCH_INVARIANT=1, connector traced_kvshrink_connector (raise_trace lines).
1. Prompts A and B (~5k tokens each, fresh per run) are sent once each, alone: references,
   cache misses that store their KV.
2. Each trial sends A (DDR hit, alone in flight -> async N=16), then B after `delay` ms (DDR
   hit, second in flight -> sync), and waits for both. Delays are swept so that B's sync load
   sometimes lands in the step where A first enters a forward and is raised.
3. Checks: every A / B output is byte-identical to its reference and has no U+FFFD / control
   characters; references have externally-cached tokens 0 and trial requests > 0.
   Reports per trial the scheduler decision of A and B, and over the run the raise_trace lines
   with sync_batch=1 and their sync_done / sync_done_prefix (known issue K4: sync-batch layers
   already decompressed before the raise; harmless while below N=16).
Writes request bodies, all responses, the log lines used and report.json to --out.
Exit code 0 only if every output check passes.
"""

import argparse
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from async_load_correctness_check import bad_chars, build_prompt, post  # noqa: E402


def read_log(path: str, start: int) -> str:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read().decode(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8010/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vllm-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=260)
    ap.add_argument("--delays-ms", default="0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_start = os.path.getsize(args.vllm_log)

    seed = int(time.time())
    body_a = {"model": args.model, "prompt": build_prompt(args.records, seed),
              "max_tokens": 64, "temperature": 0, "seed": 0}
    body_b = {"model": args.model, "prompt": build_prompt(args.records, seed + 1),
              "max_tokens": 64, "temperature": 0, "seed": 0}
    ref_a = post(args.api, body_a)
    ref_b = post(args.api, body_b)
    time.sleep(5)  # let the references' KV be stored

    delays = []
    for value in args.delays_ms.split(","):
        delays.append(int(value))
    trials = []
    for rep in range(args.reps):
        for delay in delays:
            result = {}

            def send_a() -> None:
                result["a"] = post(args.api, body_a)

            thread = threading.Thread(target=send_a)
            thread.start()
            time.sleep(delay / 1000)
            result["b"] = post(args.api, body_b)
            thread.join()
            trials.append({"rep": rep, "delay_ms": delay, "a": result["a"], "b": result["b"]})
            time.sleep(0.5)
    time.sleep(2)

    text = read_log(args.vllm_log, log_start)
    log_lines = []
    decisions = {}
    raises = []
    decision_re = re.compile(r"req-(\S+), externally-cached tokens: (\d+), .*async=(\w+), "
                             r"selected_load_layers=(-?\d+), async_load_layers=(-?\d+)")
    raise_re = re.compile(r"raise_trace rank=(\d+) raised_reqs=(\d+) sync_batch=(\d) bundles=(\d+) "
                          r"moved=(\d+) sync_layers=(\d+) sync_done=(\d+) sync_done_prefix=(\d+)")
    for line in text.split("\n"):
        m = decision_re.search(line)
        if m:
            log_lines.append(line + "\n")
            decisions[m.group(1)] = {"external": int(m.group(2)), "async": m.group(3) == "True",
                                     "N": int(m.group(5))}
        m = raise_re.search(line)
        if m:
            log_lines.append(line + "\n")
            raises.append({"rank": int(m.group(1)), "raised_reqs": int(m.group(2)),
                           "sync_batch": int(m.group(3)), "moved": int(m.group(5)),
                           "sync_layers": int(m.group(6)), "sync_done": int(m.group(7)),
                           "sync_done_prefix": int(m.group(8))})

    def decision_of(resp_id: str) -> dict:
        found = []
        for req_id, decision in decisions.items():
            if req_id.startswith(resp_id + "-"):
                found.append(decision)
        if len(found) != 1:
            return {"external": -1, "async": None, "N": None}
        return found[0]

    ref_a_text = ref_a["choices"][0]["text"]
    ref_b_text = ref_b["choices"][0]["text"]
    report_refs = {
        "prompt_tokens": [ref_a["usage"]["prompt_tokens"], ref_b["usage"]["prompt_tokens"]],
        "external": [decision_of(ref_a["id"])["external"], decision_of(ref_b["id"])["external"]],
        "bad_chars": len(bad_chars(ref_a_text)) + len(bad_chars(ref_b_text)),
    }
    ok = report_refs["external"] == [0, 0] and report_refs["bad_chars"] == 0
    report_trials = []
    expected_pattern = 0
    for trial in trials:
        a_text = trial["a"]["choices"][0]["text"]
        b_text = trial["b"]["choices"][0]["text"]
        da = decision_of(trial["a"]["id"])
        db = decision_of(trial["b"]["id"])
        row = {
            "rep": trial["rep"], "delay_ms": trial["delay_ms"],
            "a_external": da["external"], "a_async": da["async"], "a_N": da["N"],
            "b_external": db["external"], "b_async": db["async"],
            "a_identical": a_text.encode() == ref_a_text.encode(),
            "b_identical": b_text.encode() == ref_b_text.encode(),
            "bad_chars": len(bad_chars(a_text)) + len(bad_chars(b_text)),
        }
        if row["a_async"] is True and row["a_N"] == 16 and row["b_async"] is False:
            expected_pattern += 1
        if not (row["a_identical"] and row["b_identical"] and row["bad_chars"] == 0
                and row["a_external"] > 0 and row["b_external"] > 0):
            ok = False
        report_trials.append(row)

    with_sync = []
    for entry in raises:
        if entry["sync_batch"] == 1:
            with_sync.append(entry)
    above_n = 0
    for entry in with_sync:
        if entry["sync_done_prefix"] > 16:
            above_n += 1
    report = {
        "references": report_refs,
        "trials": len(report_trials),
        "trials_a_async_N16_b_sync": expected_pattern,
        "trials_all_identical": sum(1 for row in report_trials
                                    if row["a_identical"] and row["b_identical"]),
        "raise_lines": len(raises),
        "raise_lines_with_sync_batch": len(with_sync),
        "with_sync_batch_detail": with_sync,
        "with_sync_batch_done_prefix_above_N16": above_n,
        "per_trial": report_trials,
        "verdict": "PASS" if ok else "FAIL",
    }
    with open(os.path.join(args.out, "responses.json"), "w") as f:
        json.dump({"request_a": body_a, "request_b": body_b, "reference_a": ref_a,
                   "reference_b": ref_b, "trials": trials}, f, indent=1)
    with open(os.path.join(args.out, "connector_log.txt"), "w") as f:
        f.writelines(log_lines)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    summary = {}
    for key in ("references", "trials", "trials_a_async_N16_b_sync", "trials_all_identical",
                "raise_lines", "raise_lines_with_sync_batch",
                "with_sync_batch_done_prefix_above_N16", "verdict"):
        summary[key] = report[key]
    print(json.dumps(summary, indent=1))
    for entry in with_sync:
        print("with_sync_batch:", json.dumps(entry))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
