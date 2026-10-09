# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check OMP-Main queue order against the expected async-load scheme behavior.

Inputs (doc/design/async-load-priority.zh-CN.md section 7.5):
  * queue trace JSON from ``GET /v1/cache/queue_trace`` (controller or worker)
  * vLLM log containing the connector ``get_num_new_matched_tokens`` and
    ``async_load step=...`` lines of the same run

Outputs (section 7.6): submission-structure errors, order inversions and
scenario coverage counts. Exit code 0 only if structure errors == 0,
inversions == 0 and the requested scenario was covered at least once.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

LAYER_RE = re.compile(r"layers\.(\d+)\.")
MATCH_RE = re.compile(
    r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+),"
    r".*async=(True|False),.*async_load_layers=(-?\d+)"
)
SUBMIT_RE = re.compile(
    r"async_load step=(\d+) t=(\d+) submit kind=(\w+) reqs=(\S+) layers=(\d+)-(\d+)"
)
PROMOTE_RE = re.compile(r"async_load step=(\d+) t=(\d+) promote reqs=(\S+)")

FIFO_SCHEMES = ("naive", "batch_reqs_async_load_submit")


@dataclass
class Submission:
    step: int
    t_ns: int
    kind: str
    reqs: tuple[str, ...]
    first_layer: int
    last_layer: int
    tasks: dict[int, dict] = field(default_factory=dict)

    @property
    def desc(self) -> str:
        return ",".join(self.reqs)


def parse_log(path: str):
    """Return (req_info, submissions, promote_ns)."""
    req_info: dict[str, dict] = {}
    submissions: list[Submission] = []
    promote_ns: dict[str, int] = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = MATCH_RE.search(line)
            if m:
                req, ext, is_async, n = m.groups()
                req_info[req] = {
                    "ext_tokens": int(ext),
                    "async": is_async == "True",
                    "n": int(n),
                }
                continue
            m = SUBMIT_RE.search(line)
            if m:
                step, t, kind, reqs, a, b = m.groups()
                submissions.append(
                    Submission(int(step), int(t), kind, tuple(reqs.split(",")),
                               int(a), int(b))
                )
                continue
            m = PROMOTE_RE.search(line)
            if m:
                _, t, reqs = m.groups()
                for r in reqs.split(","):
                    promote_ns.setdefault(r, int(t))
    return req_info, submissions, promote_ns


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if "workers" in data:
        workers = data["workers"]
        if len(workers) != 1:
            raise SystemExit(f"expected 1 worker (TP=1), got {sorted(workers)}")
        data = next(iter(workers.values()))
    if data.get("dropped"):
        raise SystemExit(f"trace dropped {data['dropped']} events; result invalid")
    events = data["events"]
    for e in events:
        kind, desc, key = e["label"].split("|", 2) if e["label"] else ("", "", "")
        e["kind"], e["desc"] = kind, desc
        m = LAYER_RE.search(key)
        e["layer"] = int(m.group(1)) if m else -1
    return events


def attach_tasks(subs: list[Submission], events: list[dict], errors: list[str]):
    by_desc: dict[str, list[Submission]] = defaultdict(list)
    for s in subs:
        by_desc[s.desc].append(s)
    for desc, ss in by_desc.items():
        if len(ss) > 1:
            errors.append(f"description {desc!r} used by {len(ss)} submissions")
    for e in events:
        if e["kind"] not in ("unzip", "unzip_retry"):
            continue
        ss = by_desc.get(e["desc"])
        if not ss:
            errors.append(f"unzip task {e['label']} has no submit log line")
            continue
        s = ss[0]
        if e["kind"] == "unzip":
            if e["layer"] in s.tasks:
                errors.append(f"duplicate task for layer {e['layer']} in {s.desc}")
            s.tasks[e["layer"]] = e


def check_structure(scheme, subs, req_info, events, errors):
    """Scheme-independent checks + per-scheme submission rules."""
    for s in subs:
        want = set(range(s.first_layer, s.last_layer + 1))
        if set(s.tasks) != want:
            errors.append(
                f"step {s.step} {s.kind} {s.desc}: layers {sorted(set(s.tasks) ^ want)}"
                " missing or unexpected in trace"
            )
        seqs = [s.tasks[j]["seq"] for j in sorted(s.tasks)]
        if seqs != sorted(seqs):
            errors.append(f"step {s.step} {s.desc}: tasks not enqueued in layer order")

    # Every hit request must be loaded exactly once, on the path the log chose.
    loaded: dict[str, list[Submission]] = defaultdict(list)
    for s in subs:
        for r in s.reqs:
            loaded[r].append(s)
    for r, info in req_info.items():
        if info["ext_tokens"] == 0:
            continue
        ss = loaded.get(r, [])
        kinds = {s.kind for s in ss}
        want_kind = "async" if info["async"] else "sync"
        if kinds != {want_kind}:
            errors.append(f"{r}: expected {want_kind} submit, got {sorted(kinds)}")

    steps: dict[int, list[Submission]] = defaultdict(list)
    for s in subs:
        steps[s.step].append(s)
    for step, ss in steps.items():
        asyncs = [s for s in ss if s.kind == "async"]
        if scheme == "naive":
            for s in asyncs:
                if len(s.reqs) != 1:
                    errors.append(f"step {step}: naive async submit has {len(s.reqs)} reqs")
        elif scheme == "batch_reqs_async_load_submit":
            if len(asyncs) > 1:
                errors.append(f"step {step}: {len(asyncs)} async submits, expected 1")
        else:
            raise SystemExit(f"structure rules for scheme {scheme!r} not implemented")


def sort_key(scheme: str, e: dict):
    if scheme in FIFO_SCHEMES:
        return (e["priority"], e["seq"])
    raise SystemExit(f"order key for scheme {scheme!r} not implemented")


def count_inversions(scheme: str, events: list[dict]):
    """Tasks that started while a smaller-key task was already queued."""
    inversions = []
    for e in events:
        ke = sort_key(scheme, e)
        for f in events:
            if (f["enqueue_ns"] <= e["start_ns"] < f["start_ns"]
                    and sort_key(scheme, f) < ke):
                inversions.append({"ran": e["label"], "seq": e["seq"],
                                   "waiting": f["label"], "waiting_seq": f["seq"]})
                break
    return inversions


def classify(task, s, req_info, promote_ns, at_ns):
    """Category of a queued task at time at_ns, for the coverage report."""
    if s.kind == "sync":
        return "sync"
    n = min(req_info.get(r, {}).get("n", -1) for r in s.reqs)
    if n < 0 or task["layer"] < n:
        return "async_head"
    promoted = all(promote_ns.get(r, 1 << 62) <= at_ns for r in s.reqs)
    return "async_tail_promoted" if promoted else "async_tail_unpromoted"


def coverage(subs, req_info, promote_ns):
    """For each submission, what was queued ahead of its first task."""
    all_tasks = [(t, s) for s in subs for t in s.tasks.values()]
    out = []
    for s in subs:
        if not s.tasks:
            continue
        head = s.tasks[min(s.tasks)]
        ahead = defaultdict(int)
        for t, ts in all_tasks:
            if ts.step < s.step and t["enqueue_ns"] <= head["enqueue_ns"] < t["start_ns"]:
                ahead[classify(t, ts, req_info, promote_ns, head["enqueue_ns"])] += 1
        out.append({"step": s.step, "kind": s.kind, "reqs": len(s.reqs),
                    "first_task_wait_ms": (head["start_ns"] - head["enqueue_ns"]) / 1e6,
                    "ahead_from_earlier_steps": dict(ahead)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--scenario", choices=("s1", "s2"), required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    req_info, subs, promote_ns = parse_log(args.log)
    events = load_trace(args.trace)
    errors: list[str] = []
    attach_tasks(subs, events, errors)
    check_structure(args.scheme, subs, req_info, events, errors)
    inversions = count_inversions(args.scheme, events)
    cov = coverage(subs, req_info, promote_ns)

    steps_multi_async = sum(
        1 for st in {s.step for s in subs}
        if sum(len(s.reqs) for s in subs if s.step == st and s.kind == "async") >= 2
    )
    behind_async = [c for c in cov if any(k.startswith("async") for k in c["ahead_from_earlier_steps"])]
    a_cases = [c for c in behind_async if c["kind"] == "async"]
    b_cases = [c for c in behind_async if c["kind"] == "sync"]
    covered = len(a_cases) if args.scenario == "s1" else len(b_cases)

    hit = [i for i in req_info.values() if i["ext_tokens"] > 0]
    report = {
        "scheme": args.scheme,
        "scenario": args.scenario,
        "hit_requests": len(hit),
        "hit_async": sum(1 for i in hit if i["async"]),
        "hit_sync": sum(1 for i in hit if not i["async"]),
        "submissions": len(subs),
        "trace_events": len(events),
        "structure_errors": errors,
        "inversions": len(inversions),
        "inversion_samples": inversions[:10],
        "steps_with_multiple_async_reqs": steps_multi_async,
        "case_a_async_behind_earlier_async": len(a_cases),
        "case_b_sync_behind_earlier_async": len(b_cases),
        "per_submission": cov,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    ok = not errors and not inversions and covered > 0
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_submission", "inversion_samples")}, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
