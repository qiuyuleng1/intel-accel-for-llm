# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check OMP-Main queue order against the expected async-load scheme behavior.

Inputs (doc/design/async-load-priority.zh-CN.md section 7.5):
  * queue trace JSON from ``GET /v1/cache/queue_trace`` (controller or worker)
  * vLLM log containing the connector ``get_num_new_matched_tokens`` and
    ``async_load step=...`` lines of the same run

Outputs (section 7.6): submission-structure errors, order inversions and
scenario coverage counts. Verdict and exit code:
  PASS (0)      structure errors == 0, inversions == 0, scenario covered
  FAIL (1)      structure errors > 0 or inversions > 0
  UNCOVERED (2) no errors, but the load did not produce the scenario
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
ACTIVATE_RE = re.compile(r"async_load step=(\d+) t=(\d+) activate reqs=(\S*) held=")

SPLIT_FIRST_N_BATCH_REQS = "split_async_load_submit_first_n_batch_reqs"
SPLIT_FIRST_N_PER_REQ = "split_async_load_submit_first_n_per_req"
SPLIT_SCHEMES = (SPLIT_FIRST_N_BATCH_REQS, SPLIT_FIRST_N_PER_REQ)
FIFO_SCHEMES = ("naive", "batch_reqs_async_load_submit") + SPLIT_SCHEMES
PRIORITY_FIRST_N_PER_REQ = "priority_3level_first_n_per_req"
PRIORITY_SCHEMES = (PRIORITY_FIRST_N_PER_REQ,)
# OMP-Main levels used by the connector (kvshrink_connector.LOAD_PRIORITY_*).
P0, P1, P2 = 0, 1, 2


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
    """Return (req_info, submissions, promote_ns, promote_step, activate_ns)."""
    req_info: dict[str, dict] = {}
    submissions: list[Submission] = []
    promote_ns: dict[str, int] = {}
    promote_step: dict[str, int] = {}
    activate_ns: dict[str, int] = {}
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
                step, t, reqs = m.groups()
                for r in reqs.split(","):
                    promote_ns.setdefault(r, int(t))
                    promote_step.setdefault(r, int(step))
                continue
            m = ACTIVATE_RE.search(line)
            if m:
                _, t, reqs = m.groups()
                for r in reqs.split(","):
                    if r:
                        activate_ns.setdefault(r, int(t))
    return req_info, submissions, promote_ns, promote_step, activate_ns


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
    # split submits a request's head and tail with the same description but
    # disjoint layer ranges, so match a task by description AND layer.
    by_desc: dict[str, list[Submission]] = defaultdict(list)
    for s in subs:
        by_desc[s.desc].append(s)
    for desc, ss in by_desc.items():
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if a.first_layer <= b.last_layer and b.first_layer <= a.last_layer:
                    errors.append(f"description {desc!r} used by submissions with "
                                  "overlapping layers")
    for e in events:
        if e["kind"] not in ("unzip", "unzip_retry"):
            continue
        ss = [s for s in by_desc.get(e["desc"], [])
              if s.first_layer <= e["layer"] <= s.last_layer]
        if not ss:
            errors.append(f"unzip task {e['label']} has no submit log line")
            continue
        s = ss[0]
        if e["kind"] == "unzip":
            if e["layer"] in s.tasks:
                errors.append(f"duplicate task for layer {e['layer']} in {s.desc}")
            s.tasks[e["layer"]] = e


def head_layers(info: dict, num_layers: int) -> int:
    return num_layers if info["n"] == -1 else min(info["n"], num_layers)


def check_structure(scheme, subs, req_info, events, errors):
    """Scheme-independent checks + per-scheme submission rules."""
    num_layers = max(s.last_layer for s in subs) + 1 if subs else 0
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
        want = {"async"} if info["async"] else {"sync"}
        if scheme in SPLIT_SCHEMES and info["async"] and head_layers(info, num_layers) < num_layers:
            want = {"async", "async_remaining"}
        if kinds != want:
            errors.append(f"{r}: expected {sorted(want)} submits, got {sorted(kinds)}")

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
        elif scheme in SPLIT_SCHEMES:
            pass  # checked by check_split()
        elif scheme in PRIORITY_SCHEMES:
            for s in asyncs:
                if len(s.reqs) != 1:
                    errors.append(f"step {step}: per-request async submit has "
                                  f"{len(s.reqs)} reqs")
        else:
            raise SystemExit(f"structure rules for scheme {scheme!r} not implemented")


def check_split(scheme, subs, req_info, promote_ns, promote_step, errors):
    """split_async_load_submit_* rules (doc section 9.3)."""
    num_layers = max(s.last_layer for s in subs) + 1 if subs else 0
    steps: dict[int, list[tuple[int, Submission]]] = defaultdict(list)
    for i, s in enumerate(subs):
        steps[s.step].append((i, s))
    for step, ss in steps.items():
        # Tails of promoted requests are submitted before new heads.
        tails = [i for i, s in ss if s.kind == "async_remaining"]
        heads = [i for i, s in ss if s.kind == "async"]
        if tails and heads and max(tails) > min(heads):
            errors.append(f"step {step}: a tail submitted after a head")
        # Remaining layers of one step always form one batch: one segment per
        # layer. First N layers do too under first_n_batch_reqs; under
        # first_n_per_req every first-N submission holds exactly one request.
        kinds_one_segment_per_layer = ["async_remaining"]
        if scheme == SPLIT_FIRST_N_BATCH_REQS:
            kinds_one_segment_per_layer.append("async")
        else:
            for _, s in ss:
                if s.kind == "async" and len(s.reqs) != 1:
                    errors.append(f"step {step}: first-N submit has {len(s.reqs)} reqs")
        for kind in kinds_one_segment_per_layer:
            layers = [j for _, s in ss if s.kind == kind
                      for j in range(s.first_layer, s.last_layer + 1)]
            if len(layers) != len(set(layers)):
                errors.append(f"step {step}: {kind} segments overlap")

    by_req: dict[str, list[Submission]] = defaultdict(list)
    for s in subs:
        for r in s.reqs:
            by_req[r].append(s)
    for r, info in req_info.items():
        if info["ext_tokens"] == 0 or not info["async"]:
            continue
        h = head_layers(info, num_layers)
        for kind, want in (("async", range(0, h)), ("async_remaining", range(h, num_layers))):
            ss = [s for s in by_req.get(r, []) if s.kind == kind]
            layers = sorted(j for s in ss for j in range(s.first_layer, s.last_layer + 1))
            if layers != list(want):
                errors.append(f"{r}: {kind} layers {layers[:1]}..{layers[-1:]} "
                              f"({len(layers)}), expected {want}")
            if len({s.step for s in ss}) > 1:
                errors.append(f"{r}: {kind} submitted in several steps")
            if kind == "async_remaining" and ss:
                if r not in promote_step:
                    errors.append(f"{r}: tail submitted but never promoted")
                    continue
                if ss[0].step != promote_step[r] + 1:
                    errors.append(f"{r}: tail at step {ss[0].step}, promoted at "
                                  f"step {promote_step[r]}")
                if any(t["enqueue_ns"] < promote_ns[r]
                       for s in ss for t in s.tasks.values()):
                    errors.append(f"{r}: tail task enqueued before promotion")


def check_priority(subs, req_info, promote_ns, activate_ns, errors):
    """priority_3level_first_n_per_req rules (doc sections 4.3, 8.4).

    Per task: submit level (sync P0; async first N layers P1, the rest P2), group
    (sync 0; one distinct non-zero group per async request), and raise to P0 only
    for tail tasks, after the request's promotion and no later than its activation.
    """
    num_layers = max(s.last_layer for s in subs) + 1 if subs else 0
    group_owner: dict[int, str] = {}
    for s in subs:
        for layer, t in s.tasks.items():
            where = f"step {s.step} {s.kind} {s.desc} layer {layer}"
            if s.kind == "sync":
                if t["submit_priority"] != P0 or t["group"] != 0:
                    errors.append(f"{where}: sync task submit_priority "
                                  f"{t['submit_priority']} group {t['group']}")
                continue
            r = s.reqs[0]
            h = head_layers(req_info.get(r, {"n": -1}), num_layers)
            want = P1 if layer < h else P2
            if t["submit_priority"] != want:
                errors.append(f"{where}: submit_priority {t['submit_priority']}, "
                              f"expected {want}")
            if t["group"] == 0:
                errors.append(f"{where}: async task has group 0")
            elif group_owner.setdefault(t["group"], r) != r:
                errors.append(f"{where}: group {t['group']} shared with "
                              f"{group_owner[t['group']]}")
            raised = t["p0_enter_ns"] > 0
            if raised != (t["priority"] == P0 and t["submit_priority"] != P0):
                errors.append(f"{where}: priority {t['priority']} inconsistent with "
                              f"p0_enter_ns {t['p0_enter_ns']}")
            if not raised:
                continue
            if want != P2:
                errors.append(f"{where}: first-N task raised to P0")
            if t["p0_enter_ns"] < promote_ns.get(r, 1 << 62):
                errors.append(f"{where}: raised before promotion")
            if r not in activate_ns or t["p0_enter_ns"] > activate_ns[r]:
                errors.append(f"{where}: raised later than its activate log line")


def sort_key(scheme: str, e: dict):
    if scheme in FIFO_SCHEMES:
        return (e["priority"], e["seq"])
    raise SystemExit(f"order key for scheme {scheme!r} not implemented")


def level_key_at(e: dict, at_ns: int):
    """(level, time it entered that level, seq) of task e at time at_ns.

    A task raised by set_group_priority sits in its submit level until
    p0_enter_ns and in P0 (behind everything already there) afterwards.
    """
    if e.get("p0_enter_ns", 0) and e["p0_enter_ns"] <= at_ns:
        return (P0, e["p0_enter_ns"], e["seq"])
    return (e.get("submit_priority", e["priority"]), e["enqueue_ns"], e["seq"])


def count_inversions(scheme: str, events: list[dict]):
    """Tasks that started while a smaller-key task was already queued."""
    inversions = []
    for e in events:
        if scheme in PRIORITY_SCHEMES:
            ke = level_key_at(e, e["start_ns"])
        else:
            ke = sort_key(scheme, e)
        for f in events:
            if not f["enqueue_ns"] <= e["start_ns"] < f["start_ns"]:
                continue
            if scheme in PRIORITY_SCHEMES:
                kf = level_key_at(f, e["start_ns"])
            else:
                kf = sort_key(scheme, f)
            if kf < ke:
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
                    "first_layer": s.first_layer,
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

    req_info, subs, promote_ns, promote_step, activate_ns = parse_log(args.log)
    events = load_trace(args.trace)
    errors: list[str] = []
    attach_tasks(subs, events, errors)
    check_structure(args.scheme, subs, req_info, events, errors)
    if args.scheme in SPLIT_SCHEMES:
        check_split(args.scheme, subs, req_info, promote_ns, promote_step, errors)
    if args.scheme in PRIORITY_SCHEMES:
        check_priority(subs, req_info, promote_ns, activate_ns, errors)
    inversions = count_inversions(args.scheme, events)
    cov = coverage(subs, req_info, promote_ns)

    steps_multi_async = sum(
        1 for st in {s.step for s in subs}
        if len({r for s in subs if s.step == st and s.kind == "async" for r in s.reqs}) >= 2
    )
    # Only the submission carrying a request's layer 0 represents its arrival.
    behind_async = [c for c in cov if c["first_layer"] == 0 and
                    any(k.startswith("async") for k in c["ahead_from_earlier_steps"])]
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
    if errors or inversions:
        verdict, code = "FAIL", 1
    elif covered == 0:
        verdict, code = "UNCOVERED", 2
    else:
        verdict, code = "PASS", 0
    report["verdict"] = verdict
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_submission", "inversion_samples")}, indent=2))
    print(verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
