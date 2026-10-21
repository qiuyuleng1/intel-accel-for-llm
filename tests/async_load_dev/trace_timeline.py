# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-request load timeline from an OMP-Main queue trace (profiling helper).

Usage:
    python3 tests/async_load_dev/trace_timeline.py <tag>.trace.json.gz \
        <label>.serve.log [--ranks 0,6] [--csv out.csv]

Inputs come from a TRACE=1 run of perf-qwen72b-bench.sh: the queue trace of all
ranks for one cell (``GET /v1/cache/queue_trace`` through the controller) and the
server log of the same server, which has the connector ``get_num_new_matched_tokens``
lines (scheduler) and the ``async_load ... promote / activate`` lines of every
worker (KVSHRINK_ASYNC_LOAD_EVENT_LOG=1).

For every async request found in the trace, per rank (all times in ms):
    head_wait     first enqueue -> first N layers finished
    to_promote    first N layers finished -> promote (get_finished)
    to_activate   promote -> activate (K3 moves it into the forward)
    left_layers   layers of the request not finished at activate
    fwd_wait      activate -> last layer finished (0 if already done)
    load_total    first enqueue -> last layer finished
For priority_3level_*: tail tasks run in P2 (prefetched before activate) vs
raised to P0, and the estimated set_group_priority cost from the number of tasks
waiting in P1..P3 at each raise (37 ns per waiting task, measured by
iaxl/csrc/test/task_queue_bench.cpp).
Worker overhead between two tasks when the next one was already queued (traces
with release_ns): release = task returned -> its callable and captures destroyed,
pick = destroyed -> next task started.
"""

import argparse
import csv
import gzip
import json
import re
import statistics
from collections import defaultdict

LAYER_RE = re.compile(r"layers\.(\d+)\.")
MATCH_RE = re.compile(
    r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+),"
    r".*async=(True|False),.*async_load_layers=(-?\d+)"
)
WORKER_RE = re.compile(r"\(Worker_TP(\d+) ")
PROMOTE_RE = re.compile(r"async_load step=\d+ t=(\d+) promote reqs=(\S+)")
ACTIVATE_RE = re.compile(r"async_load step=\d+ t=(\d+) activate reqs=(\S*) held=")
NS_PER_WAITING_TASK = 37.0
P0, P1, P2, P3 = 0, 1, 2, 3
MS = 1e6


def load_trace(path: str) -> dict[int, list[dict]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    workers = data["workers"] if "workers" in data else {"0": data}
    by_rank: dict[int, list[dict]] = {}
    for rank, payload in workers.items():
        if "events" not in payload:
            raise SystemExit(f"rank {rank}: no events ({payload.get('error')})")
        if payload.get("dropped"):
            raise SystemExit(f"rank {rank}: trace dropped {payload['dropped']} events")
        events = payload["events"]
        for event in events:
            kind, description, tensor_key = "", "", ""
            if event["label"]:
                kind, description, tensor_key = event["label"].split("|", 2)
            event["kind"] = kind
            event["reqs"] = description.split(",") if description else []
            layer_match = LAYER_RE.search(tensor_key)
            event["layer"] = int(layer_match.group(1)) if layer_match else -1
        by_rank[int(rank)] = events
    return by_rank


def parse_log(path: str):
    """Return (async N per request, promote_ns[rank][req], activate_ns[rank][req])."""
    async_first_n: dict[str, int] = {}
    promote: dict[int, dict[str, int]] = defaultdict(dict)
    activate: dict[int, dict[str, int]] = defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            match = MATCH_RE.search(line)
            if match:
                req_id, _, is_async, num_first_n_layers = match.groups()
                if is_async == "True":
                    async_first_n[req_id] = int(num_first_n_layers)
                continue
            worker_match = WORKER_RE.search(line)
            # TP=1 runs the worker inside EngineCore: no Worker_TP prefix.
            rank = int(worker_match.group(1)) if worker_match else 0
            match = PROMOTE_RE.search(line)
            if match:
                for req_id in match.group(2).split(","):
                    promote[rank].setdefault(req_id, int(match.group(1)))
                continue
            match = ACTIVATE_RE.search(line)
            if match:
                for req_id in match.group(2).split(","):
                    if req_id:
                        activate[rank].setdefault(req_id, int(match.group(1)))
    return async_first_n, promote, activate


def request_rows(rank, events, async_first_n, promote, activate):
    tasks_by_req: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if event["kind"] not in ("unzip", "unzip_retry"):
            continue
        for req_id in event["reqs"]:
            if req_id in async_first_n:
                tasks_by_req[req_id].append(event)
    rows = []
    for req_id, tasks in tasks_by_req.items():
        num_first_n_layers = async_first_n[req_id]
        first_enqueue_ns = tasks[0]["enqueue_ns"]
        last_done_ns = tasks[0]["end_ns"]
        first_n_done_ns = 0
        tail_in_p2 = 0
        tail_raised = 0
        for task in tasks:
            first_enqueue_ns = min(first_enqueue_ns, task["enqueue_ns"])
            last_done_ns = max(last_done_ns, task["end_ns"])
            # N = -1 waits for all layers, so every layer counts as first-N.
            if num_first_n_layers < 0 or task["layer"] < num_first_n_layers:
                first_n_done_ns = max(first_n_done_ns, task["end_ns"])
            if task["submit_priority"] == P2 and task["priority"] == P2:
                tail_in_p2 += 1
            if task["submit_priority"] == P2 and task["priority"] == P0:
                tail_raised += 1
        promote_ns = promote[rank].get(req_id)
        activate_ns = activate[rank].get(req_id)

        row = {"rank": rank, "req": req_id, "n": num_first_n_layers,
               "head_wait": (first_n_done_ns - first_enqueue_ns) / MS,
               "to_promote": None, "to_activate": None, "left_layers": None,
               "fwd_wait": None, "load_total": (last_done_ns - first_enqueue_ns) / MS,
               "tail_p2": tail_in_p2, "tail_raised": tail_raised}
        if promote_ns:
            row["to_promote"] = (promote_ns - first_n_done_ns) / MS
        if promote_ns and activate_ns:
            row["to_activate"] = (activate_ns - promote_ns) / MS
        if activate_ns:
            layers_left = set()
            for task in tasks:
                if task["end_ns"] > activate_ns:
                    layers_left.add(task["layer"])
            row["left_layers"] = len(layers_left)
            row["fwd_wait"] = max(0, last_done_ns - activate_ns) / MS
        rows.append(row)
    return rows


def raise_costs(events):
    """Waiting P1..P3 tasks at each raise moment, from tasks seen in the trace."""
    raise_times = set()
    for event in events:
        if event["p0_enter_ns"]:
            raise_times.add(event["p0_enter_ns"])
    costs = []
    for raise_ns in sorted(raise_times):
        waiting = 0
        for event in events:
            if not (event["enqueue_ns"] <= raise_ns < event["start_ns"]):
                continue
            # A task raised at or before this moment already sits in P0.
            if event["p0_enter_ns"] and event["p0_enter_ns"] <= raise_ns:
                continue
            if event["submit_priority"] in (P1, P2, P3):
                waiting += 1
        costs.append(waiting)
    return costs


def summarize(values):
    present = []
    for value in values:
        if value is not None:
            present.append(value)
    if not present:
        return "n/a"
    present.sort()
    p95_index = max(0, int(len(present) * 0.95) - 1)
    return (f"mean={statistics.fmean(present):8.1f} "
            f"med={present[len(present) // 2]:8.1f} "
            f"p95={present[p95_index]:8.1f} n={len(present)}")


def column(rows, key):
    values = []
    for row in rows:
        values.append(row[key])
    return values


def worker_overhead(events):
    """(run_us, release_us, pick_us) per task whose successor was already queued."""
    ordered = sorted(events, key=lambda event: event["start_ns"])
    run_us, release_us, pick_us = [], [], []
    for index in range(len(ordered) - 1):
        task = ordered[index]
        successor = ordered[index + 1]
        if successor["enqueue_ns"] >= task["end_ns"]:
            continue  # the worker may have been idle waiting for work
        run_us.append((task["end_ns"] - task["start_ns"]) / 1000)
        release_us.append((task["release_ns"] - task["end_ns"]) / 1000)
        pick_us.append((successor["start_ns"] - task["release_ns"]) / 1000)
    return run_us, release_us, pick_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("log")
    parser.add_argument("--ranks", default="", help="comma list, default all")
    parser.add_argument("--csv", default="", help="write per-request rows here")
    args = parser.parse_args()

    by_rank = load_trace(args.trace)
    async_first_n, promote, activate = parse_log(args.log)
    ranks = sorted(by_rank)
    if args.ranks:
        ranks = []
        for rank_text in args.ranks.split(","):
            ranks.append(int(rank_text))
    all_rows = []
    for rank in ranks:
        events = by_rank[rank]
        rows = request_rows(rank, events, async_first_n, promote, activate)
        all_rows.extend(rows)
        busy_ns = 0
        first_start_ns = events[0]["start_ns"]
        last_end_ns = events[0]["end_ns"]
        for event in events:
            busy_ns += event["end_ns"] - event["start_ns"]
            first_start_ns = min(first_start_ns, event["start_ns"])
            last_end_ns = max(last_end_ns, event["end_ns"])
        span_ns = last_end_ns - first_start_ns
        print(f"== rank {rank}: tasks={len(events)} async_reqs={len(rows)} "
              f"queue_busy={busy_ns / MS:.0f} ms of {span_ns / MS:.0f} ms")
        if "release_ns" in events[0]:
            run_us, release_us, pick_us = worker_overhead(events)
            print(f"   worker, next task already queued (us): "
                  f"run {summarize(run_us)}")
            print(f"      release {summarize(release_us)}")
            print(f"      pick    {summarize(pick_us)}")
        for key in ("head_wait", "to_promote", "to_activate", "left_layers",
                    "fwd_wait", "load_total"):
            print(f"   {key:12s} {summarize(column(rows, key))}")
        tail_in_p2 = sum(column(rows, "tail_p2"))
        tail_raised = sum(column(rows, "tail_raised"))
        if not (tail_in_p2 or tail_raised):
            continue
        print(f"   tail tasks run in P2 (prefetched)={tail_in_p2} "
              f"raised to P0={tail_raised}")
        costs = raise_costs(events)
        if not costs:
            continue
        waiting_counts = []
        estimated_us = []
        for waiting in costs:
            waiting_counts.append(float(waiting))
            estimated_us.append(waiting * NS_PER_WAITING_TASK / 1000)
        total_us = sum(estimated_us)
        print(f"   raises={len(costs)} waiting P1..P3 at raise: "
              f"{summarize(waiting_counts)}")
        print(f"   est. set_group_priority cost (us): {summarize(estimated_us)}; "
              f"total {total_us / 1000:.2f} ms = "
              f"{total_us * 1000 / span_ns * 100:.4f}% of span")
    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
