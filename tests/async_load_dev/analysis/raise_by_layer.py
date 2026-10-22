# SPDX-License-Identifier: Apache-2.0
"""Per-activation stats of P0 raises in a §7 queue-test trace (ad-hoc, §11.15).

One raise = all unzip tasks sharing one p0_enter_ns (set_groups_priority stamps
every task it moves with the same time). For each raise: groups moved, whether
the sync batch (submit_priority 0) is in it, and whether the tasks *executed*
in non-decreasing layer order.

usage (repo root): python3 tests/async_load_dev/analysis/raise_by_layer.py \
    _data/queue-test/<run> [...]
"""

import json
import sys
from collections import defaultdict


def load_events(run_dir):
    with open(f"{run_dir}/trace.json") as f:
        data = json.load(f)
    if "workers" in data:
        data = next(iter(data["workers"].values()))
    return data["events"]


def layer_of(ev):
    return int(ev["label"].rsplit("layers.", 1)[1].split(".", 1)[0])


def main():
    print(
        "run | raises | multi_group | with_sync | tasks | "
        "groups max | layer_ordered | switches multi (mean)"
    )
    for run_dir in sys.argv[1:]:
        raises = defaultdict(list)
        for ev in load_events(run_dir):
            if ev["label"].startswith("unzip|") and ev["p0_enter_ns"] > 0:
                raises[ev["p0_enter_ns"]].append(ev)
        multi = 0
        with_sync = 0
        tasks = 0
        groups_max = 0
        ordered = 0
        switches_multi = []
        for evs in raises.values():
            evs.sort(key=lambda e: e["start_ns"])
            groups = {e["group"] for e in evs}
            tasks += len(evs)
            groups_max = max(groups_max, len(groups))
            if any(e["submit_priority"] == 0 for e in evs):
                with_sync += 1
            layers = [layer_of(e) for e in evs]
            keys = [e["order_key"] for e in evs]
            if keys != layers:
                raise SystemExit(f"{run_dir}: order_key != layer")
            if layers == sorted(layers):
                ordered += 1
            if len(groups) > 1:
                multi += 1
                sw = sum(
                    1 for a, b in zip(evs, evs[1:]) if a["group"] != b["group"]
                )
                switches_multi.append(sw)
        mean_sw = sum(switches_multi) / len(switches_multi) if switches_multi else 0
        print(
            f"{run_dir.rstrip('/').rsplit('/', 1)[-1]} | {len(raises)} | {multi} | "
            f"{with_sync} | {tasks} | {groups_max} | {ordered}/{len(raises)} | "
            f"{mean_sw:.1f}"
        )


if __name__ == "__main__":
    main()
