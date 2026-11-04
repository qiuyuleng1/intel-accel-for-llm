# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVShrinkConnector that measures every call vLLM makes into it, for run_cprofile_bench.sh.

vLLM loads it with --kv-transfer-config '{"kv_connector": "ProfiledKVShrinkConnector",
"kv_connector_module_path": "profiled_kvshrink_connector", ...}' and this directory on PYTHONPATH.
kvshrink itself is not changed: every method below calls the real one and only measures around it.

The run script switches the measurement with a one-line control file, PROFILE_CONTROL_DIR/control:
  "<seq> start timers"    count the calls of each entry method and add up their wall time
  "<seq> start cprofile"  the same, and run one cProfile.Profile per entry method during its calls
  "<seq> stop"            write the results of the running mode and stop measuring
The EngineCore main thread reads the file at the start of an entry call, at most every 0.2 s, so a
command takes effect with the next request. Results go to PROFILE_CONTROL_DIR/profile_<mode>.json
and, for cprofile, cprofile_<entry method>.pstats; every handled command is appended to control.log.

The methods of IAXL's extension classes (Context, Record, Mem, Storage) are replaced with Python
functions named IAXL_<class>.<method> that call the original. cProfile in Python 3.12 records a C
call only when the callable is a built-in function or method descriptor (_lsprof.c,
get_cfunc_from_callable); a pybind11 method reached through an instance is a bound method, so its
C++ time could otherwise land in the Python time of the calling line.
"""

import cProfile
import json
import os
import pstats
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from iaxl import torch_ext
from kvshrink.kvshrink_connector import KVShrinkConnector

CONTROL_DIR = os.environ["PROFILE_CONTROL_DIR"]
CONTROL_FILE = os.path.join(CONTROL_DIR, "control")
CONTROL_LOG = os.path.join(CONTROL_DIR, "control.log")
CONTROL_CHECK_INTERVAL_S = 0.2
# Calls per measurement of the cost that cProfile adds to one profiled call (see
# measure_cprofile_cost_ns); enough for a stable mean, short enough to stay under 0.1 s.
COST_LOOPS = 200000
COST_REPEATS = 3
COST_SAMPLE_LIST = [0, 1, 2]
IAXL_CLASS_NAMES = ("Context", "Record", "Mem", "Storage")
# analyze_cprofile.py counts every function whose name starts with this as IAXL C++.
IAXL_WRAPPER_PREFIX = "IAXL_"


@dataclass
class EntryTotals:
    calls: int = 0
    wall_ns: int = 0


@dataclass
class MeasureState:
    mode: str = "off"  # off | timers | cprofile
    # Entry methods currently running: request_finished_all_groups calls request_finished, and the
    # inner call belongs to the outer one.
    depth: int = 0
    next_check: float = 0.0
    last_seq: int = 0
    totals: Dict[str, EntryTotals] = field(default_factory=dict)
    profilers: Dict[str, cProfile.Profile] = field(default_factory=dict)


# One state for the whole process: with TP=1 the scheduler-side and the worker-side connector are
# two objects in the same EngineCore main thread, and their calls belong to the same measurement.
STATE = MeasureState()


def make_iaxl_wrapper(label: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def call_iaxl(*args, **kwargs):
        return original(*args, **kwargs)

    # cProfile tells functions apart by their code object, so each IAXL method gets its own name.
    call_iaxl.__code__ = call_iaxl.__code__.replace(co_name=label, co_qualname=label)
    return call_iaxl


def wrap_iaxl_classes() -> None:
    for class_name in IAXL_CLASS_NAMES:
        iaxl_class = getattr(torch_ext, class_name)
        for attribute_name, attribute in list(vars(iaxl_class).items()):
            if attribute_name.startswith("_"):
                continue
            label = f"{IAXL_WRAPPER_PREFIX}{class_name}.{attribute_name}"
            if isinstance(attribute, staticmethod):
                wrapper = make_iaxl_wrapper(label, attribute.__func__)
                setattr(iaxl_class, attribute_name, staticmethod(wrapper))
            elif type(attribute).__name__ == "instancemethod":
                # A plain function on the class is bound to the instance like the original was.
                setattr(iaxl_class, attribute_name, make_iaxl_wrapper(label, attribute.__func__))


wrap_iaxl_classes()


def log_control(text: str) -> None:
    with open(CONTROL_LOG, "a") as file:
        file.write(f"{time.strftime('%H:%M:%S')} {text}\n")


def empty_function() -> None:
    return None


def run_python_calls(loops: int) -> None:
    for _ in range(loops):
        empty_function()


def run_c_calls(loops: int) -> None:
    for _ in range(loops):
        len(COST_SAMPLE_LIST)


def own_time_ns(profiler: cProfile.Profile) -> float:
    # Sum of the time cProfile charges to the measured functions, without the disable() call that
    # ends the measurement.
    stats = pstats.Stats(profiler).stats
    total = 0.0
    for key, values in stats.items():
        if "_lsprof.Profiler" in key[2]:
            continue
        total += values[2]
    return total * 1e9


def measure_one_cost_ns(run: Callable[[int], None]) -> float:
    best_extra = None
    for _ in range(COST_REPEATS):
        start = time.perf_counter_ns()
        run(COST_LOOPS)
        plain_ns = time.perf_counter_ns() - start
        profiler = cProfile.Profile()
        profiler.enable()
        run(COST_LOOPS)
        profiler.disable()
        extra = (own_time_ns(profiler) - plain_ns) / COST_LOOPS
        if best_extra is None or extra < best_extra:
            best_extra = extra
    return best_extra


def measure_cprofile_cost_ns() -> Dict[str, float]:
    """Time cProfile adds to the profiled total per recorded call, measured in this process.

    The analysis subtracts it from the Python time of each entry method: cProfile spends this
    much on every call it records, and almost all of kvshrink's recorded calls are short ones.
    """
    return {
        "python_call_ns": measure_one_cost_ns(run_python_calls),
        "c_call_ns": measure_one_cost_ns(run_c_calls),
    }


def start_measuring(mode: str) -> None:
    if mode not in ("timers", "cprofile"):
        raise ValueError(f"unknown measurement mode {mode}")
    STATE.mode = mode
    STATE.totals = {}
    STATE.profilers = {}


def stop_measuring() -> None:
    mode = STATE.mode
    if mode == "off":
        return
    STATE.mode = "off"
    entries: Dict[str, Dict[str, int]] = {}
    for name, totals in STATE.totals.items():
        entries[name] = {"calls": totals.calls, "wall_ns": totals.wall_ns}
    results: Dict[str, Any] = {"mode": mode, "entries": entries}
    if mode == "cprofile":
        for name, profiler in STATE.profilers.items():
            profiler.dump_stats(os.path.join(CONTROL_DIR, f"cprofile_{name}.pstats"))
        results["cprofile_cost_ns"] = measure_cprofile_cost_ns()
    with open(os.path.join(CONTROL_DIR, f"profile_{mode}.json"), "w") as file:
        json.dump(results, file, indent=2)
    STATE.totals = {}
    STATE.profilers = {}


def poll_control() -> None:
    try:
        with open(CONTROL_FILE) as file:
            words = file.read().split()
    except FileNotFoundError:
        return
    if not words:
        return
    seq = int(words[0])
    if seq <= STATE.last_seq:
        return
    STATE.last_seq = seq
    if words[1] == "start":
        start_measuring(words[2])
        log_control(f"{seq} started {words[2]}")
    elif words[1] == "stop":
        mode = STATE.mode
        stop_measuring()
        log_control(f"{seq} stopped {mode}")
    else:
        raise ValueError(f"unknown control command {' '.join(words)}")


def measured_call(name: str, method: Callable[..., Any], args: tuple, kwargs: Dict[str, Any]) -> Any:
    if STATE.depth == 0:
        now = time.monotonic()
        if now >= STATE.next_check:
            STATE.next_check = now + CONTROL_CHECK_INTERVAL_S
            poll_control()
    if STATE.mode == "off" or STATE.depth > 0:
        STATE.depth += 1
        try:
            return method(*args, **kwargs)
        finally:
            STATE.depth -= 1

    profiler: Optional[cProfile.Profile] = None
    if STATE.mode == "cprofile":
        profiler = STATE.profilers.get(name)
        if profiler is None:
            profiler = cProfile.Profile()
            STATE.profilers[name] = profiler
    STATE.depth += 1
    if profiler is not None:
        profiler.enable()
    start = time.perf_counter_ns()
    try:
        return method(*args, **kwargs)
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        if profiler is not None:
            profiler.disable()
        STATE.depth -= 1
        totals = STATE.totals.get(name)
        if totals is None:
            totals = EntryTotals()
            STATE.totals[name] = totals
        totals.calls += 1
        totals.wall_ns += elapsed_ns


class ProfiledKVShrinkConnector(KVShrinkConnector):
    """KVShrinkConnector whose vLLM-facing methods are measured by measured_call."""

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return measured_call(
            "get_num_new_matched_tokens", super().get_num_new_matched_tokens,
            (request, num_computed_tokens), {},
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return measured_call(
            "update_state_after_alloc", super().update_state_after_alloc,
            (request, blocks, num_external_tokens), {},
        )

    def request_finished(self, request, block_ids):
        return measured_call(
            "request_finished", super().request_finished, (request, block_ids), {},
        )

    def request_finished_all_groups(self, request, block_ids):
        return measured_call(
            "request_finished_all_groups", super().request_finished_all_groups,
            (request, block_ids), {},
        )

    def build_connector_meta(self, scheduler_output):
        return measured_call(
            "build_connector_meta", super().build_connector_meta, (scheduler_output,), {},
        )

    def start_load_kv(self, forward_context, **kwargs):
        return measured_call(
            "start_load_kv", super().start_load_kv, (forward_context,), kwargs,
        )

    def wait_for_layer_load(self, layer_name):
        return measured_call(
            "wait_for_layer_load", super().wait_for_layer_load, (layer_name,), {},
        )

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return measured_call(
            "save_kv_layer", super().save_kv_layer, (layer_name, kv_layer, attn_metadata), kwargs,
        )

    def wait_for_save(self):
        return measured_call("wait_for_save", super().wait_for_save, (), {})

    def get_finished(self, finished_req_ids):
        return measured_call(
            "get_finished", super().get_finished, (finished_req_ids,), {},
        )
