# SPDX-License-Identifier: Apache-2.0
"""Times kvshrink's Python code inside a running vLLM; test only, product code is unchanged.

vLLM loads this module as the connector module
(kv_transfer_config kv_connector_module_path="timed_kvshrink", with this directory on
PYTHONPATH). On import it puts a timer around
  - every method of KVShrinkConnector, ReqMeta, RequestMetadata, KVStoreLocal and KVFlow,
    and generate_block_hashs (category "python"),
  - every method of ScratchPool (category "scratch_pool"),
  - every method of the IAXL extension classes Context, Record, Mem, Storage and the two
    task_queue_* functions (category "iaxl_cpp"),
and then gives vLLM the unchanged KVShrinkConnector class.

Every timed call adds to a counter keyed by (entry function, function), where the entry
function is the outermost timed call on the thread's stack, i.e. the one vLLM called:
  calls         number of calls
  inclusive_ns  time from entering to leaving the call
  exclusive_ns  inclusive_ns minus the time of the timed calls made inside it
  child_calls   number of timed calls made directly inside it
For a "python" function, exclusive time is its own statements plus everything it calls that
is not timed (xxhash, numpy, torch, logging, vLLM helpers). "scratch_pool" and "iaxl_cpp"
calls are leaves: only their inclusive time is used.

Time is the wall clock (perf_counter_ns): it costs about 60 ns per read against 210 ns for
thread_time_ns (measured in the container), and Python statements do not block. The timer
itself adds a fixed cost to every call; it is measured once at import (see _calibrate) and
the analysis subtracts it.

SIGUSR1 writes all counters to $PYOH_TIMING_DIR/timing_<label>.json, where <label> is the
content of $PYOH_TIMING_DIR/timing_label.
"""

# Standard
import inspect
import json
import os
import signal
import statistics
import threading
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, Dict, List, Tuple

# First Party
import kvshrink.kvshrink_connector as connector_module
from iaxl import torch_ext
from iaxl.kvflow import flow as flow_module
from iaxl.kvflow import scratch_pool as scratch_pool_module
from iaxl.kvstore import kvstore as kvstore_module

PYTHON = "python"
SCRATCH_POOL = "scratch_pool"
IAXL_CPP = "iaxl_cpp"
# Runs for the life of the process on the iaxl-monitor thread; a timer around it would never
# close.
NOT_TIMED = {"KVStoreLocal._monitor_loop"}
CALIBRATION_CALLS = 200000
CALIBRATION_ROUNDS = 5


@dataclass(slots=True)
class Counter:
    category: str
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    child_calls: int = 0


@dataclass(slots=True)
class OpenCall:
    name: str
    start_ns: int = 0
    child_ns: int = 0
    child_calls: int = 0


@dataclass
class ThreadState:
    thread_name: str
    is_main: bool
    stack: List[OpenCall] = field(default_factory=list)
    # (entry function, function) -> Counter
    counters: Dict[Tuple[str, str], Counter] = field(default_factory=dict)


_local = threading.local()
_all_states: List[ThreadState] = []
_all_states_lock = threading.Lock()
_calibration: Dict[str, float] = {}
_snapshot_ns: List[int] = []


def _thread_state() -> ThreadState:
    state = getattr(_local, "state", None)
    if state is None:
        thread = threading.current_thread()
        state = ThreadState(thread_name=thread.name, is_main=thread is threading.main_thread())
        _local.state = state
        with _all_states_lock:
            _all_states.append(state)
    return state


def _close_call(state: ThreadState, call: OpenCall, category: str, end_ns: int) -> None:
    inclusive_ns = end_ns - call.start_ns
    stack = state.stack
    if stack:
        parent = stack[-1]
        parent.child_ns += inclusive_ns
        parent.child_calls += 1
        entry = stack[0].name
    else:
        entry = call.name
    key = (entry, call.name)
    counter = state.counters.get(key)
    if counter is None:
        counter = Counter(category=category)
        state.counters[key] = counter
    counter.calls += 1
    counter.inclusive_ns += inclusive_ns
    counter.exclusive_ns += inclusive_ns - call.child_ns
    counter.child_calls += call.child_calls


def _timed(original: Callable[..., Any], name: str, category: str) -> Callable[..., Any]:
    def timed_call(*args: Any, **kwargs: Any) -> Any:
        state = _thread_state()
        call = OpenCall(name=name)
        state.stack.append(call)
        call.start_ns = perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            end_ns = perf_counter_ns()
            state.stack.pop()
            _close_call(state, call, category, end_ns)

    timed_call.__name__ = getattr(original, "__name__", name)
    timed_call.__qualname__ = getattr(original, "__qualname__", name)
    timed_call.__doc__ = getattr(original, "__doc__", None)
    timed_call.__wrapped__ = original
    return timed_call


def _time_python_class(cls: type, category: str) -> List[str]:
    """Puts a timer around every plain method defined in cls itself; returns their names."""
    names: List[str] = []
    for attribute_name, value in list(vars(cls).items()):
        if attribute_name.startswith("__"):
            continue
        if not inspect.isfunction(value):
            continue
        name = f"{cls.__name__}.{attribute_name}"
        if name in NOT_TIMED:
            continue
        if inspect.isgeneratorfunction(value):
            # A timer would only cover creating the generator, not running it.
            continue
        setattr(cls, attribute_name, _timed(value, name, category))
        names.append(name)
    return names


def _time_extension_class(cls: type) -> List[str]:
    """Puts a timer around every method of a pybind11 class; returns their names."""
    names: List[str] = []
    for attribute_name, value in list(vars(cls).items()):
        if attribute_name.startswith("_"):
            continue
        name = f"{cls.__name__}.{attribute_name}"
        if isinstance(value, staticmethod):
            setattr(cls, attribute_name, staticmethod(_timed(value.__func__, name, IAXL_CPP)))
            names.append(name)
        elif type(value).__name__ == "instancemethod":
            # Read through the class: that gives the plain function, called as f(self, ...).
            setattr(cls, attribute_name, _timed(getattr(cls, attribute_name), name, IAXL_CPP))
            names.append(name)
    return names


def _per_call_ns(loop: Callable[[], None]) -> float:
    rounds: List[float] = []
    for _ in range(CALIBRATION_ROUNDS):
        start_ns = perf_counter_ns()
        loop()
        rounds.append((perf_counter_ns() - start_ns) / CALIBRATION_CALLS)
    return statistics.median(rounds)


def _calibrate() -> Dict[str, float]:
    """Measures what the timer adds to one call, on this process's CPU.

    Returns:
        raw_call_ns: one call of an empty function without a timer.
        timed_call_ns: one call of the same function with a timer.
        inside_ns: how much the timed call's own inclusive_ns over-reports (timer cost that
            falls between its two clock reads).
        outside_ns: the rest of the timer cost, which the caller's exclusive_ns absorbs.
    """
    def empty() -> None:
        return None

    timed_empty = _timed(empty, "calibration.empty", PYTHON)

    def no_call_loop() -> None:
        for _ in range(CALIBRATION_CALLS):
            pass

    def raw_loop() -> None:
        for _ in range(CALIBRATION_CALLS):
            empty()

    def timed_loop() -> None:
        for _ in range(CALIBRATION_CALLS):
            timed_empty()

    loop_ns = _per_call_ns(no_call_loop)
    raw_call_ns = _per_call_ns(raw_loop) - loop_ns
    state = _thread_state()
    state.counters.clear()
    timed_call_ns = _per_call_ns(timed_loop) - loop_ns
    counter = state.counters.pop(("calibration.empty", "calibration.empty"))
    measured_inclusive_ns = counter.inclusive_ns / counter.calls
    inside_ns = measured_inclusive_ns - raw_call_ns
    return {
        "raw_call_ns": raw_call_ns,
        "timed_call_ns": timed_call_ns,
        "measured_inclusive_ns": measured_inclusive_ns,
        "inside_ns": inside_ns,
        "outside_ns": timed_call_ns - raw_call_ns - inside_ns,
    }


def _write_snapshot(signum: int, frame: Any) -> None:
    start_ns = perf_counter_ns()
    directory = os.environ["PYOH_TIMING_DIR"]
    with open(os.path.join(directory, "timing_label")) as file:
        label = file.read().strip()
    threads: List[Dict[str, Any]] = []
    with _all_states_lock:
        states = list(_all_states)
    for state in states:
        counters: List[Dict[str, Any]] = []
        for (entry, name), counter in list(state.counters.items()):
            counters.append({
                "entry": entry,
                "name": name,
                "category": counter.category,
                "calls": counter.calls,
                "inclusive_ns": counter.inclusive_ns,
                "exclusive_ns": counter.exclusive_ns,
                "child_calls": counter.child_calls,
            })
        threads.append({"thread_name": state.thread_name, "is_main": state.is_main, "counters": counters})
    data = {
        "label": label,
        "pid": os.getpid(),
        "calibration": _calibration,
        # How long the earlier snapshots took; each one ran inside whatever the main thread was doing.
        "previous_snapshot_ns": list(_snapshot_ns),
        "threads": threads,
    }
    path = os.path.join(directory, f"timing_{label}.json")
    with open(path + ".tmp", "w") as file:
        json.dump(data, file)
    os.replace(path + ".tmp", path)
    _snapshot_ns.append(perf_counter_ns() - start_ns)


TIMED_FUNCTIONS: List[str] = []
for _cls in (
    connector_module.KVShrinkConnector,
    connector_module.ReqMeta,
    connector_module.RequestMetadata,
    kvstore_module.KVStoreLocal,
    flow_module.KVFlow,
):
    TIMED_FUNCTIONS.extend(_time_python_class(_cls, PYTHON))
TIMED_FUNCTIONS.extend(_time_python_class(scratch_pool_module.ScratchPool, SCRATCH_POOL))
# The connector calls it through the name it imported.
connector_module.generate_block_hashs = _timed(
    connector_module.generate_block_hashs, "generate_block_hashs", PYTHON
)
TIMED_FUNCTIONS.append("generate_block_hashs")
for _cls in (torch_ext.Context, torch_ext.Record, torch_ext.Mem, torch_ext.Storage):
    TIMED_FUNCTIONS.extend(_time_extension_class(_cls))
# flow.py calls these through the names it imported.
for _function_name in ("task_queue_set_bundles_priority", "task_queue_sort_priority_level"):
    setattr(flow_module, _function_name, _timed(getattr(flow_module, _function_name), _function_name, IAXL_CPP))
    TIMED_FUNCTIONS.append(_function_name)

if threading.current_thread() is threading.main_thread():
    _calibration.update(_calibrate())
    if "PYOH_TIMING_DIR" in os.environ:
        signal.signal(signal.SIGUSR1, _write_snapshot)

KVShrinkConnector = connector_module.KVShrinkConnector
