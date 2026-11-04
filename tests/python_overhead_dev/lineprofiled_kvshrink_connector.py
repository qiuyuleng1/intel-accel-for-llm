# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVShrinkConnector that runs line_profiler over kvshrink's Python code, for run_lineprofiler_bench.sh.

vLLM loads it with --kv-transfer-config '{"kv_connector": "LineProfiledKVShrinkConnector",
"kv_connector_module_path": "lineprofiled_kvshrink_connector", ...}' and this directory on PYTHONPATH.
kvshrink itself is not changed: every method below calls the real one, with line_profiler switched
on for the duration of the call.

The run script controls the measurement with a one-line file, PROFILE_CONTROL_DIR/control:
  "<seq> start"  profile every following call vLLM makes into the connector
  "<seq> stop"   stop, and write line_profile.lprof and line_profile.txt
The EngineCore main thread reads the file at the start of a connector call, at most every 0.2 s,
so a command takes effect with the next request. Every handled command is appended to control.log.
"""

import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from line_profiler import LineProfiler

import iaxl.kvflow.flow
import iaxl.kvflow.scratch_pool
import iaxl.kvstore.kvstore
import iaxl.utils.hash
import kvshrink.kvshrink_connector
from kvshrink.kvshrink_connector import KVShrinkConnector

CONTROL_DIR = os.environ["PROFILE_CONTROL_DIR"]
CONTROL_FILE = os.path.join(CONTROL_DIR, "control")
CONTROL_LOG = os.path.join(CONTROL_DIR, "control.log")
CONTROL_CHECK_INTERVAL_S = 0.2
# kvshrink's Python code that runs in the EngineCore main thread.
PROFILED_MODULES = (
    kvshrink.kvshrink_connector,
    iaxl.kvstore.kvstore,
    iaxl.kvflow.flow,
    iaxl.utils.hash,
    iaxl.kvflow.scratch_pool,
)
REPORT_UNIT_S = 1e-6


@dataclass
class ControlState:
    profiler: Optional[LineProfiler] = None
    next_check: float = 0.0
    last_seq: int = 0


# One state for the whole process: with TP=1 the scheduler-side and the worker-side connector are
# two objects in the same EngineCore main thread, and their calls belong to the same measurement.
STATE = ControlState()


def log_control(text: str) -> None:
    with open(CONTROL_LOG, "a") as file:
        file.write(f"{time.strftime('%H:%M:%S')} {text}\n")


def plain_function(attribute: Any) -> Optional[Callable[..., Any]]:
    if isinstance(attribute, (staticmethod, classmethod)):
        attribute = attribute.__func__
    elif isinstance(attribute, property):
        attribute = attribute.fget
    if inspect.isfunction(attribute):
        return attribute
    return None


def add_module_functions(profiler: LineProfiler, module: Any) -> None:
    # Only what the module's own file defines: LineProfiler.add_module would also add every class
    # and function it imports, such as vLLM's, and the __init__ that @dataclass generates has no
    # source file to show.
    for item in vars(module).values():
        if inspect.isclass(item) and item.__module__ == module.__name__:
            for attribute in vars(item).values():
                function = plain_function(attribute)
                if function is not None and function.__code__.co_filename == module.__file__:
                    profiler.add_function(function)
        elif inspect.isfunction(item) and item.__code__.co_filename == module.__file__:
            profiler.add_function(item)


def start_profiling() -> None:
    profiler = LineProfiler()
    for module in PROFILED_MODULES:
        add_module_functions(profiler, module)
    STATE.profiler = profiler


def stop_profiling() -> None:
    profiler = STATE.profiler
    if profiler is None:
        return
    STATE.profiler = None
    profiler.dump_stats(os.path.join(CONTROL_DIR, "line_profile.lprof"))
    with open(os.path.join(CONTROL_DIR, "line_profile.txt"), "w") as file:
        profiler.print_stats(stream=file, output_unit=REPORT_UNIT_S, stripzeros=True)


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
        start_profiling()
        log_control(f"{seq} started")
    elif words[1] == "stop":
        stop_profiling()
        log_control(f"{seq} stopped")
    else:
        raise ValueError(f"unknown control command {' '.join(words)}")


def profiled_call(method: Callable[..., Any], args: tuple, kwargs: Dict[str, Any]) -> Any:
    now = time.monotonic()
    if now >= STATE.next_check:
        STATE.next_check = now + CONTROL_CHECK_INTERVAL_S
        poll_control()
    profiler = STATE.profiler
    if profiler is None:
        return method(*args, **kwargs)
    # Counted, so that request_finished called from request_finished_all_groups keeps it on.
    profiler.enable_by_count()
    try:
        return method(*args, **kwargs)
    finally:
        profiler.disable_by_count()


class LineProfiledKVShrinkConnector(KVShrinkConnector):
    """KVShrinkConnector whose vLLM-facing methods run under line_profiler once started."""

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return profiled_call(super().get_num_new_matched_tokens, (request, num_computed_tokens), {})

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return profiled_call(
            super().update_state_after_alloc, (request, blocks, num_external_tokens), {},
        )

    def request_finished(self, request, block_ids):
        return profiled_call(super().request_finished, (request, block_ids), {})

    def request_finished_all_groups(self, request, block_ids):
        return profiled_call(super().request_finished_all_groups, (request, block_ids), {})

    def build_connector_meta(self, scheduler_output):
        return profiled_call(super().build_connector_meta, (scheduler_output,), {})

    def start_load_kv(self, forward_context, **kwargs):
        return profiled_call(super().start_load_kv, (forward_context,), kwargs)

    def wait_for_layer_load(self, layer_name):
        return profiled_call(super().wait_for_layer_load, (layer_name,), {})

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return profiled_call(super().save_kv_layer, (layer_name, kv_layer, attn_metadata), kwargs)

    def wait_for_save(self):
        return profiled_call(super().wait_for_save, (), {})

    def get_finished(self, finished_req_ids):
        return profiled_call(super().get_finished, (finished_req_ids,), {})
