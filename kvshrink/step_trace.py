# SPDX-License-Identifier: Apache-2.0
"""Per-step trace of vLLM GPU workers, to find where a decode step's time goes.

KVSHRINK_STEP_TRACE=1 wraps the model runner's execute_model in every worker
(the kvshrink and the plain vLLM baselines alike, through the vllm_patch
plugin) and logs one "step-trace" line per step and TP rank. A step's line is
logged when the next step starts, because only then are its period and GPU
time known. vLLM 0.29 runs dense models on its V2 model runner
(vllm/v1/worker/gpu/model_runner.py); the V1 runner is wrapped as well.

Fields of a line:
  rank, step, t            TP rank, step counter of this worker, epoch seconds
                           when the step started
  reqs, decode_reqs,       requests in the batch, how many of them decode one
  prefill_tokens, tokens   token, tokens of the others, all scheduled tokens
  period_ms                this step's start to the next step's start
  exec_ms                  wall time inside execute_model (sampling excluded)
  gpu_ms                   compute-stream time between execute_model's entry
                           and exit (CUDA events), -1 if not finished in time
  main_cpu_ms              CPU time of the worker's main thread in the period
  main_runq_ms             time that thread was runnable but waited for a CPU
                           core (/proc/thread-self/schedstat, 2nd field)
  other_threads_cpu_ms     CPU time of all other threads of the worker process
  start_load_kv_ms ...     KVShrink connector time in this step, summed over
  get_finished_ms          its calls (wait_for_layer_load and save_kv_layer run
                           once per layer); wait_for_layer_load_blocked_ms is
                           the part spent blocked in KVStore.get_wait
  loads_started            requests whose load the connector submitted
  loads_unfinished         loads submitted earlier and not finished when the
                           step started
  promoted                 early-promoted loads this step's forward waits on
  block_refs, block_unique KV block references of the batch's requests,
  block_runs               distinct physical blocks among them, and runs of
                           consecutive block ids; rank 0 every
                           BLOCK_TABLE_EVERY_STEPS steps, -1 otherwise

KVSHRINK_THREAD_MAP=1 makes the KVShrink worker log its Python threads (name
and Linux thread id) whenever that set changes, so an outside /proc sampler
can tell Python threads from native ones.
"""

import functools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from vllm.logger import init_logger

# Under the "vllm" namespace so vLLM's log handler prints it.
logger = init_logger(f"vllm.{__name__}")

STEP_TRACE_ENABLED = os.environ.get("KVSHRINK_STEP_TRACE", "0") == "1"
THREAD_MAP_ENABLED = os.environ.get("KVSHRINK_THREAD_MAP", "0") == "1"
# The distinct-block count over 48 requests x 2048 blocks costs about a
# millisecond on the worker's main thread, so it is taken only this often, and
# only on rank 0: every rank holds the same block table.
BLOCK_TABLE_EVERY_STEPS = 20
# get_finished runs once per step; checking the thread set every call is waste.
THREAD_MAP_EVERY_CALLS = 500
MODEL_RUNNER_MODULES = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
)


@dataclass
class KVShrinkStepTimes:
    """KVShrink connector time in one step, summed over its calls."""
    start_load_kv_ms: float = 0.0
    wait_for_layer_load_ms: float = 0.0
    wait_for_layer_load_blocked_ms: float = 0.0
    save_kv_layer_ms: float = 0.0
    wait_for_save_ms: float = 0.0
    get_finished_ms: float = 0.0
    loads_started: int = 0
    loads_unfinished: int = 0
    promoted: int = 0


@dataclass
class BlockTableStats:
    """KV block references of a step's requests (see module docstring)."""
    block_refs: int = -1
    block_unique: int = -1
    block_runs: int = -1


@dataclass
class StepRecord:
    """What is known about a step when execute_model returns."""
    step: int
    start_epoch: float
    start_perf: float
    start_main_cpu: float
    start_process_cpu: float
    start_runq_ns: int
    reqs: int
    decode_reqs: int
    prefill_tokens: int
    tokens: int
    gpu_start: Any
    kvshrink: KVShrinkStepTimes
    exec_ms: float = 0.0
    gpu_end: Any = None
    block_table: BlockTableStats = field(default_factory=BlockTableStats)


kvshrink_step_times = KVShrinkStepTimes()
# Rank 0 only: request id -> physical block ids of KV cache group 0, rebuilt
# from the scheduler output because the V2 runner keeps its block table on
# the GPU, and reading it back would synchronize the step.
_block_ids_by_request: dict[str, list[int]] = {}
_runq_fd: Optional[int] = None
_rank: Optional[int] = None
_step = 0
_previous: Optional[StepRecord] = None
_thread_map_calls = 0
_thread_map_logged = ""


def add_connector_time(field_name: str):
    """Decorator adding the method's wall time (ms) to kvshrink_step_times.

    Returns the method unchanged when the trace is off, so it costs nothing.
    """

    def decorate(method):
        if not STEP_TRACE_ENABLED:
            return method

        @functools.wraps(method)
        def timed_method(*args, **kwargs):
            start = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1e3
                total_ms = getattr(kvshrink_step_times, field_name) + elapsed_ms
                setattr(kvshrink_step_times, field_name, total_ms)

        return timed_method

    return decorate


def add_blocked_load_wait(start: float) -> None:
    """Adds the time since ``start`` (perf_counter) to the blocked-wait total."""
    if not STEP_TRACE_ENABLED:
        return
    kvshrink_step_times.wait_for_layer_load_blocked_ms += (
        time.perf_counter() - start
    ) * 1e3


def record_loads(started: int, unfinished: int) -> None:
    if not STEP_TRACE_ENABLED:
        return
    kvshrink_step_times.loads_started += started
    kvshrink_step_times.loads_unfinished = unfinished


def record_promoted(promoted: int) -> None:
    if not STEP_TRACE_ENABLED:
        return
    kvshrink_step_times.promoted += promoted


def maybe_log_python_threads() -> None:
    """Logs this process's Python threads when the set changed (THREAD_MAP)."""
    global _thread_map_calls, _thread_map_logged
    if not THREAD_MAP_ENABLED:
        return
    _thread_map_calls += 1
    if _thread_map_calls % THREAD_MAP_EVERY_CALLS != 1:
        return
    parts = []
    for thread in threading.enumerate():
        parts.append(f"{thread.name}:{thread.native_id}")
    thread_map = ",".join(sorted(parts))
    if thread_map == _thread_map_logged:
        return
    _thread_map_logged = thread_map
    logger.info("thread-map pid=%d threads=%s", os.getpid(), thread_map)


def install_execute_model_trace() -> None:
    """Wraps execute_model of vLLM's V2 and V1 GPU model runners."""
    import importlib

    for module_name in MODEL_RUNNER_MODULES:
        runner_class = importlib.import_module(module_name).GPUModelRunner
        if getattr(runner_class, "_kvshrink_step_trace", False):
            continue
        runner_class.execute_model = _traced_execute_model(runner_class.execute_model)
        runner_class._kvshrink_step_trace = True
    logger.info("KVShrink step trace installed on the GPU model runners")


def _traced_execute_model(original_execute_model):

    @functools.wraps(original_execute_model)
    def execute_model(self, scheduler_output, *args, **kwargs):
        # Warm-up and profiling runs have no real batch.
        if kwargs.get("dummy_run", False):
            return original_execute_model(self, scheduler_output, *args, **kwargs)
        record = _start_step(scheduler_output)
        try:
            return original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            _finish_step(record, scheduler_output)

    return execute_model


def _read_runq_ns() -> int:
    """Run-queue wait (ns) of the calling thread: 2nd field of its schedstat."""
    global _runq_fd
    if _runq_fd is None:
        # Opened by the worker's main thread, so it reports that thread.
        _runq_fd = os.open("/proc/thread-self/schedstat", os.O_RDONLY)
    fields = os.pread(_runq_fd, 128, 0).split()
    return int(fields[1])


def _start_step(scheduler_output) -> StepRecord:
    global _step, _previous, kvshrink_step_times
    start_perf = time.perf_counter()
    start_main_cpu = time.thread_time()
    start_process_cpu = time.process_time()
    start_runq_ns = _read_runq_ns()
    if _previous is not None:
        _log_step(_previous, start_perf, start_main_cpu, start_process_cpu, start_runq_ns)
        _previous = None
    _step += 1
    kvshrink_step_times = KVShrinkStepTimes()

    decode_reqs = 0
    prefill_tokens = 0
    for num_tokens in scheduler_output.num_scheduled_tokens.values():
        if num_tokens == 1:
            decode_reqs += 1
        else:
            prefill_tokens += num_tokens

    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    return StepRecord(
        step=_step,
        start_epoch=time.time(),
        start_perf=start_perf,
        start_main_cpu=start_main_cpu,
        start_process_cpu=start_process_cpu,
        start_runq_ns=start_runq_ns,
        reqs=len(scheduler_output.num_scheduled_tokens),
        decode_reqs=decode_reqs,
        prefill_tokens=prefill_tokens,
        tokens=scheduler_output.total_num_scheduled_tokens,
        gpu_start=gpu_start,
        kvshrink=kvshrink_step_times,
    )


def _finish_step(record: StepRecord, scheduler_output) -> None:
    global _previous, _rank
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_end.record()
    record.gpu_end = gpu_end
    record.exec_ms = (time.perf_counter() - record.start_perf) * 1e3
    if _rank is None:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        _rank = get_tensor_model_parallel_rank()
    if _rank == 0:
        _update_block_ids(scheduler_output)
        if record.step % BLOCK_TABLE_EVERY_STEPS == 0:
            record.block_table = _block_table_stats(scheduler_output)
    _previous = record


def _update_block_ids(scheduler_output) -> None:
    for req_id in scheduler_output.finished_req_ids:
        _block_ids_by_request.pop(req_id, None)
    for new_request in scheduler_output.scheduled_new_reqs:
        _block_ids_by_request[new_request.req_id] = list(new_request.block_ids[0])
    cached = scheduler_output.scheduled_cached_reqs
    for index in range(len(cached.req_ids)):
        req_id = cached.req_ids[index]
        new_block_ids = cached.new_block_ids[index]
        if new_block_ids is None:
            continue
        # A resumed request gets its whole block list again.
        if req_id in cached.resumed_req_ids or req_id not in _block_ids_by_request:
            _block_ids_by_request[req_id] = list(new_block_ids[0])
        else:
            _block_ids_by_request[req_id].extend(new_block_ids[0])


def _block_table_stats(scheduler_output) -> BlockTableStats:
    rows = []
    runs = 0
    for req_id in scheduler_output.num_scheduled_tokens:
        block_ids = _block_ids_by_request.get(req_id)
        if not block_ids:
            continue
        row = np.asarray(block_ids, dtype=np.int64)
        rows.append(row)
        # A new run starts wherever the next block id is not this one + 1.
        runs += 1 + int(np.count_nonzero(np.diff(row) != 1))
    if not rows:
        return BlockTableStats(block_refs=0, block_unique=0, block_runs=0)
    all_blocks = np.concatenate(rows)
    return BlockTableStats(
        block_refs=int(all_blocks.size),
        block_unique=int(np.unique(all_blocks).size),
        block_runs=runs,
    )


def _log_step(record: StepRecord, next_perf: float, next_main_cpu: float,
              next_process_cpu: float, next_runq_ns: int) -> None:
    gpu_ms = -1.0
    if record.gpu_end is not None and record.gpu_end.query():
        gpu_ms = record.gpu_start.elapsed_time(record.gpu_end)
    main_cpu_ms = (next_main_cpu - record.start_main_cpu) * 1e3
    process_cpu_ms = (next_process_cpu - record.start_process_cpu) * 1e3
    times = record.kvshrink
    logger.info(
        "step-trace rank=%d step=%d t=%.3f reqs=%d decode_reqs=%d "
        "prefill_tokens=%d tokens=%d period_ms=%.3f exec_ms=%.3f gpu_ms=%.3f "
        "main_cpu_ms=%.3f main_runq_ms=%.3f other_threads_cpu_ms=%.3f "
        "start_load_kv_ms=%.3f wait_for_layer_load_ms=%.3f "
        "wait_for_layer_load_blocked_ms=%.3f save_kv_layer_ms=%.3f "
        "wait_for_save_ms=%.3f get_finished_ms=%.3f loads_started=%d "
        "loads_unfinished=%d promoted=%d block_refs=%d block_unique=%d "
        "block_runs=%d",
        _rank, record.step, record.start_epoch, record.reqs, record.decode_reqs,
        record.prefill_tokens, record.tokens, (next_perf - record.start_perf) * 1e3,
        record.exec_ms, gpu_ms, main_cpu_ms,
        (next_runq_ns - record.start_runq_ns) / 1e6,
        process_cpu_ms - main_cpu_ms, times.start_load_kv_ms,
        times.wait_for_layer_load_ms, times.wait_for_layer_load_blocked_ms,
        times.save_kv_layer_ms, times.wait_for_save_ms, times.get_finished_ms,
        times.loads_started, times.loads_unfinished, times.promoted,
        record.block_table.block_refs, record.block_table.block_unique,
        record.block_table.block_runs,
    )
