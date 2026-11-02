# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Times the release of per-block data on a TaskQueue thread, without a GPU or vLLM.

release_bench_ext.cpp hands blocks to IAXL's TaskQueue the way Context::unzip_from_mem and
Context::zip_to_mem do. For each way of passing blocks (variant) and each thing the Python main
thread does meanwhile (main-thread state), this script submits rounds of tasks, one task per layer
of one request, and reports:
  - the gap between a task's end and the next task's start on the queue thread, i.e. the release of
    what the task held plus taking the next task;
  - the main-thread time of allocating blocks, submitting a task, releasing blocks, and (for the
    zip_to_mem shape) waiting for a task.
Run it through run_release_bench.sh.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.cpp_extension import load

from iaxl.kvflow.scratch_pool import ScratchPool

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(BENCH_DIR))
EXT_SOURCE = os.path.join(BENCH_DIR, "release_bench_ext.cpp")
IAXL_INCLUDE_DIR = os.path.join(REPO_DIR, "iaxl", "csrc", "include")

# Qwen2.5-72B at TP=8, one layer of one block: K and V, 16 tokens, 1 KV head per rank, head size 128.
BLOCK_SHAPE = (2, 16, 1, 128)
BLOCK_DTYPE = torch.bfloat16
LABEL = "kv"
# Upper bound for the made-up GPU block ids; their values do not affect the release.
GPU_NUM_BLOCKS = 20000

PY_TENSORS = "py_tensors"
PY_TENSORS_KEPT_FUTURE = "py_tensors_kept_future"
SLOTS = "slots"
CPP_VIEWS = "cpp_views"
SLOTS_NO_LABELS = "slots_no_labels"

VARIANT_DESCRIPTIONS = {
    PY_TENSORS: "现状 unzip_from_mem:每个 block 一个 ScratchPool 的 Python tensor + label 字符串",
    PY_TENSORS_KEPT_FUTURE: "现状 zip_to_mem:同上,但 lambda 捕获 tensor 且 future 留到 wait",
    SLOTS: "方案 B:只传 slot 编号,C++ 存地址;label 字符串照旧",
    CPP_VIEWS: "方案 B':只传 slot 编号,C++ 用自建的 view(无 Python 对象);label 照旧",
    SLOTS_NO_LABELS: "下限:只传 slot 编号,且不传 label 字符串",
}

WAIT_IN_CPP = "wait_in_cpp"
TORCH_OPS_LOOP = "torch_ops_loop"
PURE_PYTHON_LOOP = "pure_python_loop"

STATE_DESCRIPTIONS = {
    WAIT_IN_CPP: "主线程停在 C++ 里等(已放 GIL)",
    TORCH_OPS_LOOP: "主线程循环调用小 torch 算子(每个算子前后放、拿一次 GIL)",
    PURE_PYTHON_LOOP: "主线程循环跑纯 Python 代码(不主动放 GIL)",
}


@dataclass
class RunPlan:
    variant: str
    main_thread_state: str
    rounds: int
    tasks_per_round: int


@dataclass
class RequestBlocks:
    """What one request passes for every layer: block hashes as strings and GPU block ids."""
    chunk_labels: List[str]
    chunk_indices: List[int]


@dataclass
class TaskRecord:
    variant: str
    main_thread_state: str
    round_index: int
    layer_index: int
    start_ns: int
    end_ns: int
    # None for the last task of a round, which has no next task.
    gap_to_next_ns: Optional[int]


@dataclass
class MainThreadCall:
    variant: str
    main_thread_state: str
    call: str
    duration_ns: int


@dataclass
class RoundResult:
    tasks: List[TaskRecord] = field(default_factory=list)
    main_thread_calls: List[MainThreadCall] = field(default_factory=list)


class SlotPool:
    """Free list of slot numbers, the way ScratchPool would keep it under scheme B (no views)."""

    def __init__(self, num_blocks: int):
        self.free_slots: List[int] = list(range(num_blocks))

    def allocate(self, count: int) -> List[int]:
        if count > len(self.free_slots):
            raise ValueError(f"SlotPool: need {count} slots, {len(self.free_slots)} free")
        cut = len(self.free_slots) - count
        slots = self.free_slots[cut:]
        del self.free_slots[cut:]
        return slots

    def release(self, slots: List[int]) -> None:
        self.free_slots.extend(slots)


@dataclass
class BenchContext:
    bench: object
    scratch_pool: ScratchPool
    slot_pool: SlotPool
    blocks_per_task: int
    rng: random.Random
    next_task_id: int = 0


def wait_in_cpp(bench) -> None:
    bench.wait_fence()


def run_torch_ops_until_done(bench) -> None:
    # Each torch op releases the GIL around its kernel, as the ops of vLLM's eager forward do.
    scratch = torch.zeros(16)
    while not bench.fence_done():
        scratch.add_(1.0)
    bench.wait_fence()


def run_python_until_done(bench) -> None:
    # Pure bytecode never releases the GIL by itself: another thread gets it only after asking and
    # waiting out the interpreter's switch interval.
    total = 0
    while not bench.fence_done():
        for step in range(1000):
            total += step
    bench.wait_fence()


MAIN_THREAD_STATES: Dict[str, Callable] = {
    WAIT_IN_CPP: wait_in_cpp,
    TORCH_OPS_LOOP: run_torch_ops_until_done,
    PURE_PYTHON_LOOP: run_python_until_done,
}


def build_plans(args: argparse.Namespace) -> List[RunPlan]:
    plans: List[RunPlan] = []
    # The zip_to_mem shape releases its tensors on the thread that waits, after the queue is done,
    # so the main-thread state during the round does not matter for it: run it once.
    for variant in (PY_TENSORS, PY_TENSORS_KEPT_FUTURE, SLOTS, CPP_VIEWS, SLOTS_NO_LABELS):
        plans.append(RunPlan(variant, WAIT_IN_CPP, args.rounds, args.tasks_per_round))
    for variant in (PY_TENSORS, SLOTS, CPP_VIEWS, SLOTS_NO_LABELS):
        plans.append(RunPlan(variant, TORCH_OPS_LOOP, args.rounds, args.tasks_per_round))
    for variant in (PY_TENSORS, SLOTS, CPP_VIEWS, SLOTS_NO_LABELS):
        plans.append(RunPlan(variant, PURE_PYTHON_LOOP, args.pure_python_rounds,
                             args.pure_python_tasks_per_round))
    return plans


def make_request_blocks(rng: random.Random, blocks_per_task: int) -> RequestBlocks:
    chunk_labels: List[str] = []
    chunk_indices: List[int] = []
    for _ in range(blocks_per_task):
        # kvshrink_connector passes block hashes as str() of a 63-bit integer.
        chunk_labels.append(str(rng.getrandbits(63)))
        chunk_indices.append(rng.randrange(GPU_NUM_BLOCKS))
    return RequestBlocks(chunk_labels, chunk_indices)


def run_round(context: BenchContext, plan: RunPlan, round_index: int) -> RoundResult:
    bench = context.bench
    request = make_request_blocks(context.rng, context.blocks_per_task)
    first_task_id = context.next_task_id
    context.next_task_id += plan.tasks_per_round
    result = RoundResult()
    held_tensors: List[List[torch.Tensor]] = []
    held_slots: List[List[int]] = []

    def record_call(call: str, duration_ns: int) -> None:
        result.main_thread_calls.append(
            MainThreadCall(plan.variant, plan.main_thread_state, call, duration_ns))

    bench.hold_queue()
    for layer_index in range(plan.tasks_per_round):
        task_id = first_task_id + layer_index
        # kvshrink_connector keys KVStore layers by str(layer index), so labels end in ":<index>".
        tensor_key = str(layer_index)
        if plan.variant == PY_TENSORS or plan.variant == PY_TENSORS_KEPT_FUTURE:
            allocate_start = time.perf_counter_ns()
            blocks = context.scratch_pool.allocate(context.blocks_per_task)
            record_call("allocate", time.perf_counter_ns() - allocate_start)
            if blocks is None:
                raise RuntimeError(f"ScratchPool exhausted: {context.scratch_pool.status()}")
            held_tensors.append(blocks)
            submit_start = time.perf_counter_ns()
            if plan.variant == PY_TENSORS:
                bench.submit_py_tensors(task_id, LABEL, tensor_key, request.chunk_labels,
                                        request.chunk_indices, blocks)
            else:
                bench.submit_py_tensors_kept_future(task_id, LABEL, tensor_key,
                                                    request.chunk_labels, blocks)
            record_call("submit", time.perf_counter_ns() - submit_start)
        else:
            allocate_start = time.perf_counter_ns()
            slots = context.slot_pool.allocate(context.blocks_per_task)
            record_call("allocate", time.perf_counter_ns() - allocate_start)
            held_slots.append(slots)
            submit_start = time.perf_counter_ns()
            if plan.variant == SLOTS:
                bench.submit_slots(task_id, LABEL, tensor_key, request.chunk_labels,
                                   request.chunk_indices, slots)
            elif plan.variant == CPP_VIEWS:
                bench.submit_cpp_views(task_id, LABEL, tensor_key, request.chunk_labels,
                                       request.chunk_indices, slots)
            elif plan.variant == SLOTS_NO_LABELS:
                bench.submit_slots(task_id, LABEL, tensor_key, [], request.chunk_indices, slots)
            else:
                raise ValueError(f"unknown variant {plan.variant}")
            record_call("submit", time.perf_counter_ns() - submit_start)
    bench.submit_fence()
    bench.resume_queue()
    MAIN_THREAD_STATES[plan.main_thread_state](bench)

    if plan.variant == PY_TENSORS_KEPT_FUTURE:
        for layer_index in range(plan.tasks_per_round):
            record_call("wait_kept_future", bench.wait_kept_future(first_task_id + layer_index))

    for blocks in held_tensors:
        release_start = time.perf_counter_ns()
        context.scratch_pool.release(blocks)
        record_call("release", time.perf_counter_ns() - release_start)
    for slots in held_slots:
        release_start = time.perf_counter_ns()
        context.slot_pool.release(slots)
        record_call("release", time.perf_counter_ns() - release_start)

    start_times = bench.start_times(first_task_id, plan.tasks_per_round)
    end_times = bench.end_times(first_task_id, plan.tasks_per_round)
    for layer_index in range(plan.tasks_per_round):
        gap_to_next_ns = None
        if layer_index + 1 < plan.tasks_per_round:
            gap_to_next_ns = start_times[layer_index + 1] - end_times[layer_index]
        result.tasks.append(TaskRecord(plan.variant, plan.main_thread_state, round_index,
                                       layer_index, start_times[layer_index],
                                       end_times[layer_index], gap_to_next_ns))
    return result


def microseconds(values_ns: List[int]) -> np.ndarray:
    return np.asarray(values_ns, dtype=np.float64) / 1000.0


def format_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summarize(tasks: List[TaskRecord], calls: List[MainThreadCall], plans: List[RunPlan],
              blocks_per_task: int) -> str:
    gaps: Dict[str, Dict[str, List[int]]] = {}
    runs: Dict[str, Dict[str, List[int]]] = {}
    for task in tasks:
        state_gaps = gaps.setdefault(task.main_thread_state, {})
        state_runs = runs.setdefault(task.main_thread_state, {})
        state_runs.setdefault(task.variant, []).append(task.end_ns - task.start_ns)
        if task.gap_to_next_ns is not None:
            state_gaps.setdefault(task.variant, []).append(task.gap_to_next_ns)

    sections: List[str] = []
    sections.append("## 1. 队列线程上每个任务结束到下一个任务开始的间隔(release + 取下一个任务)")
    rows: List[List[str]] = []
    for plan in plans:
        values = microseconds(gaps[plan.main_thread_state][plan.variant])
        run_values = microseconds(runs[plan.main_thread_state][plan.variant])
        rows.append([plan.main_thread_state, plan.variant, str(len(values)),
                     f"{np.median(values):.1f}", f"{np.mean(values):.1f}",
                     f"{np.percentile(values, 95):.1f}", f"{np.max(values):.1f}",
                     f"{np.median(run_values):.1f}"])
    sections.append(format_table(
        ["主线程状态", "传法", "间隔个数", "间隔中位(µs)", "间隔平均(µs)", "间隔p95(µs)",
         "间隔最大(µs)", "任务执行中位(µs)"], rows))
    sections.append(
        "列说明:主线程状态 / 传法 见文末;间隔个数 = 统计到的间隔数(每轮最后一个任务后面没有任务,不计);"
        "间隔 = 下一个任务开始时刻 − 本任务结束时刻,即释放本任务持有的数据 + 队列取下一个任务;"
        f"任务执行 = 任务函数本身(把 {blocks_per_task} 个 block 各写一遍)的耗时。")

    sections.append("## 2. 间隔拆分(按中位数相减)")
    rows = []
    for state in (WAIT_IN_CPP, TORCH_OPS_LOOP, PURE_PYTHON_LOOP):
        state_gaps = gaps[state]
        current = float(np.median(microseconds(state_gaps[PY_TENSORS])))
        slots = float(np.median(microseconds(state_gaps[SLOTS])))
        cpp_views = float(np.median(microseconds(state_gaps[CPP_VIEWS])))
        floor = float(np.median(microseconds(state_gaps[SLOTS_NO_LABELS])))
        current_mean = float(np.mean(microseconds(state_gaps[PY_TENSORS])))
        slots_mean = float(np.mean(microseconds(state_gaps[SLOTS])))
        rows.append([state, f"{current:.1f}", f"{current - slots:.1f}",
                     f"{(current - slots) * 1000.0 / blocks_per_task:.0f}",
                     f"{slots - floor:.1f}", f"{floor:.1f}",
                     f"{(current - slots) / current * 100.0:.1f}%",
                     f"{(current - cpp_views) / current * 100.0:.1f}%",
                     f"{(current_mean - slots_mean) / current_mean * 100.0:.1f}%"])
    sections.append(format_table(
        ["主线程状态", "现状间隔(µs)", "tensor 部分(µs)", "tensor 每 block(ns)",
         "label 部分(µs)", "下限(µs)", "B 去掉的比例", "B' 去掉的比例",
         "B 去掉的比例(按均值)"], rows))
    sections.append(
        "列说明:现状间隔 = py_tensors 的间隔中位;tensor 部分 = py_tensors − slots(两者只差"
        "传 tensor 还是传编号);tensor 每 block = tensor 部分 ÷ 每任务 block 数;label 部分 = "
        "slots − slots_no_labels(只差有没有 label 字符串);下限 = slots_no_labels,即没有 tensor "
        "也没有 label 时剩下的间隔;B / B' 去掉的比例 = (现状 − slots 或 cpp_views) ÷ 现状;"
        "最后一列用两者的间隔均值代替中位数计算。")

    sections.append("## 3. 主线程上每次调用的耗时(主线程状态 wait_in_cpp 的各轮)")
    call_values: Dict[str, Dict[str, List[int]]] = {}
    for call in calls:
        if call.main_thread_state != WAIT_IN_CPP:
            continue
        call_values.setdefault(call.variant, {}).setdefault(call.call, []).append(call.duration_ns)
    rows = []
    for variant in (PY_TENSORS, PY_TENSORS_KEPT_FUTURE, SLOTS, CPP_VIEWS, SLOTS_NO_LABELS):
        row = [variant]
        for call_name in ("allocate", "submit", "release", "wait_kept_future"):
            values = call_values[variant].get(call_name)
            if values is None:
                row.append("-")
            else:
                values_us = microseconds(values)
                row.append(f"{np.median(values_us):.1f} / {np.mean(values_us):.1f}")
        rows.append(row)
    sections.append(format_table(
        ["传法", "allocate(µs)", "submit(µs)", "release(µs)", "wait(µs)"], rows))
    sections.append(
        f"列说明:每格是一个任务({blocks_per_task} 个 block)一次调用耗时的 中位 / 均值;allocate / release ="
        " 从 pool 取、还 block(现状用 ScratchPool,返回 Python tensor;方案 B 用整数编号);"
        "submit = 调一次 C++ 提交函数,含 pybind 把列表逐个转成 C++ 值;wait = zip_to_mem 形态下"
        "主线程在 C++ 里 future.get() 的耗时(此时队列线程已做完,lambda 在这里销毁)。")

    sections.append("## 传法 / 主线程状态")
    for variant, description in VARIANT_DESCRIPTIONS.items():
        sections.append(f"- {variant}: {description}")
    for state, description in STATE_DESCRIPTIONS.items():
        sections.append(f"- {state}: {description}")
    return "\n\n".join(sections) + "\n"


def file_md5(path: str) -> str:
    with open(path, "rb") as source:
        return hashlib.md5(source.read()).hexdigest()


def write_json(path: str, value: object) -> None:
    with open(path, "w") as output:
        json.dump(value, output, indent=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=10,
                        help="measured rounds (requests) per variant and main-thread state")
    parser.add_argument("--tasks-per-round", type=int, default=80,
                        help="tasks per round, one per layer (Qwen2.5-72B has 80)")
    parser.add_argument("--blocks-per-task", type=int, default=500,
                        help="blocks per task, about input tokens / 16 (block size): "
                             "500 = 8k, 1000 = 16k, 2000 = 32k")
    # The current code under a pure-Python main thread waits for the GIL block by block, up to
    # about 1 s per task measured here, so this state runs far fewer tasks.
    parser.add_argument("--pure-python-rounds", type=int, default=3)
    parser.add_argument("--pure-python-tasks-per-round", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("rounds", "tasks_per_round", "blocks_per_task", "pure_python_rounds"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    # A round needs two tasks for one gap.
    if args.tasks_per_round < 2 or args.pure_python_tasks_per_round < 2:
        raise ValueError("tasks per round must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_dir = os.path.join(BENCH_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    extension = load(name="omp_release_bench_ext", sources=[EXT_SOURCE],
                     extra_include_paths=[IAXL_INCLUDE_DIR], extra_cflags=["-O2"],
                     build_directory=build_dir, verbose=False)

    scratch_pool = ScratchPool(BLOCK_SHAPE, BLOCK_DTYPE, pin_memory=False)
    plans = build_plans(args)
    blocks_in_flight = 0
    max_tasks = 0
    for plan in plans:
        blocks_in_flight = max(blocks_in_flight, plan.tasks_per_round * args.blocks_per_task)
        # +1: the warmup round of every plan.
        max_tasks += (plan.rounds + 1) * plan.tasks_per_round
    if blocks_in_flight > scratch_pool.num_blocks:
        raise ValueError(f"a round needs {blocks_in_flight} blocks, the pool has "
                         f"{scratch_pool.num_blocks}; raise IAXL_SCRATCH_POOL_SIZE_GB")

    bench = extension.ReleaseBench(scratch_pool.pool, max_tasks)
    context = BenchContext(bench, scratch_pool, SlotPool(scratch_pool.num_blocks),
                           args.blocks_per_task, random.Random(args.seed))
    meta = {
        "args": vars(args),
        "md5": {
            "release_bench.py": file_md5(os.path.abspath(__file__)),
            "release_bench_ext.cpp": file_md5(EXT_SOURCE),
            "task_queue.h": file_md5(os.path.join(IAXL_INCLUDE_DIR, "task_queue.h")),
            "kv_pool.h": file_md5(os.path.join(IAXL_INCLUDE_DIR, "kv_pool.h")),
            "scratch_pool.py": file_md5(os.path.join(REPO_DIR, "iaxl", "kvflow",
                                                     "scratch_pool.py")),
        },
        "block_shape": list(BLOCK_SHAPE),
        "block_dtype": str(BLOCK_DTYPE),
        "block_bytes": bench.block_bytes,
        "scratch_pool_blocks": scratch_pool.num_blocks,
        "torch": torch.__version__,
        "python": sys.version,
        "switch_interval_s": sys.getswitchinterval(),
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "tcmalloc_release_rate": os.environ.get("TCMALLOC_RELEASE_RATE", ""),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host": platform.node(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(args.out_dir, "meta.json"), meta)
    print(json.dumps(meta, indent=1), flush=True)

    plans_by_state: Dict[str, List[RunPlan]] = {}
    for plan in plans:
        plans_by_state.setdefault(plan.main_thread_state, []).append(plan)

    tasks: List[TaskRecord] = []
    calls: List[MainThreadCall] = []
    for state, state_plans in plans_by_state.items():
        state_start = time.perf_counter()
        for plan in state_plans:
            run_round(context, plan, -1)  # warmup, not recorded
        # Rounds of the variants take turns: the main-thread cost of allocating the label strings
        # grows over a run, so running one variant's rounds after another's would bias the later.
        max_rounds = 0
        for plan in state_plans:
            max_rounds = max(max_rounds, plan.rounds)
        for round_index in range(max_rounds):
            for plan in state_plans:
                if round_index >= plan.rounds:
                    continue
                result = run_round(context, plan, round_index)
                tasks.extend(result.tasks)
                calls.extend(result.main_thread_calls)
        print(f"{state}: {len(state_plans)} variants x {max_rounds} rounds in "
              f"{time.perf_counter() - state_start:.1f} s", flush=True)
    bench.shutdown()

    task_dicts: List[dict] = []
    for task in tasks:
        task_dicts.append(asdict(task))
    call_dicts: List[dict] = []
    for call in calls:
        call_dicts.append(asdict(call))
    write_json(os.path.join(args.out_dir, "tasks.json"), task_dicts)
    write_json(os.path.join(args.out_dir, "main_thread_calls.json"), call_dicts)
    summary = summarize(tasks, calls, plans, args.blocks_per_task)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as output:
        output.write(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
