# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Writes summary.md for one result directory of run_cprofile_bench.sh.

Two benches of the same workload are measured inside the connector (profiled_kvshrink_connector.py):
  timers    calls and wall time of every connector method vLLM calls ("entry method")
  cprofile  the same, plus one cProfile per entry method
Everything cProfile records inside an entry method is put in one category:
  IAXL C++      the IAXL_<class>.<method> wrappers around IAXL's extension methods (with the C++
                call inside them) and IAXL's module functions; includes pybind11 converting the
                arguments
  ScratchPool   functions in iaxl/kvflow/scratch_pool.py, with everything they call
  Python 开销    anything else: kvshrink's Python functions and the C functions they call directly
                (xxhash, numpy, torch, list and dict operations, ...)
cProfile adds a fixed cost to every call it records. The connector measured that cost in the same
process (profile_cprofile.json), and the Python time of an entry method is reduced by it once per
recorded call. Times are reported in ms per request of the bench.

    python3 analyze_cprofile.py <result dir>
"""

import argparse
import json
import os
import pstats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from analyze_pyspy import (
    BENCH_METRICS,
    MatchedTokens,
    matched_tokens,
    read_bench_ranges,
    read_bench_result,
    read_cpu_ticks,
    read_log_range,
)

SCRATCH_POOL_FILE = os.path.join("iaxl", "kvflow", "scratch_pool.py")
WRAPPER_FILE = "profiled_kvshrink_connector.py"
# Same prefix as in profiled_kvshrink_connector.py.
IAXL_WRAPPER_PREFIX = "IAXL_"
# cProfile names a pybind11 module function "<built-in method iaxl.torch_ext.<name>>".
IAXL_MODULE_LABEL = "iaxl.torch_ext."
PROFILER_LABEL = "_lsprof.Profiler"
TOP_FUNCTIONS = 25
MEASURED_BENCHES = ("timers", "cprofile")

FunctionKey = Tuple[str, int, str]


@dataclass
class FunctionTime:
    """Python-overhead time of one function inside one entry method."""
    entry: str
    key: FunctionKey
    calls: int
    seconds: float
    top_caller: str


@dataclass
class IaxlMethodTime:
    entry: str
    label: str
    calls: int = 0
    seconds: float = 0.0


@dataclass
class EntryProfile:
    """What cProfile recorded inside one entry method during the cprofile bench."""
    entry: str
    total_s: float = 0.0
    iaxl_s: float = 0.0
    scratch_pool_s: float = 0.0
    # Python overhead before the cost of cProfile itself is taken out
    python_s: float = 0.0
    python_calls: int = 0
    c_calls: int = 0
    functions: List[FunctionTime] = field(default_factory=list)
    iaxl_methods: Dict[str, IaxlMethodTime] = field(default_factory=dict)


@dataclass
class EntryResult:
    entry: str
    calls_per_request: float
    wall_ms: float  # timers bench, ms per request
    iaxl_ms: float
    scratch_pool_ms: float
    python_ms: float  # after taking out cProfile's own cost
    python_raw_ms: float


def is_profiler(key: FunctionKey) -> bool:
    return PROFILER_LABEL in key[2]


def is_c_function(key: FunctionKey) -> bool:
    return key[0] == "~"


def is_scratch_pool(key: FunctionKey) -> bool:
    return key[0].endswith(SCRATCH_POOL_FILE)


def is_iaxl_wrapper(key: FunctionKey) -> bool:
    return key[0].endswith(WRAPPER_FILE) and key[2].startswith(IAXL_WRAPPER_PREFIX)


def is_iaxl_c_function(key: FunctionKey) -> bool:
    return is_c_function(key) and IAXL_MODULE_LABEL in key[2]


def is_measurement_code(key: FunctionKey) -> bool:
    return key[0].endswith(WRAPPER_FILE) and not key[2].startswith(IAXL_WRAPPER_PREFIX)


def function_name(key: FunctionKey) -> str:
    return key[2]


def add_iaxl_time(profile: EntryProfile, label: str, calls: int, seconds: float) -> None:
    method = profile.iaxl_methods.get(label)
    if method is None:
        method = IaxlMethodTime(profile.entry, label)
        profile.iaxl_methods[label] = method
    method.calls += calls
    method.seconds += seconds


def load_entry_profile(entry: str, path: str) -> EntryProfile:
    profile = EntryProfile(entry)
    stats = pstats.Stats(path).stats
    for key, values in stats.items():
        if is_profiler(key):
            continue
        calls, own_s, callers = values[1], values[2], values[4]
        profile.total_s += own_s
        if is_iaxl_wrapper(key):
            # The wrapper's cumulative time holds the C++ call, whether cProfile recorded that call
            # as a function of its own or not. Calls from ScratchPool are part of ScratchPool.
            for caller, edge in callers.items():
                if not is_scratch_pool(caller):
                    add_iaxl_time(profile, function_name(key), edge[1], edge[3])
                    profile.iaxl_s += edge[3]
            continue
        if is_iaxl_c_function(key):
            # Recorded on its own only when it was not reached through a wrapper.
            for caller, edge in callers.items():
                if not is_iaxl_wrapper(caller) and not is_scratch_pool(caller):
                    add_iaxl_time(profile, function_name(key), edge[1], edge[2])
                    profile.iaxl_s += edge[2]
            continue
        if is_scratch_pool(key):
            for caller, edge in callers.items():
                if not is_scratch_pool(caller):
                    profile.scratch_pool_s += edge[3]
            continue
        # Python overhead: the function's own time, without the part spent on behalf of ScratchPool.
        top_caller = ""
        top_caller_s = -1.0
        for caller, edge in callers.items():
            # The perf_counter_ns() calls of measured_call run while the profiler is on; they are
            # the measurement, not kvshrink.
            if is_scratch_pool(caller) or (is_c_function(key) and is_measurement_code(caller)):
                calls -= edge[1]
                own_s -= edge[2]
                continue
            if edge[2] > top_caller_s:
                top_caller_s = edge[2]
                top_caller = function_name(caller)
        if calls <= 0:
            continue
        profile.python_s += own_s
        if is_c_function(key):
            profile.c_calls += calls
        else:
            profile.python_calls += calls
        profile.functions.append(FunctionTime(entry, key, calls, own_s, top_caller))
    return profile


def check_iaxl_visible(profiles: Dict[str, EntryProfile], scenario: str) -> None:
    # A hit loads (start_load_kv), a miss saves (save_kv_layer); both call into IAXL. If no IAXL
    # time shows up there, the wrappers did not work and the C++ time sits in the Python time.
    entry = "start_load_kv" if scenario == "hit" else "save_kv_layer"
    profile = profiles.get(entry)
    if profile is None or profile.iaxl_s <= 0:
        raise ValueError(f"no IAXL C++ time recorded in {entry}; the IAXL wrappers did not take effect")


def read_json(path: str) -> Dict:
    with open(path) as file:
        return json.load(file)


def location_text(key: FunctionKey, repo_dir: str) -> str:
    path = key[0]
    if is_c_function(key):
        return "C 函数"
    if path.startswith(repo_dir + os.sep):
        path = os.path.relpath(path, repo_dir)
    else:
        for marker in ("dist-packages/", "site-packages/", "python3.12/"):
            index = path.find(marker)
            if index >= 0:
                path = path[index + len(marker):]
                break
    return f"{path}:{key[1]}"


def cell(text: str) -> str:
    # Keep table cells on one line and their backticks balanced.
    return text.replace("|", "\\|").replace("`", "'")


def build_results(profiles: Dict[str, EntryProfile], timers: Dict, cost_ns: Dict[str, float],
                  requests: int) -> List[EntryResult]:
    results: List[EntryResult] = []
    for entry, totals in timers["entries"].items():
        profile = profiles.get(entry)
        if profile is None:
            profile = EntryProfile(entry)
        cost_s = (profile.python_calls * cost_ns["python_call_ns"] + profile.c_calls * cost_ns["c_call_ns"]) / 1e9
        results.append(
            EntryResult(
                entry=entry,
                calls_per_request=totals["calls"] / requests,
                wall_ms=totals["wall_ns"] / 1e6 / requests,
                iaxl_ms=profile.iaxl_s * 1000 / requests,
                scratch_pool_ms=profile.scratch_pool_s * 1000 / requests,
                python_ms=(profile.python_s - cost_s) * 1000 / requests,
                python_raw_ms=profile.python_s * 1000 / requests,
            )
        )
    results.sort(key=lambda result: result.python_ms, reverse=True)
    return results


def write_summary(result_dir: str, meta: Dict, results: List[EntryResult],
                  profiles: Dict[str, EntryProfile], cost_ns: Dict[str, float],
                  bench_results: Dict[str, Dict[str, float]], log_checks: Dict, control_log: str,
                  main_thread_ms: Dict[str, float]) -> str:
    requests = meta["num_prompts"]
    repo_dir = meta["repo_dir"]
    lines: List[str] = []

    lines.append(f"# cProfile 结果:{meta['workload']},{meta['scenario']}")
    lines.append("")
    lines.append("## 这次运行是什么")
    lines.append("")
    lines.append(
        f"- vLLM {meta['vllm_version']} + kvshrink,模型 `{meta['model']}`,TP=1,GPU {meta['gpu']},"
        f"vLLM 绑定 CPU {meta['cpu_bind']},KV 压缩和 DSA 关闭。调度、模型计算和 kvshrink 都在 EngineCore 进程的主线程上。"
    )
    lines.append(
        f"- 一次 bench = `vllm bench serve` 发 {requests} 个请求,输入 {meta['input_len']} token、"
        f"输出 {meta['output_len']} token,并发 {meta['concurrency']}。"
    )
    if meta["scenario"] == "miss":
        lines.append("- 场景 miss:每个请求的 prompt 都是新的,kvshrink 只保存、不加载。")
    else:
        lines.append("- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。")
    lines.append(
        "- vLLM 加载的是 `ProfiledKVShrinkConnector`:kvshrink 代码不变,vLLM 每次调用 kvshrink 的方法"
        "(下文叫“入口函数”)都被记录。同一种请求跑两次 bench:计时 bench 只记每个入口函数的调用次数和耗时;"
        "cProfile bench 另外用 cProfile 记下入口函数里每个函数的调用次数和耗时,用来把时间分成 Python 开销、"
        "IAXL C++、ScratchPool 三类。"
    )
    lines.append(
        f"- 代码 git {meta['git_head']},代码 md5 {meta['code_md5']},Python {meta['python_version']}。"
    )
    lines.append("")

    lines.append("## 核对")
    lines.append("")
    lines.append("| 核对项 | 结果 |")
    lines.append("|---|---|")
    for bench_name in MEASURED_BENCHES:
        check = log_checks[bench_name]
        value_texts: List[str] = []
        for value in check.external_token_values[:5]:
            value_texts.append(str(value))
        lines.append(f"| {bench_name} bench 的请求数(connector 日志) | {check.requests} |")
        lines.append(f"| {bench_name} bench 里有外部命中的请求数 | {check.with_external_tokens} |")
        lines.append(f"| {bench_name} bench 里 externally-cached tokens 的不同取值 | {', '.join(value_texts)} |")
    lines.append(
        f"| cProfile 每记录一次调用多出的时间(Python 函数 / C 函数) | "
        f"{cost_ns['python_call_ns']:.0f} ns / {cost_ns['c_call_ns']:.0f} ns |"
    )
    lines.append(f"| connector 处理的控制命令 | {cell(control_log.strip().replace(chr(10), '; '))} |")
    lines.append("")
    lines.append(
        "列说明:“核对项”是检查什么,“结果”是从 connector 日志、控制日志数出来的值。miss 场景外部命中的请求数应为 0,"
        "hit 场景应等于请求数。“cProfile 每记录一次调用多出的时间”是 connector 在同一个进程里实测的:"
        "同样的空调用,开 cProfile 比不开多花的时间,表 1 用它扣除 cProfile 自身的开销。"
    )
    lines.append("")

    lines.append("### 计时 bench 和 cProfile bench 对比")
    lines.append("")
    lines.append("| 指标 | 计时 bench | cProfile bench | cProfile / 计时 |")
    lines.append("|---|---|---|---|")
    for metric in BENCH_METRICS:
        timers_value = bench_results["timers"][metric]
        cprofile_value = bench_results["cprofile"][metric]
        ratio_text = "-"
        if timers_value:
            ratio_text = f"{cprofile_value / timers_value:.3f}"
        lines.append(f"| {metric} | {timers_value:.2f} | {cprofile_value:.2f} | {ratio_text} |")
    lines.append("")
    lines.append(
        "列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,其余单位 ms,"
        "mean 是均值、median 是中位数);后两列是两次 bench 的值;最后一列是比值,说明 cProfile 把 vLLM 拖慢了多少。"
    )
    lines.append("")

    lines.append("## 表 1:按入口函数汇总(毫秒/请求)")
    lines.append("")
    lines.append("| 入口函数 | 每请求调用次数 | 总耗时 | 其中 IAXL C++ | 其中 ScratchPool | 其中 Python 开销 | 校验 |")
    lines.append("|---|---|---|---|---|---|---|")
    total_wall = 0.0
    total_iaxl = 0.0
    total_scratch = 0.0
    total_python = 0.0
    for result in results:
        check_text = "-"
        if result.wall_ms > 0:
            parts = result.python_ms + result.iaxl_ms + result.scratch_pool_ms
            check_text = f"{parts / result.wall_ms:.2f}"
        lines.append(
            f"| `{result.entry}` | {result.calls_per_request:.1f} | {result.wall_ms:.2f} | {result.iaxl_ms:.2f} | "
            f"{result.scratch_pool_ms:.2f} | {result.python_ms:.2f} | {check_text} |"
        )
        total_wall += result.wall_ms
        total_iaxl += result.iaxl_ms
        total_scratch += result.scratch_pool_ms
        total_python += result.python_ms
    lines.append(
        f"| 合计 | - | {total_wall:.2f} | {total_iaxl:.2f} | {total_scratch:.2f} | {total_python:.2f} | - |"
    )
    lines.append("")
    share_text = "-"
    if main_thread_ms["timers"] > 0:
        share_text = f"{total_python / main_thread_ms['timers'] * 100:.2f}%"
    lines.append(
        f"计时 bench 里 EngineCore 主线程的 CPU 时间(内核记的 utime + stime):{main_thread_ms['timers']:.2f} 毫秒/请求;"
        f"kvshrink 的 Python 开销占其中 {share_text}。"
    )
    lines.append("")
    lines.append(
        "列说明:单位都是毫秒/请求(这次 bench 的总量 ÷ 请求数),“每请求调用次数”除外。“入口函数”是 vLLM 调用的 kvshrink 方法;"
        "“每请求调用次数”和“总耗时”来自计时 bench,总耗时是这个方法从被调用到返回的时间,包括它里面的 IAXL C++ 和等待;"
        "后三列来自 cProfile bench:“IAXL C++”是调用 IAXL 扩展的时间(C++ 执行、pybind11 转换参数、等待),"
        "“ScratchPool”是取块还块(归同事),“Python 开销”是剩下的时间,即 kvshrink 的 Python 代码和它直接调用的"
        "xxhash、numpy、torch 等,已扣除 cProfile 自身的开销;“校验”=(后三列之和)÷ 总耗时,两次 bench 测的是同样的请求,"
        "接近 1 说明扣除得对、三类加起来就是总耗时。"
    )
    lines.append("")

    factors: Dict[str, float] = {}
    for result in results:
        factors[result.entry] = 0.0
        if result.python_raw_ms > 0:
            factors[result.entry] = result.python_ms / result.python_raw_ms
    functions: List[FunctionTime] = []
    for profile in profiles.values():
        for function_time in profile.functions:
            functions.append(function_time)
    functions.sort(key=lambda item: item.seconds * factors.get(item.entry, 0.0), reverse=True)

    lines.append(f"## 表 2:Python 开销最多的前 {TOP_FUNCTIONS} 个函数")
    lines.append("")
    lines.append("| 入口函数 | 函数 | 位置 | 每请求调用次数 | 每次调用(µs) | 毫秒/请求 |")
    lines.append("|---|---|---|---|---|---|")
    for function_time in functions[:TOP_FUNCTIONS]:
        factor = factors.get(function_time.entry, 0.0)
        milliseconds = function_time.seconds * factor * 1000 / requests
        microseconds_per_call = function_time.seconds * factor * 1e6 / function_time.calls
        name = function_name(function_time.key)
        if is_c_function(function_time.key):
            name = f"{name}(由 {function_time.top_caller} 调用)"
        lines.append(
            f"| `{function_time.entry}` | `{cell(name)}` | `{cell(location_text(function_time.key, repo_dir))}` | "
            f"{function_time.calls / requests:.1f} | {microseconds_per_call:.2f} | {milliseconds:.2f} |"
        )
    lines.append("")
    lines.append(
        "列说明:每一行是一个函数自己的时间,不含它再调用的 Python 函数(那些单独成行);C 函数(如 `repr`、xxhash)"
        "单独成行并写出调用它的函数。“入口函数”同表 1;“位置”是函数定义所在的文件和行号;“每请求调用次数”是 cProfile bench "
        "里的调用次数 ÷ 请求数;“毫秒/请求”和“每次调用”已按所在入口函数扣除 cProfile 的开销(乘以表 1 的 Python 开销 ÷ "
        "附 2 的 cProfile 原始数)。"
    )
    lines.append("")

    iaxl_methods: List[IaxlMethodTime] = []
    for profile in profiles.values():
        for method in profile.iaxl_methods.values():
            iaxl_methods.append(method)
    iaxl_methods.sort(key=lambda item: item.seconds, reverse=True)
    lines.append("## 附 1:IAXL C++ 各方法(对照,不算 Python 开销)")
    lines.append("")
    lines.append("| 入口函数 | IAXL 方法 | 每请求调用次数 | 毫秒/请求 |")
    lines.append("|---|---|---|---|")
    for method in iaxl_methods:
        lines.append(
            f"| `{method.entry}` | `{cell(method.label)}` | {method.calls / requests:.1f} | "
            f"{method.seconds * 1000 / requests:.2f} |"
        )
    lines.append("")
    lines.append("列说明:“IAXL 方法”是被调用的 IAXL 扩展方法;其余同表 2,时间包括 pybind11 转换参数和 C++ 里的等待。")
    lines.append("")

    lines.append("## 附 2:cProfile 原始数(未扣除 cProfile 自身开销)")
    lines.append("")
    lines.append("| 入口函数 | cProfile 记录的总时间 | Python 开销(原始) | 每请求记录的 Python 函数调用 | 每请求记录的 C 函数调用 |")
    lines.append("|---|---|---|---|---|")
    for result in results:
        profile = profiles.get(result.entry)
        if profile is None:
            continue
        lines.append(
            f"| `{result.entry}` | {profile.total_s * 1000 / requests:.2f} | {result.python_raw_ms:.2f} | "
            f"{profile.python_calls / requests:.0f} | {profile.c_calls / requests:.0f} |"
        )
    lines.append("")
    lines.append(
        "列说明:时间单位毫秒/请求,来自 cProfile bench;“Python 开销(原始)”减去 调用次数 × 每次调用多出的时间(核对表)"
        "就是表 1 的 Python 开销。"
    )
    lines.append("")

    summary_path = os.path.join(result_dir, "summary.md")
    with open(summary_path, "w") as file:
        file.write("\n".join(lines))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir")
    args = parser.parse_args()
    result_dir = os.path.abspath(args.result_dir)

    meta = read_json(os.path.join(result_dir, "meta.json"))
    requests = meta["num_prompts"]
    timers = read_json(os.path.join(result_dir, "profile_timers.json"))
    cprofile_results = read_json(os.path.join(result_dir, "profile_cprofile.json"))
    cost_ns = cprofile_results["cprofile_cost_ns"]

    profiles: Dict[str, EntryProfile] = {}
    for entry in cprofile_results["entries"]:
        path = os.path.join(result_dir, f"cprofile_{entry}.pstats")
        profiles[entry] = load_entry_profile(entry, path)
    check_iaxl_visible(profiles, meta["scenario"])
    results = build_results(profiles, timers, cost_ns, requests)

    log_path = os.path.join(result_dir, "vllm.log")
    bench_ranges = read_bench_ranges(os.path.join(result_dir, "bench_offsets.txt"))
    cpu_ticks = read_cpu_ticks(os.path.join(result_dir, "bench_cpu_ticks.txt"))
    log_checks: Dict[str, MatchedTokens] = {}
    bench_results: Dict[str, Dict[str, float]] = {}
    main_thread_ms: Dict[str, float] = {}
    for bench_name in MEASURED_BENCHES:
        log_checks[bench_name] = matched_tokens(read_log_range(log_path, bench_ranges[bench_name]))
        bench_results[bench_name] = read_bench_result(os.path.join(result_dir, f"bench_{bench_name}.json"))
        ticks = cpu_ticks[bench_name]
        main_thread_ms[bench_name] = (
            (ticks.main_thread_end - ticks.main_thread_start) / meta["clock_ticks_per_second"] * 1000 / requests
        )
    with open(os.path.join(result_dir, "control.log")) as file:
        control_log = file.read()

    summary_path = write_summary(
        result_dir, meta, results, profiles, cost_ns, bench_results, log_checks, control_log, main_thread_ms
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
