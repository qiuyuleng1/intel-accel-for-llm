# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Writes summary.md for one result directory of run_pyspy_bench.sh.

py-spy runs without --native, so the innermost frame of a sample is always Python: time spent in a
C/C++ function shows up on the Python line that called it. Every sample of the EngineCore main
thread whose call chain contains kvshrink code (a .py file under kvshrink/ or iaxl/) is put in one
category by the innermost kvshrink frame:
  ScratchPool   it is in iaxl/kvflow/scratch_pool.py
  IAXL C++      its line calls into IAXL's extension (a Context, Record, Mem or Storage method, or a
                task_queue_* function); this includes pybind11 converting the arguments
  Python 开销    anything else: kvshrink's Python statements, the Python code they call and the C
                code of the libraries they call directly (xxhash, numpy, torch, ...)
A sample stands for 1 / rate seconds of its thread; times are reported in ms per request.

    python3 analyze_pyspy.py <result dir>
"""

import argparse
import ast
import json
import linecache
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

PYTHON_OVERHEAD = "Python 开销"
SCRATCH_POOL = "ScratchPool"
IAXL_CPP = "IAXL C++"
SCRATCH_POOL_FILE = os.path.join("iaxl", "kvflow", "scratch_pool.py")
KVSHRINK_PACKAGES = ("kvshrink", "iaxl")
MAIN_THREAD = "MainThread"
# The outermost frame of a complete EngineCore main-thread call chain (multiprocessing spawn).
MAIN_THREAD_ROOT = "<module>"
# Names that hold IAXL extension objects in kvshrink's code: a method called on them runs in C++.
EXTENSION_RECEIVERS = {"ctx", "Context", "_iqt", "record", "mem", "storage"}
# IAXL extension functions and classes that kvshrink's code calls by name.
EXTENSION_FUNCTIONS = {
    "task_queue_set_bundles_priority",
    "task_queue_sort_priority_level",
    "Record",
    "Mem",
    "Storage",
}
TOP_LOCATIONS = 20
TOP_EXTENSION_LINES = 15
# The connector logs one such line each time the scheduler asks how much of a request is cached.
MATCHED_TOKENS_LOG = re.compile(
    r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+)"
)
BENCH_METRICS = [
    "duration",
    "completed",
    "request_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
]


@dataclass
class Frame:
    name: str
    file: str
    line: int  # 0 when py-spy reports no line number


@dataclass
class Sample:
    thread: str
    frames: List[Frame]  # outermost first
    count: int


@dataclass
class Classified:
    category: str
    entry_function: str  # outermost kvshrink frame: the function vLLM called
    relative_file: str  # innermost kvshrink frame, file relative to the repo
    line: int
    function: str


@dataclass
class CategoryTotals:
    samples: Dict[str, int] = field(default_factory=dict)

    def add(self, category: str, count: int) -> None:
        self.samples[category] = self.samples.get(category, 0) + count

    def get(self, category: str) -> int:
        return self.samples.get(category, 0)

    def total(self) -> int:
        result = 0
        for count in self.samples.values():
            result += count
        return result


@dataclass
class LocationTotals:
    relative_file: str
    line: int
    function: str
    entry_function: str
    samples: int = 0


@dataclass
class Profile:
    main_thread_samples: int = 0
    # Main-thread samples whose call chain misses its outermost frames: --nonblocking reads the
    # stack while the thread runs, so a chain is sometimes cut off.
    partial_main_thread_samples: int = 0
    main_thread_kvshrink: CategoryTotals = field(default_factory=CategoryTotals)
    other_threads_kvshrink: CategoryTotals = field(default_factory=CategoryTotals)
    by_entry: Dict[str, CategoryTotals] = field(default_factory=dict)
    # Python overhead per code location and entry function
    by_location: Dict[str, LocationTotals] = field(default_factory=dict)
    # IAXL C++ samples per code location, to check which lines were counted as C++
    extension_locations: Dict[str, LocationTotals] = field(default_factory=dict)
    thread_samples: Dict[str, int] = field(default_factory=dict)


@dataclass
class BenchRange:
    start: int  # byte offsets of vllm.log before and after the bench
    end: int


@dataclass
class MatchedTokens:
    requests: int
    with_external_tokens: int
    external_token_values: List[int]  # distinct values, sorted


@dataclass
class CpuTicks:
    """Kernel CPU time (utime + stime, clock ticks) of EngineCore before and after one bench."""
    main_thread_start: int
    main_thread_end: int
    process_start: int
    process_end: int


def parse_frame(text: str) -> Frame:
    # py-spy writes a frame as "name (file:line)", "name (file)" or "name".
    if not text.endswith(")"):
        return Frame(name=text, file="", line=0)
    open_index = text.rfind(" (")
    if open_index < 0:
        return Frame(name=text, file="", line=0)
    location = text[open_index + 2 : -1]
    file = location
    line = 0
    colon_index = location.rfind(":")
    if colon_index >= 0 and location[colon_index + 1 :].isdigit():
        file = location[:colon_index]
        line = int(location[colon_index + 1 :])
    return Frame(name=text[:open_index], file=file, line=line)


def parse_thread(text: str) -> str:
    # --threads puts "thread (0x7F3A2C1B7740): MainThread" in front of every call chain.
    if not text.startswith("thread ("):
        raise ValueError(f"call chain does not start with a thread (py-spy without --threads?): {text}")
    name_index = text.find("): ")
    if name_index < 0:
        return text
    return text[name_index + 3 :]


def read_samples(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            stack_text, count_text = line.rsplit(" ", 1)
            frame_texts = stack_text.split(";")
            frames: List[Frame] = []
            for frame_text in frame_texts[1:]:
                frames.append(parse_frame(frame_text))
            samples.append(Sample(parse_thread(frame_texts[0]), frames, int(count_text)))
    return samples


def is_kvshrink_frame(frame: Frame, repo_dir: str) -> bool:
    if not frame.file.endswith(".py"):
        return False
    for package in KVSHRINK_PACKAGES:
        if frame.file.startswith(os.path.join(repo_dir, package) + os.sep):
            return True
    return False


def calls_extension(call: ast.Call) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id in EXTENSION_FUNCTIONS
    if not isinstance(function, ast.Attribute):
        return False
    receiver = function.value
    if isinstance(receiver, ast.Name):
        return receiver.id in EXTENSION_RECEIVERS
    if isinstance(receiver, ast.Attribute):
        return receiver.attr in EXTENSION_RECEIVERS
    return False


def find_extension_lines(repo_dir: str) -> Dict[str, Set[int]]:
    """Lines of kvshrink's Python files that call into IAXL's extension, per relative file."""
    extension_lines: Dict[str, Set[int]] = {}
    for package in KVSHRINK_PACKAGES:
        for directory, _, file_names in os.walk(os.path.join(repo_dir, package)):
            for file_name in file_names:
                if not file_name.endswith(".py"):
                    continue
                path = os.path.join(directory, file_name)
                with open(path) as file:
                    tree = ast.parse(file.read(), filename=path)
                lines: Set[int] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and calls_extension(node):
                        # While a call runs, Python 3.12 reports the line where the called name
                        # ends (the method name of a multi-line call); keep the first line too.
                        lines.add(node.lineno)
                        lines.add(node.func.end_lineno)
                extension_lines[os.path.relpath(path, repo_dir)] = lines
    return extension_lines


def classify(sample: Sample, repo_dir: str, extension_lines: Dict[str, Set[int]]) -> Optional[Classified]:
    outermost = -1
    innermost = -1
    for index in range(len(sample.frames)):
        if is_kvshrink_frame(sample.frames[index], repo_dir):
            if outermost < 0:
                outermost = index
            innermost = index
    if innermost < 0:
        return None

    inner_frame = sample.frames[innermost]
    relative_file = os.path.relpath(inner_frame.file, repo_dir)
    if relative_file == SCRATCH_POOL_FILE:
        category = SCRATCH_POOL
    elif inner_frame.line in extension_lines.get(relative_file, set()):
        category = IAXL_CPP
    else:
        category = PYTHON_OVERHEAD
    return Classified(
        category=category,
        entry_function=sample.frames[outermost].name,
        relative_file=relative_file,
        line=inner_frame.line,
        function=inner_frame.name,
    )


def add_location(locations: Dict[str, LocationTotals], classified: Classified, count: int) -> None:
    key = f"{classified.relative_file}:{classified.line}|{classified.entry_function}"
    if key not in locations:
        locations[key] = LocationTotals(
            relative_file=classified.relative_file,
            line=classified.line,
            function=classified.function,
            entry_function=classified.entry_function,
        )
    locations[key].samples += count


def build_profile(samples: List[Sample], repo_dir: str, extension_lines: Dict[str, Set[int]]) -> Profile:
    profile = Profile()
    for sample in samples:
        profile.thread_samples[sample.thread] = (
            profile.thread_samples.get(sample.thread, 0) + sample.count
        )
        if sample.thread == MAIN_THREAD:
            profile.main_thread_samples += sample.count
            if not sample.frames or sample.frames[0].name != MAIN_THREAD_ROOT:
                profile.partial_main_thread_samples += sample.count
        classified = classify(sample, repo_dir, extension_lines)
        if classified is None:
            continue
        if sample.thread != MAIN_THREAD:
            profile.other_threads_kvshrink.add(classified.category, sample.count)
            continue

        profile.main_thread_kvshrink.add(classified.category, sample.count)
        if classified.entry_function not in profile.by_entry:
            profile.by_entry[classified.entry_function] = CategoryTotals()
        profile.by_entry[classified.entry_function].add(classified.category, sample.count)
        if classified.category == PYTHON_OVERHEAD:
            add_location(profile.by_location, classified, sample.count)
        elif classified.category == IAXL_CPP:
            add_location(profile.extension_locations, classified, sample.count)
    return profile


def read_bench_ranges(path: str) -> Dict[str, BenchRange]:
    ranges: Dict[str, BenchRange] = {}
    with open(path) as file:
        for line in file:
            name, start, end = line.split()
            ranges[name] = BenchRange(int(start), int(end))
    return ranges


def read_cpu_ticks(path: str) -> Dict[str, CpuTicks]:
    ticks: Dict[str, CpuTicks] = {}
    with open(path) as file:
        for line in file:
            name, main_start, main_end, process_start, process_end = line.split()
            ticks[name] = CpuTicks(int(main_start), int(main_end), int(process_start), int(process_end))
    return ticks


def read_log_range(log_path: str, bench_range: BenchRange) -> str:
    # By byte offset: the log has '\r' progress output, so line numbers would not line up.
    with open(log_path, "rb") as file:
        file.seek(bench_range.start)
        data = file.read(bench_range.end - bench_range.start)
    return data.decode("utf-8", errors="replace")


def matched_tokens(log_text: str) -> MatchedTokens:
    # A request still waiting for KV blocks is asked again in later steps; the last answer is the
    # one it was scheduled with.
    tokens_by_request: Dict[str, int] = {}
    for match in MATCHED_TOKENS_LOG.finditer(log_text):
        tokens_by_request[match.group(1)] = int(match.group(2))
    with_external_tokens = 0
    distinct_values = set()
    for tokens in tokens_by_request.values():
        if tokens > 0:
            with_external_tokens += 1
        distinct_values.add(tokens)
    return MatchedTokens(
        requests=len(tokens_by_request),
        with_external_tokens=with_external_tokens,
        external_token_values=sorted(distinct_values),
    )


def read_bench_result(path: str) -> Dict[str, float]:
    with open(path) as file:
        data = json.load(file)
    result: Dict[str, float] = {}
    for metric in BENCH_METRICS:
        if metric not in data:
            raise ValueError(f"{path} has no {metric}")
        result[metric] = data[metric]
    return result


def ms_per_request(samples: int, rate: int, requests: int) -> float:
    return samples / rate * 1000 / requests


def source_text(repo_dir: str, relative_file: str, line: int) -> str:
    text = linecache.getline(os.path.join(repo_dir, relative_file), line).strip()
    # Keep table cells on one line and their backticks balanced.
    return text.replace("|", "\\|").replace("`", "'")


def location_row(location_totals: LocationTotals, repo_dir: str, time_text: str, share_text: str) -> str:
    location = f"{location_totals.relative_file}:{location_totals.line}"
    code = source_text(repo_dir, location_totals.relative_file, location_totals.line)
    return (
        f"| `{location}` | `{code}` | `{location_totals.function}` | `{location_totals.entry_function}` | "
        f"{time_text} | {share_text} | {location_totals.samples} |"
    )


def write_summary(result_dir: str, meta: Dict, profile: Profile, pyspy_stdout: str,
                  log_checks: Dict[str, MatchedTokens], exhausted_warnings: int,
                  bench_results: Dict[str, Dict[str, float]], cpu_ticks: Dict[str, CpuTicks]) -> str:
    rate = meta["pyspy_rate"]
    requests = meta["num_prompts"]
    repo_dir = meta["repo_dir"]
    lines: List[str] = []

    def time_cell(samples: int) -> str:
        return f"{ms_per_request(samples, rate, requests):.2f}"

    def entry_row(label: str, totals: CategoryTotals, calibrated_text: str) -> str:
        return (
            f"| {label} | {time_cell(totals.get(PYTHON_OVERHEAD))} | {calibrated_text} | "
            f"{time_cell(totals.get(SCRATCH_POOL))} | {time_cell(totals.get(IAXL_CPP))} | {totals.total()} |"
        )

    lines.append(f"# py-spy 结果:{meta['workload']},{meta['scenario']}")
    lines.append("")
    lines.append("## 这次运行是什么")
    lines.append("")
    lines.append(
        f"- vLLM {meta['vllm_version']} + kvshrink,模型 `{meta['model']}`,TP=1,GPU {meta['gpu']},"
        f"vLLM 进程绑定 CPU {meta['cpu_bind']}。调度、模型计算和 kvshrink 的全部函数都在 EngineCore "
        "进程的主线程上。"
    )
    lines.append(
        f"- 一次 bench = `vllm bench serve` 发 {requests} 个请求,输入 {meta['input_len']} token、"
        f"输出 {meta['output_len']} token,并发 {meta['concurrency']}。"
    )
    if meta["scenario"] == "miss":
        lines.append("- 场景 miss:每个请求的 prompt 都是新的,kvshrink 只保存、不加载。")
    else:
        lines.append(
            "- 场景 hit:所有请求的 prompt 相同,测量前已存进 DDR,kvshrink 只加载、不保存。"
        )
    if meta["pyspy_mode"] == "nonblocking":
        mode_text = "不暂停被测进程"
    else:
        mode_text = "每次采样把被测进程暂停一下再读调用栈"
    lines.append(
        f"- py-spy 从测量 bench 开始前采到结束后,采样率 {rate} 次/秒,只采 EngineCore 进程"
        f"(pid {meta['engine_core_pid']}),参数 `{meta['pyspy_args']}`:{mode_text},不记录 C/C++ 函数,"
        "在 C/C++ 里花的时间算在调用它的那一行 Python 上。一个样本约代表 "
        f"{1000 / rate:.0f} ms;“毫秒/请求”= 样本数 ÷ 采样率 × 1000 ÷ 请求数。相对误差约 1/√样本数。"
    )
    lines.append(f"- 代码 git {meta['git_head']},代码 md5 {meta['code_md5']},py-spy {meta['pyspy_version']}。")
    lines.append("")

    lines.append("## 核对")
    lines.append("")
    lines.append("| 核对项 | 结果 |")
    lines.append("|---|---|")
    for bench_name in ("baseline", "pyspy"):
        check = log_checks[bench_name]
        lines.append(
            f"| {bench_name} 这次 bench 的请求数(connector 日志) | {check.requests} |"
        )
        lines.append(
            f"| {bench_name} 里有外部命中(externally-cached tokens > 0)的请求数 | "
            f"{check.with_external_tokens} |"
        )
        value_texts: List[str] = []
        for value in check.external_token_values[:5]:
            value_texts.append(str(value))
        values_text = ", ".join(value_texts)
        lines.append(f"| {bench_name} 里 externally-cached tokens 的不同取值(前 5 个) | {values_text} |")
    lines.append(f"| vLLM 日志里 `ScratchPool exhausted` 的次数 | {exhausted_warnings} |")
    behind_count = pyspy_stdout.count("behind in sampling")
    lines.append(f"| py-spy 跟不上采样率的警告次数 | {behind_count} |")
    summary_match = re.search(r"Samples: (\d+) Errors: (\d+)", pyspy_stdout)
    if summary_match:
        lines.append(
            f"| py-spy 自报的样本数 / 错误数 | {summary_match.group(1)} / {summary_match.group(2)} |"
        )
    partial_share = "-"
    if profile.main_thread_samples:
        partial_share = f"{profile.partial_main_thread_samples / profile.main_thread_samples * 100:.1f}%"
    lines.append(
        f"| 主线程样本里调用链不完整的样本数(占比) | {profile.partial_main_thread_samples}({partial_share}) |"
    )
    lines.append("")
    lines.append(
        "列说明:“核对项”是检查什么;“结果”是从 connector 日志、vLLM 日志或 py-spy 输出数出来的值。"
        "miss 场景外部命中的请求数应为 0;hit 场景应等于请求数。"
        "“调用链不完整”指最外层不是进程入口 `<module>` 的样本:py-spy 不暂停进程时,偶尔读到一半就变了的调用链。"
    )
    lines.append("")

    lines.append("### 有没有 py-spy 的 bench 对比(干扰检查)")
    lines.append("")
    lines.append("| 指标 | 没有 py-spy | 有 py-spy | 有 / 没有 |")
    lines.append("|---|---|---|---|")
    for metric in BENCH_METRICS:
        baseline_value = bench_results["baseline"][metric]
        pyspy_value = bench_results["pyspy"][metric]
        ratio_text = "-"
        if baseline_value:
            ratio_text = f"{pyspy_value / baseline_value:.3f}"
        lines.append(f"| {metric} | {baseline_value:.2f} | {pyspy_value:.2f} | {ratio_text} |")
    lines.append("")
    lines.append(
        "列说明:“指标”是 `vllm bench serve` 结果里的字段(duration 单位秒,throughput 单位请求/秒,"
        "其余单位 ms,mean 是均值、median 是中位数,统计对象是这次 bench 的全部请求);"
        "后两列是两次 bench 的值;“有 / 没有”是比值,越接近 1 说明 py-spy 干扰越小。"
    )
    lines.append("")

    clock_ticks = meta["clock_ticks_per_second"]

    def ticks_cell(start: int, end: int) -> str:
        return f"{(end - start) / clock_ticks * 1000 / requests:.2f}"

    lines.append("### py-spy 数的主线程忙碌时间对不对(和内核记的 CPU 时间比)")
    lines.append("")
    lines.append("| 量 | 没有 py-spy 的 bench | 有 py-spy 的 bench |")
    lines.append("|---|---|---|")
    baseline_ticks = cpu_ticks["baseline"]
    pyspy_ticks = cpu_ticks["pyspy"]
    lines.append(
        f"| EngineCore 主线程 CPU 时间(内核) | "
        f"{ticks_cell(baseline_ticks.main_thread_start, baseline_ticks.main_thread_end)} | "
        f"{ticks_cell(pyspy_ticks.main_thread_start, pyspy_ticks.main_thread_end)} |"
    )
    lines.append(
        f"| EngineCore 全进程 CPU 时间(内核) | "
        f"{ticks_cell(baseline_ticks.process_start, baseline_ticks.process_end)} | "
        f"{ticks_cell(pyspy_ticks.process_start, pyspy_ticks.process_end)} |"
    )
    lines.append(f"| py-spy 数的主线程忙碌时间 | - | {time_cell(profile.main_thread_samples)} |")
    kernel_main_ms = (
        (pyspy_ticks.main_thread_end - pyspy_ticks.main_thread_start) / clock_ticks * 1000 / requests
    )
    # py-spy loses (nonblocking) or adds (pause) busy main-thread samples; scaling them to the kernel's
    # CPU time assumes the lost or added samples are spread evenly over the code.
    calibration = kernel_main_ms / ms_per_request(profile.main_thread_samples, rate, requests)
    lines.append(f"| 校准系数 = 内核 ÷ py-spy(主线程) | - | {calibration:.3f} |")
    lines.append("")
    lines.append(
        "列说明:单位都是毫秒/请求(校准系数除外)。“内核”两行是 /proc 里 EngineCore 的 utime + stime(内核记账的、真正"
        "占用 CPU 的时间),在每次 bench 前后各读一次,差值 ÷ 请求数;主线程的线程号就是进程号,全进程包括"
        "IAXL、torch 等全部线程。“py-spy 数的主线程忙碌时间”= 主线程被判为忙的样本数 ÷ 采样率 ÷ 请求数。"
        "py-spy 采样窗口比 bench 略长(前后各多几秒,那时 EngineCore 没有请求),"
        "所以它和同一次 bench 的“主线程 CPU 时间(内核)”应基本相等;差很多说明 py-spy 漏采或多采。"
        "“校准系数”= 有 py-spy 的 bench 的“主线程 CPU 时间(内核)”÷“py-spy 数的主线程忙碌时间”,"
        "表 1 用它把 py-spy 直接数的主线程时间换算成内核口径。"
    )
    lines.append("")

    def calibrated_cell(samples: int) -> str:
        return f"{ms_per_request(samples, rate, requests) * calibration:.2f}"

    lines.append("## 表 1:按入口函数汇总(毫秒/请求)")
    lines.append("")
    lines.append(
        "| 入口函数 | Python 开销(py-spy 直接数) | Python 开销(按内核校准) | 对照:ScratchPool | "
        "对照:IAXL C++(含参数转换) | 样本数 |"
    )
    lines.append("|---|---|---|---|---|---|")
    ordered_entries = sorted(
        profile.by_entry,
        key=lambda entry_function: profile.by_entry[entry_function].get(PYTHON_OVERHEAD),
        reverse=True,
    )
    for entry_function in ordered_entries:
        entry_totals = profile.by_entry[entry_function]
        lines.append(
            entry_row(f"`{entry_function}`", entry_totals, calibrated_cell(entry_totals.get(PYTHON_OVERHEAD)))
        )
    main_kvshrink_overhead = profile.main_thread_kvshrink.get(PYTHON_OVERHEAD)
    lines.append(
        entry_row("主线程上 kvshrink 合计", profile.main_thread_kvshrink, calibrated_cell(main_kvshrink_overhead))
    )
    # The calibration was measured for the main thread only.
    lines.append(entry_row("其他线程上的 kvshrink", profile.other_threads_kvshrink, "-"))
    lines.append("")
    share_text = "-"
    if profile.main_thread_samples:
        share_text = f"{main_kvshrink_overhead / profile.main_thread_samples * 100:.1f}%"
    lines.append(
        f"主线程 CPU 忙碌总时间(含 vLLM 自己的代码):py-spy 直接数 {time_cell(profile.main_thread_samples)} 毫秒/请求"
        f"({profile.main_thread_samples} 个样本),内核记 {kernel_main_ms:.2f} 毫秒/请求;"
        f"kvshrink 的 Python 开销占 py-spy 主线程忙碌样本的 {share_text}。"
    )
    lines.append("")
    lines.append(
        "列说明:“入口函数”是调用链里最外面的 kvshrink 函数,即 vLLM 调用的那个;"
        "“Python 开销”是本次要量的量:kvshrink 的 Python 语句,加上它们直接调用的 Python 代码和"
        "xxhash、numpy、torch 等库的 C 代码;“py-spy 直接数”= 样本数 ÷ 采样率 ÷ 请求数;"
        "“按内核校准”= 直接数的值 × 校准系数(见上一节),假设 py-spy 漏掉的样本均匀分布在各段代码上,"
        "其他线程不校准(校准系数只对主线程测过);两个“对照”列不算 Python 开销,是 py-spy 直接数的值:"
        "ScratchPool 取块还块归同事,IAXL C++ 是调用 IAXL 扩展的那些行上的时间(C++ 执行、pybind11 转换"
        "参数、空转等待,哪些行见附 1);以上单位都是毫秒/请求;“样本数”是这一行三类的样本总数。"
        "“其他线程上的 kvshrink”是管理接口、内存池淘汰等后台线程。"
    )
    lines.append("")

    lines.append(f"## 表 2:Python 开销最多的前 {TOP_LOCATIONS} 个代码位置")
    lines.append("")
    lines.append("| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 Python 开销的比例 | 样本数 |")
    lines.append("|---|---|---|---|---|---|---|")
    ordered_locations = sorted(
        profile.by_location.values(),
        key=lambda location_totals: location_totals.samples,
        reverse=True,
    )
    for location_totals in ordered_locations[:TOP_LOCATIONS]:
        share = "-"
        if main_kvshrink_overhead:
            share = f"{location_totals.samples / main_kvshrink_overhead * 100:.1f}%"
        lines.append(location_row(location_totals, repo_dir, time_cell(location_totals.samples), share))
    lines.append("")
    lines.append(
        "列说明:“代码位置”是最里面的 kvshrink 帧的文件和行号,即那一刻正在执行的 kvshrink 代码;"
        "“源代码”是这一行的内容;“所在函数”是这一行所在的函数;“入口函数”同表 1;"
        "“毫秒/请求”是这一行的 Python 开销;“占 Python 开销的比例”是这一行 ÷ 主线程上 kvshrink 的"
        "Python 开销合计;“样本数”是这一行的样本数。"
    )
    lines.append("")

    lines.append(f"## 附 1:算作 IAXL C++ 的代码行(前 {TOP_EXTENSION_LINES} 个,核对归类用)")
    lines.append("")
    lines.append("| 代码位置 | 源代码 | 所在函数 | 入口函数 | 毫秒/请求 | 占 IAXL C++ 的比例 | 样本数 |")
    lines.append("|---|---|---|---|---|---|---|")
    main_extension = profile.main_thread_kvshrink.get(IAXL_CPP)
    ordered_extension = sorted(
        profile.extension_locations.values(),
        key=lambda location_totals: location_totals.samples,
        reverse=True,
    )
    for location_totals in ordered_extension[:TOP_EXTENSION_LINES]:
        share = "-"
        if main_extension:
            share = f"{location_totals.samples / main_extension * 100:.1f}%"
        lines.append(location_row(location_totals, repo_dir, time_cell(location_totals.samples), share))
    lines.append("")
    lines.append(
        "列说明:同表 2,只是这里列的是被算作 IAXL C++ 的行(调用 IAXL 扩展的行);"
        "“占 IAXL C++ 的比例”是这一行 ÷ 主线程上 IAXL C++ 的合计。"
    )
    lines.append("")

    lines.append("## 附 2:各线程的样本数")
    lines.append("")
    lines.append("| 线程 | 样本数 |")
    lines.append("|---|---|")
    thread_samples = profile.thread_samples
    for thread_name in sorted(thread_samples, key=thread_samples.get, reverse=True):
        lines.append(f"| {thread_name} | {thread_samples[thread_name]} |")
    lines.append("")
    lines.append("列说明:“线程”是 py-spy 记录的线程名;“样本数”是这个线程在忙(没睡眠、没阻塞)时被采到的次数。")
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

    with open(os.path.join(result_dir, "meta.json")) as file:
        meta = json.load(file)
    samples = read_samples(os.path.join(result_dir, "pyspy_enginecore.txt"))
    profile = build_profile(samples, meta["repo_dir"], find_extension_lines(meta["repo_dir"]))
    with open(os.path.join(result_dir, "pyspy_enginecore.stdout"), errors="replace") as file:
        pyspy_stdout = file.read()

    log_path = os.path.join(result_dir, "vllm.log")
    bench_ranges = read_bench_ranges(os.path.join(result_dir, "bench_offsets.txt"))
    log_checks: Dict[str, MatchedTokens] = {}
    bench_results: Dict[str, Dict[str, float]] = {}
    for bench_name in ("baseline", "pyspy"):
        log_checks[bench_name] = matched_tokens(read_log_range(log_path, bench_ranges[bench_name]))
        bench_results[bench_name] = read_bench_result(os.path.join(result_dir, f"bench_{bench_name}.json"))
    with open(log_path, "rb") as file:
        exhausted_warnings = file.read().count(b"ScratchPool exhausted")
    cpu_ticks = read_cpu_ticks(os.path.join(result_dir, "bench_cpu_ticks.txt"))

    summary_path = write_summary(
        result_dir, meta, profile, pyspy_stdout, log_checks, exhausted_warnings, bench_results, cpu_ticks
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
