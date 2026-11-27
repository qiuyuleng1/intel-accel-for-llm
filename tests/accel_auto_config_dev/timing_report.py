"""Turns the [timing] lines of accel_auto_config_timed.py into per-step tables (ms).
Usage: timing_report.py <stderr file>"""

import re
import sys

TIMING_LINE = re.compile(r"^\[timing\]\s+([\d.]+) (\S+) (.*)$")
MEASURE_STEPS = ("build_cuda_batch_operations", "cuda_batch_copies", "slice_copier_register", "dsa_copies",
                 "slice_copier_close")
WORKER_STEPS = ("forked", "import_iaxl_affinity", "bind_cpu_and_dsa", "pinned_host_alloc_with_cuda_init",
                "gpu_alloc_and_stream")


def main() -> int:
    events = {}
    with open(sys.argv[1], errors="replace") as stderr_file:
        for line in stderr_file:
            match = TIMING_LINE.match(line.strip())
            if match:
                events.setdefault(match.group(2), []).append((float(match.group(1)), match.group(3)))

    main_steps = {}
    for seconds, label in events.get("main", []):
        if label.startswith("step "):
            main_steps[label[5:]] = seconds
    ranks = []
    for who in events:
        if who.startswith("dsa_rank"):
            ranks.append(who)
    ranks.sort(key=lambda who: int(who[8:]))

    rows = {}
    row_order = ["fork (main fork -> worker running)", "import iaxl.utils.affinity", "bind CPU and DSA",
                 "pinned 32 MiB host alloc (first CUDA call: CUDA init)", "32 MiB GPU alloc + stream",
                 "wait for first block size", "build CUDA batch operations (sum)", "CUDA batch copies (sum)",
                 "SliceCopier register (sum)", "DSA copies (sum)", "SliceCopier close (sum)",
                 "  of which start_together waits (sum)", "wait for next block size (sum)",
                 "last measure -> worker done", "worker total (fork -> done)"]
    for name in row_order:
        rows[name] = []
    measure_counts = []
    for who in ranks:
        step_seconds = {}
        measures = []
        wait_total = 0.0
        for seconds, label in events[who]:
            if label.startswith("step "):
                step_seconds[label[5:]] = seconds
            elif label.startswith("measure_start "):
                measures.append({"start": seconds})
            elif label.startswith("measure "):
                measures[-1][label[8:]] = seconds
            elif label.startswith("wait "):
                wait_total += float(label.split()[1])
        measure_counts.append(len(measures))
        rows["fork (main fork -> worker running)"].append(step_seconds["forked"] - main_steps["fork_dsa_workers"])
        rows["import iaxl.utils.affinity"].append(step_seconds["import_iaxl_affinity"] - step_seconds["forked"])
        rows["bind CPU and DSA"].append(step_seconds["bind_cpu_and_dsa"] - step_seconds["import_iaxl_affinity"])
        rows["pinned 32 MiB host alloc (first CUDA call: CUDA init)"].append(
            step_seconds["pinned_host_alloc_with_cuda_init"] - step_seconds["bind_cpu_and_dsa"])
        rows["32 MiB GPU alloc + stream"].append(
            step_seconds["gpu_alloc_and_stream"] - step_seconds["pinned_host_alloc_with_cuda_init"])
        rows["wait for first block size"].append(measures[0]["start"] - step_seconds["gpu_alloc_and_stream"])
        sums = {}
        for step in MEASURE_STEPS:
            sums[step] = 0.0
        between = 0.0
        for index in range(len(measures)):
            previous = measures[index]["start"]
            for step in MEASURE_STEPS:
                sums[step] += measures[index][step] - previous
                previous = measures[index][step]
            if index > 0:
                between += measures[index]["start"] - measures[index - 1]["slice_copier_close"]
        rows["build CUDA batch operations (sum)"].append(sums["build_cuda_batch_operations"])
        rows["CUDA batch copies (sum)"].append(sums["cuda_batch_copies"])
        rows["SliceCopier register (sum)"].append(sums["slice_copier_register"])
        rows["DSA copies (sum)"].append(sums["dsa_copies"])
        rows["SliceCopier close (sum)"].append(sums["slice_copier_close"])
        rows["  of which start_together waits (sum)"].append(wait_total)
        rows["wait for next block size (sum)"].append(between)
        rows["last measure -> worker done"].append(step_seconds["worker_done"] - measures[-1]["slice_copier_close"])
        rows["worker total (fork -> done)"].append(step_seconds["worker_done"] - main_steps["fork_dsa_workers"])

    print(f"DSA workers: {len(ranks)}, measurements per worker: {measure_counts}")
    header = "| step (ms) |"
    separator = "|---|"
    for who in ranks:
        header += f" {who} |"
        separator += "---|"
    header += " max |"
    separator += "---|"
    print(header)
    print(separator)
    for name in row_order:
        line = f"| {name} |"
        for value in rows[name]:
            line += f" {value * 1000:.0f} |"
        line += f" {max(rows[name]) * 1000:.0f} |"
        print(line)

    print()
    print("main process (ms from script start):")
    for name in main_steps:
        print(f"  {name}: {main_steps[name] * 1000:.0f}")
    for seconds, label in events.get("main", []):
        if label.startswith("result "):
            print(f"  {seconds * 1000:.0f}: {label}")
    print("QAT worker (ms from script start):")
    for seconds, label in events.get("qat", []):
        print(f"  {seconds * 1000:.0f}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
