# SPDX-License-Identifier: Apache-2.0
"""Sample per-thread CPU and run-queue time of the vLLM TP workers from /proc.

Usage (on the HOST, while a benchmark runs; stop it with Ctrl-C or SIGTERM):
  python3 tests/bench_dev/proc-thread-sampler.py <output.csv> [interval_s]

Every interval (default 1 s) it finds the processes whose command line starts
with "VLLM::Worker_TP" and writes one CSV row per thread:
  epoch, rank, pid, tid, comm, cpu_ticks (utime + stime from
  /proc/<pid>/task/<tid>/stat), run_ns and runq_ns (1st and 2nd field of its
  schedstat: time on a CPU and time runnable but waiting for one), and
  cpus_allowed (Cpus_allowed_list; the KVShrink connector re-binds the
  worker after it starts, so it is read on every sample).
Reading /proc does not stop the sampled processes.
"""

import csv
import os
import signal
import sys
import time
from pathlib import Path

WORKER_PREFIX = "VLLM::Worker_TP"
# The workers use CPUs 0-127; 255 keeps the sampler off ranks 0-6's cores.
SAMPLER_CPU = 255

stop_requested = False


def request_stop(signum, frame) -> None:
    global stop_requested
    stop_requested = True


def find_workers() -> dict[int, int]:
    """Worker pid -> TP rank."""
    workers = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = Path(f"/proc/{entry}/cmdline").read_bytes()
        except OSError:
            continue
        name = cmdline.split(b"\0")[0].decode(errors="replace")
        if name.startswith(WORKER_PREFIX):
            workers[int(entry)] = int(name[len(WORKER_PREFIX):])
    return workers


def read_thread(pid: int, tid: int) -> tuple[str, int, int, int]:
    """(comm, cpu_ticks, run_ns, runq_ns) of one thread."""
    stat = Path(f"/proc/{pid}/task/{tid}/stat").read_text()
    # comm may contain spaces; the fields after it start past the last ')'.
    comm = stat[stat.index("(") + 1:stat.rindex(")")]
    fields = stat[stat.rindex(")") + 2:].split()
    # utime and stime are fields 14 and 15 of stat, i.e. 11 and 12 here.
    cpu_ticks = int(fields[11]) + int(fields[12])
    schedstat = Path(f"/proc/{pid}/task/{tid}/schedstat").read_text().split()
    return comm, cpu_ticks, int(schedstat[0]), int(schedstat[1])


def read_cpus_allowed(pid: int, tid: int) -> str:
    for line in Path(f"/proc/{pid}/task/{tid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def main() -> None:
    output_path = Path(sys.argv[1])
    interval_s = 1.0
    if len(sys.argv) > 2:
        interval_s = float(sys.argv[2])
    os.sched_setaffinity(0, {SAMPLER_CPU})
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "rank", "pid", "tid", "comm", "cpu_ticks",
                         "run_ns", "runq_ns", "cpus_allowed"])
        while not stop_requested:
            sample_start = time.time()
            for pid, rank in find_workers().items():
                try:
                    tids = os.listdir(f"/proc/{pid}/task")
                except OSError:
                    continue
                for tid_text in tids:
                    tid = int(tid_text)
                    try:
                        comm, cpu_ticks, run_ns, runq_ns = read_thread(pid, tid)
                        cpus_allowed = read_cpus_allowed(pid, tid)
                    except (OSError, ValueError, IndexError):
                        continue
                    writer.writerow([f"{sample_start:.3f}", rank, pid, tid, comm,
                                     cpu_ticks, run_ns, runq_ns, cpus_allowed])
            output_file.flush()
            time.sleep(max(0.0, interval_s - (time.time() - sample_start)))


if __name__ == "__main__":
    main()
