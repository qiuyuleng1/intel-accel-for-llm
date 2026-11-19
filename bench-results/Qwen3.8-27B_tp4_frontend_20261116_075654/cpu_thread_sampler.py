"""Per-thread CPU usage of the vLLM processes, sampled every second.

Every INTERVAL_S seconds it scans /proc (the vLLM container runs with
--pid host, so the host sees its processes), labels each process by its
command line and writes one row per thread that used at least
MIN_THREAD_CPU_PCT of one core since the previous sample, plus one row per
process with the sum over all its threads (tid "total"). CPU % is of one core:
100 = one core busy for the whole second.

Labels (first match in the command line wins):
  rust_frontend  vllm-rs, the Rust API server
  engine_core    VLLM::EngineCore, the scheduler process
  worker         VLLM::Worker_TP*, one GPU each
  bench_client   vllm bench serve
  api_server     vllm serve: the Python API server; with the Rust frontend it
                 only starts and watches the other processes

Usage: python3 cpu_thread_sampler.py <output csv>   (runs until SIGTERM)
"""

import csv
import os
import signal
import sys
import time

INTERVAL_S = 1.0
MIN_THREAD_CPU_PCT = 1.0
CLOCK_TICKS_PER_S = os.sysconf("SC_CLK_TCK")
LABEL_PATTERNS = (
    ("vllm-rs", "rust_frontend"),
    ("EngineCore", "engine_core"),
    ("VLLM::Worker", "worker"),
    ("bench serve", "bench_client"),
    ("vllm serve", "api_server"),
    ("APIServer", "api_server"),
)


def label_for(cmdline):
    for pattern, label in LABEL_PATTERNS:
        if pattern in cmdline:
            return label
    return ""


def read_cmdline(pid):
    """Command line of the process, "" if it has exited."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as cmdline_file:
            return cmdline_file.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def read_thread_ticks(pid):
    """{tid: (thread name, user + system CPU time in clock ticks)} of the process."""
    thread_ticks = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return thread_ticks
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as stat_file:
                stat = stat_file.read()
        except OSError:
            continue
        # The name is in parentheses and may itself contain spaces or ')'.
        name_end = stat.rindex(")")
        name = stat[stat.index("(") + 1 : name_end]
        fields_after_name = stat[name_end + 2 :].split()
        # utime and stime are fields 14 and 15 of the stat line; fields_after_name
        # starts at field 3.
        utime = int(fields_after_name[11])
        stime = int(fields_after_name[12])
        thread_ticks[tid] = (name, utime + stime)
    return thread_ticks


def stop(signum, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, stop)
    previous_ticks = {}
    previous_time = time.monotonic()
    with open(sys.argv[1], "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["unix_time_s", "label", "pid", "tid", "thread_name", "cpu_pct"])
        while True:
            sample_start = time.monotonic()
            elapsed_s = sample_start - previous_time
            previous_time = sample_start
            unix_time_s = round(time.time(), 1)
            current_ticks = {}
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                label = label_for(read_cmdline(pid))
                if not label:
                    continue
                process_cpu_pct = 0.0
                for tid, (name, ticks) in read_thread_ticks(pid).items():
                    current_ticks[(pid, tid)] = ticks
                    # A thread first seen in this sample has no interval yet.
                    if (pid, tid) not in previous_ticks:
                        continue
                    used_ticks = ticks - previous_ticks[(pid, tid)]
                    thread_cpu_pct = used_ticks / CLOCK_TICKS_PER_S / elapsed_s * 100
                    process_cpu_pct += thread_cpu_pct
                    if thread_cpu_pct >= MIN_THREAD_CPU_PCT:
                        writer.writerow([unix_time_s, label, pid, tid, name, round(thread_cpu_pct, 1)])
                if process_cpu_pct >= MIN_THREAD_CPU_PCT:
                    writer.writerow([unix_time_s, label, pid, "total", "", round(process_cpu_pct, 1)])
            output_file.flush()
            previous_ticks = current_ticks
            time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - sample_start)))


if __name__ == "__main__":
    main()
