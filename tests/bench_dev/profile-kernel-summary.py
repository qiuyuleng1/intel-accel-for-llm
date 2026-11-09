#!/usr/bin/env python3
"""GPU kernel time per decode step from vLLM torch-profiler traces.

Usage: profile-kernel-summary.py <profile-decode results dir> [settings...]
Reads <dir>/<setting>/torch-profile/rank*.pt.trace.json.gz (written by
profile-decode-qwen3-32b.sh) and, per setting and rank, sums the GPU time of
every kernel by kind. Steps = attention kernel calls / 64 layers.
"""
import glob
import gzip
import json
import os
import statistics
import sys

NUM_LAYERS = 64


def kernel_kind(name):
    if "flash_fwd_splitkv_combine" in name:
        return "attention-combine"
    if "flash_fwd" in name:
        return "attention"
    if "AllReduce" in name:
        return "allreduce"
    if "AllGather" in name:
        return "allgather"
    if "cutlass" in name or "cublas" in name or "gemm" in name.lower():
        return "gemm"
    return "other"


def read_kernel_durations(trace_path):
    """Returns {kind: [duration_us, ...]} of the GPU kernels in one trace."""
    with gzip.open(trace_path, "rt") as trace_file:
        trace = json.load(trace_file)
    durations_by_kind = {}
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        kind = kernel_kind(event["name"])
        durations_by_kind.setdefault(kind, []).append(event["dur"])
    return durations_by_kind


def main():
    results_dir = sys.argv[1]
    settings = sys.argv[2:]
    if not settings:
        settings = ["default", "gpu-prefix"]
    kinds = ["attention", "attention-combine", "allreduce", "allgather", "gemm", "other"]
    for setting in settings:
        trace_paths = sorted(glob.glob(os.path.join(results_dir, setting, "torch-profile", "rank*.pt.trace.json.gz")))
        if not trace_paths:
            raise ValueError(f"no traces under {results_dir}/{setting}/torch-profile")
        print(f"== {setting}")
        print(f"{'rank':>4} {'steps':>5} {'kind':<17} {'ms/step':>8} {'calls':>6} {'call mean us':>12} {'call median us':>14}")
        for trace_path in trace_paths:
            rank = os.path.basename(trace_path).split(".")[0]
            durations_by_kind = read_kernel_durations(trace_path)
            attention_calls = len(durations_by_kind.get("attention", []))
            if attention_calls % NUM_LAYERS != 0:
                raise ValueError(f"{trace_path}: {attention_calls} attention calls is not a multiple of {NUM_LAYERS}")
            steps = attention_calls // NUM_LAYERS
            total_ms_per_step = 0.0
            for kind in kinds:
                durations = durations_by_kind.get(kind, [])
                if not durations:
                    continue
                ms_per_step = sum(durations) / 1000 / steps
                total_ms_per_step += ms_per_step
                print(f"{rank:>4} {steps:>5} {kind:<17} {ms_per_step:>8.2f} {len(durations):>6} "
                      f"{statistics.mean(durations):>12.1f} {statistics.median(durations):>14.1f}")
            print(f"{rank:>4} {steps:>5} {'all kernels':<17} {total_ms_per_step:>8.2f}")


if __name__ == "__main__":
    main()
