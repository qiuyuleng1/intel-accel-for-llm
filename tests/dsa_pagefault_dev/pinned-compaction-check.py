#!/usr/bin/env python3
# Does the kernel try (and fail) to move pinned host memory when it compacts memory?
#
# Allocates BUFFER_GB of pinned host memory the way the chosen mode does, asks the
# kernel to compact memory once, and prints how many page migrations failed and
# whether the sampled pages of the buffer changed physical address.
#   none                 no buffer (background level of migration failures)
#   torch_pinned         torch.empty(pin_memory=True), as ScratchPool does today
#   torch_host_register  the same call with PYTORCH_CUDA_ALLOC_CONF=
#                        pinned_use_cuda_host_register:True (set by the caller)
#
# Runs as root inside the privileged container (needs /proc/self/pagemap PFNs and
# /proc/sys/vm/compact_memory), one mode per process:
#   python3 tests/dsa_pagefault_dev/pinned-compaction-check.py torch_pinned
import os
import struct
import sys
import time

import torch

BUFFER_GB = 8
PAGE_SIZE = 4096
SAMPLE_EVERY_PAGES = 256
PFN_MASK = (1 << 55) - 1
PAGE_PRESENT_BIT = 1 << 63


def read_vmstat(keys):
    values = {}
    with open("/proc/vmstat") as vmstat:
        for line in vmstat:
            name, value = line.split()
            if name in keys:
                values[name] = int(value)
    return values


def mapping_of(address):
    """The /proc/self/smaps header line and the fields that tell the memory type."""
    with open("/proc/self/smaps") as smaps:
        lines = smaps.read().split("\n")
    for index in range(len(lines)):
        fields = lines[index].split()
        if not fields or "-" not in fields[0] or len(fields) < 5:
            continue
        low, high = fields[0].split("-")
        if int(low, 16) <= address < int(high, 16):
            details = [lines[index]]
            for detail in lines[index + 1:index + 30]:
                if detail.split(":")[0] in ("Anonymous", "Locked"):
                    details.append(detail.strip())
            return " | ".join(details)
    return "not found"


def sample_pfns(start, size):
    """Physical page number of every SAMPLE_EVERY_PAGES-th page (None if not present)."""
    pfns = []
    with open("/proc/self/pagemap", "rb") as pagemap:
        for offset in range(0, size, PAGE_SIZE * SAMPLE_EVERY_PAGES):
            pagemap.seek((start + offset) // PAGE_SIZE * 8)
            (entry,) = struct.unpack("<Q", pagemap.read(8))
            if entry & PAGE_PRESENT_BIT:
                pfns.append(entry & PFN_MASK)
            else:
                pfns.append(None)
    return pfns


def compact_memory():
    with open("/proc/sys/vm/compact_memory", "w") as control:
        control.write("1")


def main():
    mode = sys.argv[1]
    if mode not in ("none", "torch_pinned", "torch_host_register"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "torch_host_register" and "pinned_use_cuda_host_register:True" not in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ):
        raise ValueError("torch_host_register needs PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True")

    buffer = None
    size = BUFFER_GB << 30
    pfns_before = []
    if mode != "none":
        allocate_start = time.perf_counter()
        buffer = torch.empty((size,), dtype=torch.uint8, device="cpu", pin_memory=True)
        print(f"{mode}: allocated {BUFFER_GB} GiB pinned={buffer.is_pinned()} "
              f"in {time.perf_counter() - allocate_start:.2f}s")
        print(f"{mode}: mapping {mapping_of(buffer.data_ptr())}")
        pfns_before = sample_pfns(buffer.data_ptr(), size)

    keys = ("pgmigrate_success", "pgmigrate_fail", "compact_isolated", "compact_fail")
    vmstat_before = read_vmstat(keys)
    compact_start = time.perf_counter()
    compact_memory()
    compact_seconds = time.perf_counter() - compact_start
    vmstat_after = read_vmstat(keys)

    deltas = []
    for key in keys:
        deltas.append(f"{key}=+{vmstat_after[key] - vmstat_before[key]}")
    print(f"{mode}: one compaction took {compact_seconds:.2f}s, machine-wide " + " ".join(deltas))

    if buffer is not None:
        pfns_after = sample_pfns(buffer.data_ptr(), size)
        moved = 0
        not_present = 0
        for index in range(len(pfns_before)):
            if pfns_before[index] is None or pfns_after[index] is None:
                not_present += 1
            elif pfns_before[index] != pfns_after[index]:
                moved += 1
        print(f"{mode}: sampled {len(pfns_before)} pages of the buffer, physical address changed "
              f"for {moved}, not present for {not_present}")


if __name__ == "__main__":
    main()
