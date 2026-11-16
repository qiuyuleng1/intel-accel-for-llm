# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-rank CPU / Intel accelerator binding, shared by kvshrink_connector (local
KVStore) and the remote_pool daemon (KVStore on the daemon node)."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def bind_cpu_affinity(rank: int, tp_size: int, spec: Optional[str]) -> None:
    """`spec` is VLLM_CPU_OMP_THREADS_BIND: `|`-separated per-rank CPU lists ("0-7,16")."""
    if not spec or spec in ("all", "auto"):
        raise ValueError("VLLM_CPU_OMP_THREADS_BIND must assign CPUs to each worker")

    worker_cpu_specs = spec.split("|")
    if len(worker_cpu_specs) < tp_size:
        raise ValueError(
            f"VLLM_CPU_OMP_THREADS_BIND has {len(worker_cpu_specs)} entries, "
            f"but tensor parallel size is {tp_size}"
        )

    cpu_ids: set[int] = set()
    for part in worker_cpu_specs[rank].split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", maxsplit=1))
            if start > end:
                raise ValueError(f"Invalid CPU range: {part}")
            cpu_ids.update(range(start, end + 1))
        else:
            cpu_ids.add(int(part))

    if not cpu_ids:
        raise ValueError(f"No CPUs configured for rank {rank}")
    os.sched_setaffinity(0, cpu_ids)
    logger.info("Bound rank %d to CPUs %s", rank, sorted(cpu_ids))


def bind_intel_accel(rank: int) -> None:
    """KVSHRINK_QAT_DEVICES / KVSHRINK_DSA_DEVICES / KVSHRINK_IAA_DEVICES (`|` per rank)
    -> IAXL_QAT_DEVICES / IAXL_DSA_WQS / IAXL_IAA_DEVICES (IAA: NUMA node indices)."""
    for source, target in (
        ("KVSHRINK_QAT_DEVICES", "IAXL_QAT_DEVICES"),
        ("KVSHRINK_DSA_DEVICES", "IAXL_DSA_WQS"),
        ("KVSHRINK_IAA_DEVICES", "IAXL_IAA_DEVICES"),
    ):
        spec = os.getenv(source)
        if not spec:
            continue
        devices = spec.split("|")
        if len(devices) <= rank:
            raise ValueError(f"{source} has {len(devices)} entries, but rank is {rank}")
        os.environ[target] = devices[rank]
        logger.info("Bound rank %d: %s=%s", rank, target, devices[rank])
