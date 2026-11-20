# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 2: force the H2D / D2H crossovers, run KVStore PUT (D2H) then GET (H2D) with
benchmark/kvstore/kvstore_benchmark.py's Benchmark, and compare every tested block.

0 sends every copy of that direction to CUDA batch, 9223372036854775807 (INT64_MAX) to DSA.
Needs IAXL_DSA_AUTO_FALLBACK_ENABLE=0 so KVStore does not calibrate over the forced values."""

import argparse
import os
import sys

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_DIR, "benchmark", "kvstore"))

import kvstore_benchmark  # noqa: E402

from iaxl import torch_ext as iqt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2d-crossover", type=int, required=True)
    parser.add_argument("--d2h-crossover", type=int, required=True)
    forced, benchmark_argv = parser.parse_known_args()
    benchmark_args = kvstore_benchmark.parse_args(benchmark_argv)
    if os.environ.get("IAXL_DSA_AUTO_FALLBACK_ENABLE") != "0":
        raise SystemExit("set IAXL_DSA_AUTO_FALLBACK_ENABLE=0, else KVStore calibration replaces the forced values")

    iqt.dsa_set_crossover(forced.h2d_crossover, forced.d2h_crossover)
    print(f"FORCED h2d_bytes={forced.h2d_crossover} d2h_bytes={forced.d2h_crossover}", flush=True)

    bench = kvstore_benchmark.Benchmark(benchmark_args)
    bench.print_config()
    bench.setup()
    tested_blocks = torch.tensor(bench.block_indices, device="cuda")
    expected = {}
    for name, tensor in bench.kv_caches.items():
        expected[name] = tensor.index_select(kvstore_benchmark.BLOCK_DIM, tested_blocks).clone()
    bench.create_store()
    bench.warmup()
    bench.put()
    bench.get()
    torch.cuda.synchronize()

    mismatched_layers = []
    for name, tensor in bench.kv_caches.items():
        loaded = tensor.index_select(kvstore_benchmark.BLOCK_DIM, tested_blocks)
        if not torch.equal(loaded, expected[name]):
            mismatched_layers.append(name)
    print(
        f"COMPARE tested_blocks_per_layer={len(bench.block_indices)} layers={len(expected)} "
        f"mismatched_layers={len(mismatched_layers)}"
    )
    if mismatched_layers:
        print(f"RESULT FAIL mismatched layers: {mismatched_layers}")
        return 1
    print("RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
