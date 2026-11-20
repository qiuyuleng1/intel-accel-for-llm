# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 1: call dsa_calibrate_crossover several times on one GPU; print each result, its time
and the GPU free memory before the first call and after the last call."""

import argparse
import time

import torch

from iaxl import torch_ext as iqt

MIB = 1 << 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()

    print(f"ENABLED {iqt.dsa_auto_fallback_enabled()}", flush=True)
    torch.cuda.set_device(args.gpu)
    free_before, _ = torch.cuda.mem_get_info(args.gpu)
    for call in range(1, args.calls + 1):
        start = time.monotonic()
        h2d_bytes, d2h_bytes = iqt.dsa_calibrate_crossover(args.gpu)
        seconds = time.monotonic() - start
        print(
            f"CALL {call} h2d_bytes={h2d_bytes} d2h_bytes={d2h_bytes} seconds={seconds:.2f}",
            flush=True,
        )
    free_after, _ = torch.cuda.mem_get_info(args.gpu)
    print(
        f"FREE_MIB before={free_before // MIB} after={free_after // MIB} "
        f"used_by_calls={(free_before - free_after) // MIB}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
