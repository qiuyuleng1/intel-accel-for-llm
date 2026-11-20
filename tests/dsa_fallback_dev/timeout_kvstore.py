# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Step 3c: a KVStore with rank=1, tp_size=2 whose TP group first rank (rank 0) never starts
must raise RuntimeError after IAXL_API_TIMEOUT seconds instead of waiting forever."""

import time

import torch

from iaxl.envs import envs
from iaxl.kvstore import KVStore


def main() -> int:
    kv_caches = {"0": torch.zeros((2, 8, 16, 4, 128), dtype=torch.bfloat16, device="cuda")}
    start = time.monotonic()
    try:
        KVStore(
            model_name="dsa_timeout_test",
            kv_caches=kv_caches,
            block_dim=1,
            rank=1,
            tp_size=2,
        )
    except RuntimeError as error:
        print(
            f"RAISED after {time.monotonic() - start:.1f}s "
            f"(IAXL_API_TIMEOUT={envs.IAXL_API_TIMEOUT}): {error}"
        )
        return 0
    print("RESULT FAIL: KVStore was created without rank 0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
