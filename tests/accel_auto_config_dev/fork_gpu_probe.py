"""Checks the order tools/accel_auto_config.py relies on: torch is loaded in the parent, and each forked child
sets CUDA_VISIBLE_DEVICES before its first CUDA call. Prints "GPU <index> <uuid> visible <count>" per child.
Usage: fork_gpu_probe.py <gpu index> [<gpu index> ...]"""

import multiprocessing
import os
import sys

import torch


def child(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU {gpu} {properties.uuid} visible {torch.cuda.device_count()}", flush=True)


def main() -> None:
    context = multiprocessing.get_context("fork")
    processes = []
    for gpu in sys.argv[1:]:
        process = context.Process(target=child, args=(gpu,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
