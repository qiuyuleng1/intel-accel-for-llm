# Minimal NCCL check: all-reduce of one tensor across the visible GPUs (torchrun).
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
tensor = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(tensor)
print(f"rank {rank}: all_reduce ok, sum {tensor.item()}")
dist.destroy_process_group()
