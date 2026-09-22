from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist



@dataclass
class DistContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cpu"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup_distributed(backend: str | None = None) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DistContext(0, 0, 1, device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return DistContext(rank, local_rank, world_size, device)


def cleanup_distributed(ctx: DistContext):
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value, ctx: DistContext):
    if not ctx.enabled:
        return float(value)
    tensor = torch.as_tensor(float(value), device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / ctx.world_size).item()


def all_gather_tensor(tensor, ctx: DistContext):
    if not ctx.enabled:
        return tensor
    buffers = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
    dist.all_gather(buffers, tensor.contiguous())
    return torch.cat(buffers, dim=0)
