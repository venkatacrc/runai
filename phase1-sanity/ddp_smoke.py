"""End-to-end DDP smoke test.

Trains a tiny MLP on synthetic data for a few seconds, just to prove the full
DDP stack (init -> forward -> backward -> all-reduce of grads -> optimizer step)
works on this cluster.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    torch.manual_seed(42 + rank)
    dim = 4096
    model = nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.GELU(),
        nn.Linear(4 * dim, 4 * dim),
        nn.GELU(),
        nn.Linear(4 * dim, dim),
    ).cuda()
    ddp = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)

    batch = 64
    steps = 50
    x = torch.randn(batch, dim, device="cuda")
    y = torch.randn(batch, dim, device="cuda")

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (ddp(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(
            f"DDP ok.  world={world}  params={params / 1e6:.1f}M  "
            f"{steps} steps in {dt:.2f}s ({steps / dt:.1f} step/s)  "
            f"final_loss={loss.item():.4f}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
