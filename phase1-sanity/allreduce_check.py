"""NCCL all-reduce micro-benchmark.

Launched by torchrun, one process per GPU. Reports busBW per message size,
which on an NVLink-connected node should saturate well into the hundreds of GB/s
(GB200 NVL: ~900 GB/s NVLink5 per GPU).
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if rank == 0:
        name = torch.cuda.get_device_name(local_rank)
        cc = torch.cuda.get_device_capability(local_rank)
        print(f"world_size={world}  device={name}  sm={cc[0]}{cc[1]}", flush=True)

    warm = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dist.all_reduce(warm)
    torch.cuda.synchronize()
    dist.barrier()

    for mb in (16, 64, 256, 1024, 4096):
        n = mb * 1024 * 1024 // 4
        x = torch.randn(n, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        # NCCL all-reduce bus bandwidth formula: 2*(N-1)/N * size / time
        bus_bw = (2 * (world - 1) / world) * (mb * 1e6) / dt / 1e9
        if rank == 0:
            print(
                f"size={mb:>5} MB  time={dt * 1000:7.2f} ms  busBW={bus_bw:7.1f} GB/s",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
