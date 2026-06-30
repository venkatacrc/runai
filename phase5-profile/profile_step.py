"""Phase 5 — standalone DDP training step instrumented with torch.profiler.

Profiles a small but representative transformer-block training loop on N GPUs:
  - forward + backward + optimizer + all_reduce
  - bf16, sdpa attention, AdamW
  - warmup steps (untraced) then a small window of traced steps

Why standalone rather than wrapping Phase 3:
  - Loads in <5 s vs. ~60 s for Gemma 4 12B → fast iteration.
  - Exercises the same kernel families (matmul + softmax + layer-norm + NCCL)
    that dominate the real workloads.
  - Output trace is small (<50 MB) so it opens instantly in chrome://tracing.

Run with:
    torchrun --standalone --nproc_per_node=4 profile_step.py

Outputs to ./trace_torch/ :
    trace.json                   open in chrome://tracing
    events.out.tfevents.<...>    open in `tensorboard --logdir trace_torch`
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class TinyBlock(nn.Module):
    """One transformer-style block — matmul-heavy on purpose."""

    def __init__(self, d: int, heads: int = 16) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=False),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=False),
        )
        self.heads = heads
        self.head_dim = d // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, n_layers: int, d: int, vocab: int, ctx: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        self.blocks = nn.ModuleList(TinyBlock(d) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.embed(idx) + self.pos[:, :t]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def main() -> None:
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)

    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")

    n_layers = env_int("N_LAYERS", 12)
    d_model = env_int("D_MODEL", 1024)
    vocab = env_int("VOCAB", 32000)
    ctx = env_int("CTX", 1024)
    bs = env_int("BS", 4)
    warmup = env_int("WARMUP_STEPS", 5)
    active = env_int("ACTIVE_STEPS", 5)
    out_dir = os.environ.get("OUT_DIR", "./trace_torch")

    if rank == 0:
        print(f"world={world}  layers={n_layers}  d={d_model}  ctx={ctx}  bs={bs}", flush=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42 + rank)
    model = TinyTransformer(n_layers, d_model, vocab, ctx).cuda().to(torch.bfloat16)
    ddp = DDP(model, device_ids=[local_rank]) if world > 1 else model
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4, fused=True)

    # Synthetic data — fits in HBM, identical each step.
    x = torch.randint(0, vocab, (bs, ctx), device="cuda")
    y = torch.randint(0, vocab, (bs, ctx), device="cuda")

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = nn.functional.cross_entropy(
            logits.float().view(-1, vocab),
            y.view(-1),
        )
        loss.backward()
        opt.step()
        return loss

    # untraced warmup so JIT / autotuning settle before tracing
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()

    if rank == 0:
        print(f"warmed up; tracing {active} steps -> {out_dir}", flush=True)

    prof_schedule = schedule(wait=0, warmup=1, active=active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(out_dir),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(active + 1):
            step()
            prof.step()

    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()

    # walltime baseline (untraced) for a sanity comparison after optimization
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if rank == 0:
        tokens = bs * ctx * world * n
        print(
            f"baseline: {n} steps in {dt:.2f}s  =  {n / dt:.2f} step/s  "
            f"= {tokens / dt / 1e6:.1f} M tok/s aggregate",
            flush=True,
        )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
