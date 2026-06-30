# Phase 1 — Portal-only path (when CLI submit is blocked by admin policy)

Many Run:AI deployments lock down CLI `workspace submit` / `training submit`
via an admission policy on annotations (the error reads
`Field: annotations, the administrator prohibited adding new items`). In that
case, drive Phase 1 from the Run:AI **portal UI** instead. The original lab
used a project named `ri-fm` — substitute your own project name throughout.

> Once admin allows CLI submission, `phase1-sanity/submit.sh` is the same flow
> as a one-liner.

---

## 1. Create the workspace in the portal

1. Open your Run:AI portal → **Workloads** → **+ NEW WORKLOAD** → **Workspace**.
2. **Project**: your project that owns the GB200 quota (the original lab used `ri-fm`).
3. **Workspace name**: `p1-sanity-8gpu`
4. **Cluster / scope**: pick the GB200 cluster.
5. **Environment (image)**:
   - **Image URL**: `nvcr.io/nvidia/pytorch:25.04-py3`
   - If your portal forces you to pick a pre-registered Environment asset, choose one whose image is **PyTorch 24.10+ / 25.x** (NGC). **Do not pick the Jupyter image** — that's why `testonegpu` had no `torch` module.
   - Make sure a tool/connection labeled **Terminal**, **Shell**, or **JupyterLab** is enabled. We need terminal access.
6. **Compute resource**:
   - **GPU devices**: `8`
   - **Large /dev/shm**: ON (the toggle is sometimes labeled "Increase shared memory size")
   - CPU / RAM: leave defaults (the GB200 node has plenty)
7. **Data sources** *(important if you have a PVC)*:
   - Click **+ Add data source** → choose your PVC.
   - Mount path: `/work`
   - This makes Phase 2's nanochat downloads survive pod restarts.
8. **Scheduling / Node pool**: pick the GB200 pool if your project has more than one. Otherwise leave default.
9. Leave **Annotations / Labels** untouched. Don't add any (that's what triggered the CLI policy block).
10. Click **CREATE WORKSPACE**.

When it reaches **Running**, click **Connect** → **JupyterLab** (or **Terminal**, whichever your asset exposes). You want a shell, not a notebook cell, for `torchrun`.

---

## 2. Verify the environment (30 seconds)

In the workspace terminal:

```bash
hostname
nvidia-smi -L                       # expect 8 lines, each "NVIDIA GB200"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())'
```

You should see torch 2.6+ and CUDA 12.6+ (NGC `pytorch:25.04-py3`). If you see
`ModuleNotFoundError: No module named 'torch'`, the workspace was started with
a non-PyTorch image — stop the workspace, edit it, change the image to
`nvcr.io/nvidia/pytorch:25.04-py3`, and restart.

---

## 3. Run the NCCL all-reduce + DDP smoke tests

Paste the **entire block below** into the workspace terminal. It writes both
test scripts to disk and runs them back-to-back via `torchrun`. Total time
~30–60 seconds.

```bash
mkdir -p /tmp/p1 && cd /tmp/p1

cat > allreduce_check.py <<'PY'
"""NCCL all-reduce micro-benchmark. One process per GPU, busBW per message size."""
import os, time
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
    torch.cuda.synchronize(); dist.barrier()

    for mb in (16, 64, 256, 1024, 4096):
        n = mb * 1024 * 1024 // 4
        x = torch.randn(n, device="cuda")
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        bus_bw = (2 * (world - 1) / world) * (mb * 1e6) / dt / 1e9
        if rank == 0:
            print(f"size={mb:>5} MB  time={dt*1000:7.2f} ms  busBW={bus_bw:7.1f} GB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

cat > ddp_smoke.py <<'PY'
"""End-to-end DDP smoke test on synthetic data."""
import os, time
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
        nn.Linear(dim, 4 * dim), nn.GELU(),
        nn.Linear(4 * dim, 4 * dim), nn.GELU(),
        nn.Linear(4 * dim, dim),
    ).cuda()
    ddp = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)

    batch, steps = 64, 50
    x = torch.randn(batch, dim, device="cuda")
    y = torch.randn(batch, dim, device="cuda")

    torch.cuda.synchronize(); dist.barrier()
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
            f"DDP ok.  world={world}  params={params/1e6:.1f}M  "
            f"{steps} steps in {dt:.2f}s ({steps/dt:.1f} step/s)  final_loss={loss.item():.4f}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

echo "=== Hostname / GPUs ==="
hostname
nvidia-smi -L

echo "=== Python / Torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())'

echo "=== NCCL all-reduce (8 GPUs) ==="
torchrun --standalone --nproc_per_node=8 allreduce_check.py

echo "=== DDP smoke (8 GPUs) ==="
torchrun --standalone --nproc_per_node=8 ddp_smoke.py

echo "=== DONE ==="
```

---

## 4. What "good" looks like

### `nvidia-smi -L`

8 lines, each like `GPU N: NVIDIA GB200 (UUID: ...)`.

### Torch / NCCL line

```
torch 2.6.x  cuda 12.6  nccl (2, 22, x)
```

### NCCL all-reduce

On NVL-style NVLink5 you should see busBW climb with message size and stabilize
in the **500–800 GB/s** range for ≥1 GB messages:

```
size=   16 MB  time=   ~0.x ms   busBW= ~200 GB/s
size=   64 MB  time=   ~0.x ms   busBW= ~400 GB/s
size=  256 MB  time=   ~0.x ms   busBW= ~550 GB/s
size= 1024 MB  time=   ~1.x ms   busBW= ~650 GB/s
size= 4096 MB  time=   ~6.x ms   busBW= ~720 GB/s
```

If big messages cap below ~50 GB/s, NCCL fell back to PCIe — flag that, NVLink
isn't being used. (`nvidia-smi topo -m` should show `NV*` between every pair.)

### DDP smoke

```
DDP ok.  world=8  params=234.9M  50 steps in ~1-2 s  (~25-50 step/s)  final_loss=0.xxxx
```

`final_loss` should be much lower than the implicit baseline — targets are
random, the model just memorizes them, which proves the full forward + backward
+ gradient all-reduce + optimizer step loop works.

---

## 5. Capture output, then stop the workspace

Copy the terminal output into `phase1-sanity/run-output.txt` (handy for later
comparison). Then:

1. Portal → Workloads → `p1-sanity-8gpu` → **Stop** (frees the 8 GPUs).
2. Or keep it running and reuse it for Phase 2 — but Phase 2 also wants 8 GPUs
   for hours, so just stopping and starting a fresh workspace is cleaner.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi -L` shows only 1 GPU | Workspace requested 1, not 8 | Stop the workspace, edit it, set GPUs to 8, restart. |
| `ModuleNotFoundError: torch` | Workspace started with a non-PyTorch image (e.g. Jupyter base) | Change image to `nvcr.io/nvidia/pytorch:25.04-py3`. |
| `torchrun: command not found` | Very old torch image | Use the recommended NGC image, or `pip install torch`. |
| `NCCL WARN ... bootstrap` and hang | NIC selection issue | `export NCCL_SOCKET_IFNAME=^lo,docker0` before `torchrun`. |
| `RuntimeError: CUDA error: no kernel image is available for execution on the device` | Image too old for Blackwell sm_100 | Use NGC `pytorch:25.04-py3` (or 24.10+). |
| `busBW` ≤ 30 GB/s on big messages | NVLink not engaged | Run `nvidia-smi topo -m`; expect `NV*` between GPUs. If `PHB` / `SYS`, this isn't an NVLinked tray. |

When NCCL and DDP both report healthy numbers, you're done with Phase 1.
Next: [`phase2-nanochat/notes.md`](../phase2-nanochat/notes.md) — same portal
mechanics, just a longer-running job.
