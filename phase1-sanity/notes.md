# Phase 1 — Distributed Training Sanity Check

What we're proving, in order:

1. **You can launch a multi-GPU pod** through Run:AI (8 GPUs in one node).
2. **NCCL works** between those 8 GPUs (the all-reduce micro-bench).
3. **Full PyTorch DDP stack works** end-to-end (forward → backward → all-reduce of grads → optimizer step).

## How to run

**If CLI submit is allowed on your project:**

```bash
source scripts/env.sh
bash phase1-sanity/submit.sh
```

Then in another terminal:

```bash
runai workload list
runai workload logs p1-sanity-... -f
```

**If CLI submit is blocked** (admission policy on annotations — common in
production clusters), follow [`portal-guide.md`](portal-guide.md) instead. It
runs the same two test scripts via a portal-launched Workspace.

## What "good" looks like

### `nvidia-smi -L`

You should see 8 lines, each saying something like:

```
GPU 0: NVIDIA GB200 (UUID: ...)
...
GPU 7: NVIDIA GB200 (UUID: ...)
```

### Torch sanity line

```
torch 2.4.x  cuda 12.6  nccl (2, 22, x)
```

(Exact versions depend on the NGC image.) On `sm_100` (Blackwell) you need torch ≥ 2.4 and CUDA ≥ 12.4 or PyTorch will fall back to slow kernels.

### NCCL all-reduce

On an NVL72-style backplane (NVLink5 at ~900 GB/s/GPU) you should see busBW
ramp up with message size and stabilize around **500–800 GB/s** for 1 GB+
messages. Anything below ~50 GB/s means NCCL fell back to PCIe — flag it.

```
size=   16 MB  time=  0.x ms  busBW= ~200 GB/s
size= 1024 MB  time=  ~2 ms   busBW= ~600 GB/s
size= 4096 MB  time=  ~8 ms   busBW= ~700 GB/s
```

### DDP smoke

```
DDP ok.  world=8  params=234.9M  50 steps in 1.x s (~30 step/s)  final_loss=0.x
```

Final loss should *decrease* relative to the initial random output (you'll
see it converging quickly because the targets are random — the model just
memorizes them).

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Pod `Pending` forever | wrong node pool, or quota exhausted | `runai workload describe $NAME` — read the Events |
| `RuntimeError: NCCL error` on init | NCCL can't pick a NIC | Add `--environment NCCL_SOCKET_IFNAME=eth0` (or `^docker0,lo`) |
| `busBW` ~10 GB/s for big messages | NVLink not enabled / fell back to PCIe | `nvidia-smi topo -m` inside the pod — should show `NV*` between GPUs |
| `kernel image not found` / `CUDA error: no kernel` | image too old for Blackwell | Use NGC `pytorch:25.04-py3` or newer |
| `torchrun` hangs at "Started worker..." | rendezvous backend confused | We use `--standalone`, which is fine on one node. If still hangs, set `MASTER_ADDR=127.0.0.1 MASTER_PORT=29500` and use `--rdzv-endpoint`. |

When all three checks are green, move on to [Phase 2](../phase2-nanochat/notes.md).
