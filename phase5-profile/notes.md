# Phase 5 — Profile + optimize on GB200

You've got two working training pipelines on 4× GB200:

- **Phase 2c** — nanoGPT GPT-2 124M pretrain on OpenWebText, ~2.6 s/iter
- **Phase 3**  — Gemma 4 12B-it LoRA SFT on Alpaca,   ~4.0 s/step (effective batch 64, seq_len 1024)

Both have headroom. Phase 5 is about finding it.

We use three profilers, each with a different field of view:

| Profiler | Granularity | Best for | Output | Where you view it |
|---|---|---|---|---|
| **`torch.profiler`** | Python op + CUDA stream | "Which Python operator is the bottleneck?" / "How big are my tensors?" / quick wins | TensorBoard event files (`.json`) | TensorBoard in a workspace, or Chrome `chrome://tracing` |
| **`nsys` (Nsight Systems)** | OS + CPU + CUDA stream + NCCL + Python markers | "Where are the GPU gaps?" "Is NCCL all-reduce overlapping?" "Is the data loader stalling?" | `.nsys-rep` file | Nsight Systems GUI on your laptop |
| **`ncu` (Nsight Compute)** | Single CUDA kernel internals | "Why is this matmul slow?" "What's the SOL (speed-of-light) ceiling?" "Tensor-core utilization?" | `.ncu-rep` file | Nsight Compute GUI on your laptop |

Rule of thumb — top-down, in this order:

1. **`torch.profiler` first** (5 min). Quick scan to find the biggest Python-visible bottleneck.
2. **`nsys` next** (15 min). Confirm or refute what torch.profiler showed. Find GPU gaps, NCCL serialization, data-loader stalls — anything below the Python level.
3. **`ncu` last** (30 min, optional). Only after you've identified a *specific* kernel that dominates and you want to know if it can be made faster. Don't ncu-everything.

---

## Pre-flight: are the tools even installed?

In your workspace:

```bash
which nsys ncu nvidia-smi
nsys --version
ncu --version
python -c "from torch.profiler import profile, ProfilerActivity; print('torch.profiler OK')"
```

Expected:
- `nvidia-smi` → present (you confirmed `/usr/bin/nvidia-smi` earlier).
- `nsys` → present at `/usr/local/cuda/bin/nsys` or `/opt/nvidia/nsight-systems/.../bin/nsys`. Version should be **≥ 2025.4** for full Blackwell (sm_100) timeline accuracy.
- `ncu` → present at `/usr/local/cuda/bin/ncu` or `/opt/nvidia/nsight-compute/.../ncu`. Version **≥ 2025.x** for Blackwell metric sections.
- `torch.profiler` is a stdlib part of any modern PyTorch, no install.

If `nsys` or `ncu` are missing, see **"Install if missing"** below before anything else.

---

## Install if missing (workspace shell, may need sudo)

NVIDIA ships both as part of the CUDA toolkit and as standalone packages. Quickest paths, in order of preference:

### a) The image already has CUDA — find the binaries

```bash
find / -name nsys -type f 2>/dev/null | head
find / -name ncu  -type f 2>/dev/null | head
ls /opt/nvidia 2>/dev/null
ls /usr/local/cuda*/bin 2>/dev/null | grep -E 'ns[ys]|ncu'
```

If found, just put them on PATH:

```bash
export PATH=/opt/nvidia/nsight-systems/<version>/bin:/opt/nvidia/nsight-compute/<version>:$PATH
```

### b) Install from apt (sudo required)

```bash
sudo apt-get update
sudo apt-get install -y nsight-systems-cli nsight-compute
```

If sudo is denied, ask the cluster admin or fall back to (c).

### c) Last resort: Python-only profiling

`torch.profiler` works without nsys/ncu and covers ~70% of useful optimization questions on its own. Phase 5 sub-phase **5a** stands alone. Skip 5b/5c if you can't install nsys/ncu.

---

## Sub-phase 5a — `torch.profiler` on a real Phase 3 step

The simplest path: drop in `profile_phase3.py`, which is a 30-line wrapper around your existing `train.py` that enables `torch.profiler` for steps 5–9 (after warmup, before things get repetitive). Output is a `trace.json` you can open in `chrome://tracing`, plus TensorBoard event files.

See `phase5-profile/profile_phase3.py` and `portal-guide.md` §5a.

What to look for in the resulting trace:
- **Step duration breakdown**: sort ops by self CUDA time. Top 3 ops should be matmuls (Q/K/V/O projections + MLP gate/up/down). If something else is in the top 3 (e.g. `aten::copy_`, `aten::to`, `aten::layer_norm`), there's an unnecessary copy or dtype cast hiding.
- **GPU idle bars**: in the timeline view, you should see *almost no gaps* between consecutive forward → backward → opt steps. If gaps exist, the Python overhead is dominating.
- **Memory peak**: per-rank, should be < 35 GB for the LoRA-12B + grad-ckpt config. If it's hitting 80+ GB, you forgot to enable gradient checkpointing.
- **CPU vs GPU split**: if CPU time > 30% of step time, your data loader is single-threaded.

Quick wins this finds:
- Forgotten `.cpu()` / `.to(device)` round-trips inside the training loop
- A loss computation accidentally running in fp32 instead of bf16
- Activations being materialized when grad-ckpt was supposed to discard them

---

## Sub-phase 5b — `nsys` on a 4-GPU step

The same Phase 3 training, but launched under `nsys profile`. Captures everything: kernel launches, NCCL all-reduce timings, NVTX annotations, OS thread state.

See `phase5-profile/run_nsys.sh` and `portal-guide.md` §5b.

What to look for in the resulting `.nsys-rep` (opened in Nsight Systems GUI on your laptop):
- **The "CUDA HW" row per GPU**: should be ~95% solid for a healthy compute-bound training step. Stripes/gaps mean the GPU is waiting on something.
- **The "NCCL" row**: if all-reduce takes >15% of step time, your communication isn't hidden behind compute. Possible fixes: enable gradient bucket overlap, FSDP `BACKWARD_PRE`, or just larger batch.
- **The "OS Runtime" / "Python" rows**: if you see long `recv`/`epoll` waits during the step, dataloader is starving the GPU. Increase `dataloader_num_workers`, pin memory, or pre-tokenize.
- **Compare rank 0 vs rank 1/2/3 timelines**: they should be near-identical. Drift between ranks = unbalanced work or a slow GPU/node.

Quick wins this finds:
- NCCL not overlapping with backward → enable `gradient_accumulation_steps` properly, or move to FSDP with `BACKWARD_PRE`
- Dataloader stalls (often caused by `dataloader_num_workers=0`)
- One GPU consistently slower than others — usually a thermal/PCIe-fallback issue worth flagging to ops

---

## Sub-phase 5c — `ncu` on a single hot kernel

After 5a/5b you'll know *which* kernel dominates (e.g. "the fused SDPA forward in Gemma's attention is 40% of the step"). Now you want to know: can it be faster?

`ncu` runs that kernel many times under heavy instrumentation, reporting:
- Achieved vs peak FLOPs
- Achieved vs peak memory bandwidth
- Tensor Core utilization
- Memory access patterns (coalesced? aligned?)
- Stall reasons (long scoreboard, MIO throttle, etc.)

See `phase5-profile/run_ncu.sh` and `portal-guide.md` §5c.

What to look for in the resulting `.ncu-rep` (opened in Nsight Compute GUI on your laptop):
- **Speed Of Light (SOL)** section: shows "achieved vs theoretical peak" for compute and memory. If achieved compute < 30% SOL → compute kernel is wasting cycles (probably stalls). If achieved memory < 30% SOL → memory-bound. Different fixes for each.
- **Tensor Core utilization**: should be 70%+ for a matmul on bf16. If 0%, the op is going through CUDA cores (e.g. fp32 path); switch dtype or use `torch.set_float32_matmul_precision('high')`.
- **Source / SASS heatmap**: maps stalls back to the kernel's source line. Useful only if you wrote the kernel; for stock cuBLAS/cuDNN it's a black box.

This is the deep end. You only need it if:
- A specific kernel dominates step time (>30% of total).
- You're considering writing a custom CUDA/Triton kernel to replace it.
- Or you're picking between alternatives (e.g. SDPA vs FA3 vs flash-attn-v3).

---

## What to actually optimize after profiling

See `optimize.md` — that's the findings → fixes playbook. Phase 5 isn't done until you measure a *concrete speedup* on Phase 2c or Phase 3.

Target this lab: get **either** Phase 2c **or** Phase 3 to be 1.5–2× faster than the baseline you measured today (Phase 2c: 2.6 s/iter; Phase 3: 4.0 s/step). On GB200 that's very achievable.
