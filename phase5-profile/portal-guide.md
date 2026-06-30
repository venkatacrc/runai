# Phase 5 — Portal-launch guide for profiling

Same pattern as Phase 3: scripts live on the PVC at `/data/training/runai-repo/phase5-profile/`, a `run.sh` per sub-phase wraps them, and the portal Training workload just runs that script via `Command: bash, Arguments: /data/training/.../run.sh`.

---

## 0. One-time prep in the workspace

```bash
# Stage Phase 5 scripts on the PVC.
cd /data/training
git clone <this repo URL or symlink> runai-repo 2>/dev/null || (cd runai-repo && git pull)
ls -la runai-repo/phase5-profile/

# Confirm tools available; install if needed (see notes.md).
which nsys ncu python
nsys --version || echo "no nsys"
ncu --version  || echo "no ncu"

# Make a directory for profile outputs.
mkdir -p /data/training/p5-profile

# Quick dependency check inside the venv:
source /data/training/gpt/venv-nanogpt/bin/activate
python -c "from torch.profiler import profile; print('torch.profiler OK')"
```

---

## 5a — `torch.profiler` on Phase 3

### Wrapper script (write once to PVC)

```bash
cat > /data/training/p3-gemma/run_torch_profile.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
hostname; date

source /data/training/gpt/venv-nanogpt/bin/activate
python -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"

cd /data/training/p3-gemma
mkdir -p out trace_torch_phase3

export HF_HOME=/data/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OUTPUT_DIR=/data/training/p3-gemma/out
export MAX_STEPS=20                # short, traced run
export PER_DEVICE_BS=4
export GRAD_ACCUM=4
export LR=2e-4
export LORA_R=16
export MAX_SEQ_LEN=1024
export PROFILE_OUT=/data/training/p3-gemma/trace_torch_phase3
export PROFILE_WARMUP=5
export PROFILE_ACTIVE=5

# PYTHONSTARTUP injects profile_phase3.py BEFORE train.py executes,
# which registers a TrainerCallback on SFTTrainer.
PYTHONSTARTUP=/data/training/runai-repo/phase5-profile/profile_phase3.py \
  torchrun --standalone --nproc_per_node=4 train.py 2>&1 | tee out/torch_profile.log

ls -la /data/training/p3-gemma/trace_torch_phase3/
echo "Open these in TensorBoard:  tensorboard --logdir /data/training/p3-gemma/trace_torch_phase3"
echo "Or in chrome://tracing :    find trace_torch_phase3 -name '*.pt.trace.json'"
RUN
chmod +x /data/training/p3-gemma/run_torch_profile.sh
```

### Launch as Training workload

In the portal, clone your working Phase 3 Training workload and change only:

| Field | Value |
|---|---|
| **Name** | `p5a-torch-profile-phase3` |
| **Command** | `bash` |
| **Arguments** | `/data/training/p3-gemma/run_torch_profile.sh` |

Everything else (image, GPUs=4, PVC, env) stays identical.

### View the trace

After the workload completes (~3–5 min), open a workspace and:

```bash
# Option A — TensorBoard inside the workspace (port-forward in Run:AI portal)
source /data/training/gpt/venv-nanogpt/bin/activate
pip install --quiet --upgrade torch-tb-profiler  # adds the "PyTorch Profiler" plugin
tensorboard --logdir /data/training/p3-gemma/trace_torch_phase3 --port 6006

# Option B — Chrome trace viewer (offline)
ls /data/training/p3-gemma/trace_torch_phase3/*.pt.trace.json
# scp the .pt.trace.json to your laptop, open chrome://tracing, drag the file in.
```

What you should see, on a healthy run:
- ~95% GPU busy bars per rank (no large gaps between steps).
- Top self-CUDA-time ops are matmuls + SDPA.
- "Memory" tab shows peak ~28–34 GB per rank.

If `aten::copy_`, `aten::to`, or `aten::layer_norm` is in the top 5 — there's a quick win hiding.

---

## 5b — `nsys` on Phase 3

### Wrapper

```bash
cat > /data/training/p3-gemma/run_nsys.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
hostname; date
which nsys && nsys --version

source /data/training/gpt/venv-nanogpt/bin/activate
PROFILE_TARGET=phase3 \
  bash /data/training/runai-repo/phase5-profile/run_nsys.sh
RUN
chmod +x /data/training/p3-gemma/run_nsys.sh
```

### Launch as Training workload

| Field | Value |
|---|---|
| **Name** | `p5b-nsys-phase3` |
| **Command** | `bash` |
| **Arguments** | `/data/training/p3-gemma/run_nsys.sh` |
| **GPUs** | `4` |

`nsys` captures 60 s starting at +10 s (skips model load). Output file lands at `/data/training/p5-profile/phase3-<timestamp>.nsys-rep`, typically 300–800 MB.

### Get the report to your laptop

```bash
# In the workspace, the file is at:
ls -lh /data/training/p5-profile/*.nsys-rep

# From your laptop (preferred — requires kubectl with read access to your namespace):
kubectl -n runai-<your-project> get pods | grep p5
kubectl -n runai-<your-project> cp <pod-name>:/data/training/p5-profile/<file>.nsys-rep ~/Downloads/

# Or if you can mount the PVC locally (some clusters allow this via runai download).
# Or simplest: open a workspace, then "Download" the file from JupyterLab's file browser.
```

### Open in Nsight Systems GUI

Install Nsight Systems on your laptop from <https://developer.nvidia.com/nsight-systems>. **Use 2025.4 or newer for full Blackwell sm_100 support** (older versions still open the file but lose some metrics).

Quick checklist when reading the timeline:

1. Pick rank 0 → expand **CUDA HW** → should be a wall of color (kernels). White gaps = idle GPU.
2. Expand **NCCL** → look at all-reduce durations. They should sit *inside* the backward time window, not after it.
3. Expand **OS Runtime** → look for blue `recv`/`epoll_wait` waiting on the dataloader.
4. Compare ranks side by side (Ctrl-click): stragglers stand out.

---

## 5c — `ncu` on a single kernel

### Wrapper

```bash
cat > /data/training/p3-gemma/run_ncu.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
hostname; date
which ncu && ncu --version

source /data/training/gpt/venv-nanogpt/bin/activate

# Profile the standalone benchmark (fast, single-process).
# To filter to specific kernels, set KERNEL_FILTER.
# e.g.  KERNEL_FILTER="gemm|attention"   or   KERNEL_FILTER="ampere_bf16_gemm"
PROFILE_TARGET=bench \
KERNEL_FILTER="${KERNEL_FILTER:-gemm|attention|softmax|layer_norm}" \
LAUNCH_SKIP=300 \
LAUNCH_COUNT=20 \
  bash /data/training/runai-repo/phase5-profile/run_ncu.sh
RUN
chmod +x /data/training/p3-gemma/run_ncu.sh
```

### Launch as Training workload (only 1 GPU!)

| Field | Value |
|---|---|
| **Name** | `p5c-ncu-bench` |
| **Command** | `bash` |
| **Arguments** | `/data/training/p3-gemma/run_ncu.sh` |
| **GPUs** | `1`  (ncu is single-process; multi-GPU adds nothing) |

ncu runs each captured kernel ~10–50× to collect detailed counters, so even a "20 kernel" capture takes 3–8 min wall-clock. Output: `/data/training/p5-profile/ncu-bench-<ts>.ncu-rep` (typically 5–50 MB).

### Open in Nsight Compute GUI

Install Nsight Compute from <https://developer.nvidia.com/nsight-compute>. **2025.x or newer for Blackwell metric sections.**

What to read first when the report loads:

1. **Summary → Top Kernels (by duration)**: which kernel ate the most time?
2. **Click a kernel → Details → "GPU Speed Of Light Throughput"**: shows achieved % of compute peak and memory peak. The bigger of the two tells you the bottleneck class:
   - Compute > 70%, memory < 50% → kernel is compute-bound, optimize tensor-core fraction
   - Memory > 70%, compute < 50% → memory-bound, tile better / fuse
   - Both low → kernel is stall-dominated; see "Warp State Statistics" → top stall reasons
3. **Tensor Core Utilization** (in the same panel for matmuls): should be > 60% for bf16 gemms on GB200. If 0%, you're routing through CUDA cores (likely fp32 path).

You don't need to understand every panel. SOL + Tensor Core utilization gets you 80% of the value.

---

## Optional but useful — quick on-pod summary commands

You don't always need to download .nsys-rep / .ncu-rep to your laptop. For triage:

```bash
# nsys: print a textual stats summary of a capture (CUDA kernel time, NCCL time, ...).
nsys stats --report cudaapisum,cudagpukernsum,nvtxsum /data/training/p5-profile/<file>.nsys-rep | head -100

# ncu: print the per-kernel summary table.
ncu --import /data/training/p5-profile/<file>.ncu-rep --page details --format csv | head -60
```

Those summaries are often enough to tell you "yep, attention forward is 38% of step time" without ever opening a GUI.
