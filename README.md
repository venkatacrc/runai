# Run:AI Learning Lab — 8× GB200 in a Day

A phased, hands-on tour of the [Run:AI](https://www.run.ai/) platform using
open-source training code. Designed for one productive day on 8× NVIDIA GB200
(Blackwell, `sm_100`).

> The original walkthrough was done against `https://runai.prod.walmart.com` in
> a project called `ri-fm`. Everywhere those names appear in this repo they're
> meant to be replaced with your own — see `scripts/env.sh`.

## Phases at a glance

| Phase | Goal | Wall-clock | Key artifact |
|------:|------|------------|--------------|
| **0** | Install CLI, login, verify quota + project | 15 min | [`docs/phase0-setup.md`](docs/phase0-setup.md) |
| **1** | Distributed-training sanity: NCCL all-reduce + tiny DDP | 20 min | [`phase1-sanity/`](phase1-sanity/) |
| **2** | Pretrain a small LLM — [nanochat](https://github.com/karpathy/nanochat) `d20` (~560M), or [nanoGPT](https://github.com/karpathy/nanoGPT) as the easier baseline | 1.5–2 h | [`phase2-nanochat/`](phase2-nanochat/) |
| **3** | LoRA fine-tune **Gemma 4 12B-it** (Apache 2.0, gated) on Alpaca with TRL | 1–2 h | [`phase3-lora/`](phase3-lora/) |
| **4** | *(stretch)* Push toward **Gemma 4 31B** full SFT with FSDP | 2–4 h | [`phase4-fsdp/`](phase4-fsdp/) |
| **5** | Profile + optimize with **`torch.profiler`**, **`nsys`**, **`ncu`** | 1–2 h | [`phase5-profile/`](phase5-profile/) |

Each phase folder contains some of:
- `train.py` (or pointer to upstream code)
- `submit.sh` — the `runai` CLI command that launches the job
- `portal-guide.md` — the portal-click equivalent for when CLI submit is blocked by admission policy
- `notes.md` — what to watch for, expected throughput, common failures

## Why this order

1. **Phase 0** proves *you* can talk to the cluster.
2. **Phase 1** proves the *cluster* can talk to itself (NCCL, multi-rank).
3. **Phase 2** is a "real" end-to-end LLM run that fits on the node — small enough that mistakes cost minutes, not hours.
4. **Phase 3** is the practical skill most teams actually need: LoRA fine-tune an off-the-shelf instruction model.
5. **Phase 4** stretches into "production-scale" territory — FSDP, sharded checkpoints, gradient checkpointing.
6. **Phase 5** closes the loop: now that things run, find out *where the time goes*, then make them faster. Three lenses (Python-level `torch.profiler`, system-wide `nsys`, kernel-level `ncu`), each telling you something the others can't.

## Quick start

```bash
# 1) Get the repo onto a machine that can reach your Run:AI portal.
git clone https://github.com/venkatacrc/runai.git
cd runai

# 2) Fill in the placeholders in scripts/env.sh:
#      RUNAI_URL, RUNAI_PROJECT, RUNAI_CLUSTER, RUNAI_NODE_POOL,
#      RUNAI_PVC, RUNAI_IMAGE, HF_TOKEN (for Phase 3+)
$EDITOR scripts/env.sh
source scripts/env.sh

# 3) Phase 0 — install the runai CLI and confirm GPU quota.
$EDITOR docs/phase0-setup.md     # follow it top to bottom

# 4) Then each phase, in order:
bash phase1-sanity/submit.sh     # OR follow phase1-sanity/portal-guide.md if CLI is blocked
bash phase2-nanochat/submit.sh
bash phase3-lora/submit.sh
# ...etc
```

If `runai workspace submit` or `runai training submit` errors with
`Field: annotations, the administrator prohibited adding new items`, switch to
the **portal flow** documented in each phase's `portal-guide.md` — it does
exactly what the CLI would do, just clicked through the UI.

## Hardware budget (8× GB200)

- 8 × Blackwell GPUs, ~186 GB HBM3e each → ~**1.5 TB** of GPU memory total.
- NVLink (typically NVL72 backplane) inside the node — NCCL `all_reduce` is essentially free.
- Compute capability `sm_100` → need **CUDA ≥ 13** and **PyTorch ≥ 2.12** (or NGC `pytorch:25.04-py3`+).

Useful sanity numbers:
- Gemma 4 E4B in bf16 ≈ 9 GB weights → trivially fits on 1 GPU (fast LoRA).
- Gemma 4 12B in bf16 ≈ 24 GB weights → 1 GPU comfortable, 4-GPU DDP very fast.
- Gemma 4 31B in bf16 ≈ 62 GB weights → FSDP across 2–4 GPUs for full SFT.
- nanochat d20 trains in **~3 h on 8× H100**; expect **~1.5–2 h on 8× GB200**.

## Conventions

- All cluster URLs assume `${RUNAI_URL}` (set in `scripts/env.sh`).
- Default container image: `nvcr.io/nvidia/pytorch:25.04-py3` (Blackwell-aware). Swap the registry prefix if your cluster mirrors NGC internally.
- Jobs use ephemeral working dirs; persistent state (model weights, datasets, checkpoints, LoRA adapters) lives on a Run:AI PVC mounted at `/work` (CLI flow) or `/data` (portal flow). The PVC name goes in `RUNAI_PVC`.
- HuggingFace `HF_TOKEN` is only needed for gated models (Phase 3+ default is Gemma 4, which is gated under Google's Gemma terms — accept the license at <https://huggingface.co/google/gemma-4-12B-it> first).

## Repo layout

```
.
├── README.md                 ← you are here
├── scripts/env.sh            ← single source of truth for all cluster env vars
├── docs/
│   └── phase0-setup.md
├── phase1-sanity/            ← NCCL all-reduce + tiny DDP
├── phase2-nanochat/          ← Karpathy nanochat speedrun (with nanoGPT fallback)
├── phase3-lora/              ← Gemma 4 12B-it LoRA SFT on Alpaca with TRL
├── phase4-fsdp/              ← (stretch) Gemma 4 31B full SFT with FSDP
└── phase5-profile/           ← torch.profiler / nsys / ncu + optimization playbook
```

## What "done" looks like

Each phase ends with a measurable signal:
- **Phase 1**: `busBW ≈ 500–800 GB/s` for ≥1 GB NCCL all-reduce on NVL5; DDP smoke loss strictly decreases.
- **Phase 2**: A pretrained model with declining `val_bpb` / val-loss curve; checkpoint saved.
- **Phase 3**: LoRA adapter saved to PVC (~100–200 MB) with `train_loss` decreasing from ~2.0 → < 1.7; A/B vs. base model shows tighter instruction-following.
- **Phase 4**: 31B model loads sharded across 4–8 GPUs without OOM; FSDP step time stable.
- **Phase 5**: Speedup table (baseline → optimized) with attributable changes — target 1.5–2× on Phase 2 or Phase 3.

## License

The code in this repository is provided as a learning aid; pick whichever
OSI-approved license suits your needs (MIT or Apache-2.0 are common choices).
The third-party models and datasets it downloads keep their own licenses —
notably **Gemma** is licensed under Google's [Gemma terms](https://ai.google.dev/gemma/terms)
(commercially permissive but with a use-restriction notice).
