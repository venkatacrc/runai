# Phase 2 — Pretrain a small LLM (nanochat + nanoGPT)

This phase has two related targets:

1. **[karpathy/nanochat](https://github.com/karpathy/nanochat)** — the
   ambitious one. A single-file, end-to-end LLM training harness:
   tokenizer → pretrain → midtrain → SFT → RL → eval. `runs/speedrun.sh` is
   the reference recipe, designed for 8× H100 (~3 h there); on 8× GB200 expect
   ~1.5–2 h.

2. **[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)** — the safer
   baseline. Same author, simpler scope (pretrain only), no `torch.compile` /
   FA3 requirement, runs cleanly on the stock NGC PyTorch image. Train
   Shakespeare-char as a 5-minute warmup, then GPT-2 124M on OpenWebText for
   a real demo.

**Recommended path for first-time GB200 use:** start with nanoGPT (it
"just works" on the stock image) and graduate to nanochat once you have
a working `torch.compile` / FlashAttention setup in your container.

The rest of this page is the nanochat speedrun reference; the nanoGPT path is
shorter and well-documented upstream — clone the repo, run `data/shakespeare_char/prepare.py`,
then `torchrun --standalone --nproc_per_node=N train.py config/train_shakespeare_char.py`
(or `config/train_gpt2.py` for the bigger demo). Use the same Run:AI portal
Training-workload flow documented in [`phase3-lora/portal-guide.md`](../phase3-lora/portal-guide.md).

## Why this is a great learning target

- One repo, ~3000 lines, no abstractions, no frameworks-of-frameworks.
- Touches every stage of the modern LLM pipeline.
- Has a published leaderboard for the d24/d26 ("GPT-2 grade") tier.
- Comes with a chat UI you can talk to when it's done.

## What `d20` is

`--depth=20` → 20-layer transformer, ~560M params, the "$100 speedrun" tier.
All other hyperparameters (width, heads, LR, training horizon, weight decay)
are derived automatically. To go bigger later, increase `--depth` (24, 26, 30…).

## Run it

```bash
source scripts/env.sh
bash phase2-nanochat/submit.sh
```

Tail the logs:

```bash
runai workload logs p2-nanochat-... -f
```

## What to watch for

Speedrun prints these signposts in order:

1. **Tokenizer training** — fast (a few minutes), reads from the FineWeb / ClimbMix shards it downloads.
2. **Base pretrain** — the big chunk. Look at:
   - `val_bpb` (validation bits-per-byte) trending **down** from ~1.0 toward ~0.75
   - `train/tok_per_sec` — should be **much higher than 8×H100's number** on GB200
   - `train/mfu` — model FLOPs utilization; >35% is healthy
3. **Midtrain** — a short follow-on on a different mix.
4. **SFT** — supervised fine-tune on SmolTalk-style conversations.
5. **RL** — short RLHF / preference step.
6. **Eval** — DCLM CORE score (the leaderboard metric), MMLU, GSM8K, HumanEval mini, ARC.

A successful run ends with a `nanochat_report.md` summary file and saved
checkpoints under `checkpoints/`.

## If you persisted to a PVC

The final chat-stage checkpoint lives at
`$RUNAI_PVC:/work/nanochat/checkpoints/`. To talk to it interactively, launch
a workspace (Jupyter or shell) on the same PVC and run:

```bash
cd /work/nanochat
source .venv/bin/activate
python -m scripts.chat_web   # then open the URL it prints
```

## Tuning knobs

| What you might want | How |
|---|---|
| Shorter run (just see it work, ~10 min) | edit `runs/speedrun.sh` and set `--depth=12` (GPT-1 sized) and a small `--num-iterations` |
| Bigger model (~1.2B, "GPT-2 grade") | `DEPTH=26 bash phase2-nanochat/submit.sh` (uses ~more memory + ~3× wall clock) |
| Force fp8 on Blackwell | check `nanochat/common.py` for `NANOCHAT_DTYPE` — default bf16 on sm_100 is fine; fp8 is experimental |
| OOM | reduce `--device-batch-size` in `runs/speedrun.sh` (32 → 16 → 8) |
| Resume / iterate | run base_train.py directly with `--run=myrun --model-tag=myrun` |

When the report file is written and `val_bpb` is below ~0.75, the speedrun was
a success. Move on to [Phase 3](../phase3-lora/notes.md).
