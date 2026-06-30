# Phase 4 (optional) — Push toward Gemma 4 31B

The point of this phase is to feel what *production-scale* fine-tuning looks
like on 8× GB200. You have ~1.5 TB of HBM total, which is enough to hold
**Gemma 4 31B** in bf16 (~62 GB weights) **plus** Adam state plus activations,
even without parameter sharding — but the interesting exercise is doing it the
"right" way with FSDP so the same recipe scales to truly large dense models.

Gemma 4's largest dense variant is `google/gemma-4-31B-it`. The 26B-A4B MoE
model is also in-family but adds expert-routing complexity that's out of scope
for a learning lab.

---

## Two viable recipes

### A) LoRA on Gemma 4 31B (safer, ~1–2 h)

Same `train.py` as Phase 3, but with:

```bash
MODEL_ID=google/gemma-4-31B-it \
GPUS=8 \
PER_DEVICE_BS=1 GRAD_ACCUM=16 \
MAX_STEPS=300 \
MAX_SEQ_LEN=2048 \
bash phase3-lora/submit.sh
```

This loads the 31B sharded across GPUs (transformers' `device_map="auto"` is
implicit when there's enough memory across visible devices). LoRA adapters
are tiny, optimizer state is tiny. On 8× GB200 you should comfortably see
~3–6 step/s at `seq_len=2048`.

### B) Full SFT with FSDP (the real deal, ~3–4 h)

This is where you actually exercise FSDP-style parameter sharding. Use
HuggingFace `accelerate` with FSDP, or DeepSpeed ZeRO-3. The submit script
below is a sketch you can fill in once Phases 0–3 are clean.

```bash
# Inside the pod (sketch):
accelerate config --config_file fsdp.yaml   # one-time, choose FSDP, BF16, full-shard, transformer_auto_wrap

accelerate launch --config_file fsdp.yaml train.py \
  --model_id google/gemma-4-31B-it \
  --dataset_id HuggingFaceH4/ultrachat_200k \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048 \
  --bf16
```

Key knobs:

| Knob | Why |
|---|---|
| `fsdp_sharding_strategy: FULL_SHARD` | shards params, grads, optimizer — the whole point |
| `fsdp_transformer_layer_cls_to_wrap: Gemma4DecoderLayer` | wraps each block; the right unit for activation checkpointing (check actual class name with `print(model)`) |
| `fsdp_cpu_offload: false` | you have enough VRAM; don't trade GPU↔CPU bandwidth |
| `gradient_checkpointing: true` | trades compute for memory; mandatory for full SFT at 31B+ |
| `fsdp_use_orig_params: true` | required for some optimizers / `torch.compile` |
| `attn_implementation: sdpa` | safer than FA2 for Gemma's softcap |

---

## What to watch

- Time-per-step (look for stability; if it varies >2x step-to-step, NCCL is unhappy)
- `gpu_mem_max` per rank (`nvidia-smi --query-gpu=memory.used --format=csv -l 5`)
- Loss curve — same shape as smaller runs but slower

---

## When to stop

If you can show:

- A 31B Gemma 4 model loading across 8 GPUs without OOM, AND
- Loss decreasing for 50+ steps,

…you've effectively passed Phase 4. Save the resulting adapter (LoRA path) or
checkpoint shard (FSDP path) and call it a day.

---

## When *not* to do Phase 4

Skip this if:

- You're new to FSDP/DeepSpeed and Phases 0–3 took longer than expected.
- You don't have a PVC — 31B checkpoint shards alone are ~62 GB (and FSDP shards
  blow that up), you don't want those evaporating with the pod.
- You haven't accepted the Gemma 4 license on HuggingFace yet (same gate as Phase 3).
