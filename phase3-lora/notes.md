# Phase 3 — LoRA fine-tune **Gemma 4 12B-it**

LoRA SFT of Google's [Gemma 4 12B-it](https://huggingface.co/google/gemma-4-12B-it)
on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instruction
dataset, using [`trl`](https://github.com/huggingface/trl)'s `SFTTrainer`.

Defaults:
- **Model**: `google/gemma-4-12B-it` (Apache 2.0, open-weights but **gated** — see prereqs)
- **Dataset**: `tatsu-lab/alpaca`
- **Adapter**: LoRA r=16 on all attention + MLP projections
- **Precision**: bf16 + gradient checkpointing
- **Attention**: SDPA (Gemma 4's softcap is best served by SDPA; FA2 is not recommended)
- **Hardware**: 4× GB200 (matches Phase 2c setup — same Training-workload recipe)

The folder is named `phase3-lora` (size-agnostic). The default model is 12B (the
closest Gemma 4 analog to the canonical "7B-class" LoRA recipe), but you can
swap in any of the Gemma 4 sizes — see the cheat-sheet below.

---

## Prereqs — one-time

Gemma 4 is **gated** on Hugging Face. Before launching the workload:

1. Open <https://huggingface.co/google/gemma-4-12B-it> in a browser, sign in.
2. Click **"Access this gated model"**, accept Google's Gemma terms.
3. Generate a token at <https://huggingface.co/settings/tokens> with **read** access to gated repos.
4. Save it locally as `HF_TOKEN` (and pass it into the workload as an env var — see `portal-guide.md`).

You'll know access works when `huggingface-cli download google/gemma-4-12B-it --local-dir-use-symlinks=False` succeeds.

---

## Run it

The portal Training-workload path (recommended on `ri-fm`, since CLI submit is
blocked by admission policy) is documented step-by-step in
[`portal-guide.md`](./portal-guide.md).

If CLI submit is open on your project:

```bash
source scripts/env.sh
bash phase3-lora/submit.sh
```

To bump scale (still LoRA, just more steps / bigger batch):

```bash
GPUS=4 MAX_STEPS=1000 PER_DEVICE_BS=4 GRAD_ACCUM=4 \
  bash phase3-lora/submit.sh
```

To swap to a smaller/faster Gemma 4 variant:

```bash
MODEL_ID=google/gemma-4-E4B-it GPUS=4 bash phase3-lora/submit.sh
```

To stretch up to the dense flagship (still LoRA — full SFT of 31B is Phase 4):

```bash
MODEL_ID=google/gemma-4-31B-it GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=8 \
  bash phase3-lora/submit.sh
```

---

## Gemma 4 size cheat-sheet

| Model | Params (effective / total) | bf16 weights | LoRA-SFT footprint per GPU | Recipe |
|---|---|---|---|---|
| `gemma-4-E4B-it` | 4.5B / 8B | ~9 GB | ~14–18 GB | 1–2 GPUs, fastest |
| `gemma-4-12B-it` | ~12B dense | ~24 GB | ~28–34 GB | **default — 4 GPUs** |
| `gemma-4-26B-A4B-it` | 4B active / 26B total (MoE) | ~52 GB | needs expert sharding | skip for first run |
| `gemma-4-31B-it` | 31B dense | ~62 GB | ~70–80 GB w/ grad-ckpt | LoRA fits on 1× GB200; FSDP on 4 is faster |

GB200 has ~186 GB HBM3e per GPU, so all dense variants above are LoRA-comfortable
on a single GPU. Going to 4 GPUs gets you a ~4× wall-clock speedup via DDP.

---

## What to watch

- **First ~60 s**: downloading the 12B base (~24 GB) — slow first time, fast if `HF_HOME=/work/hf` (PVC-backed).
- **Steps 1–50**: `train_loss` should drop from ~2.3 to ~1.4 (Gemma 4 -it starts smart; the SFT is teaching it the Alpaca response format).
- **Steps 50–200**: `train_loss` settles around ~1.1–1.3. Watch `grad_norm` — should stay < 1.5.
- **`train_runtime` and `train_samples_per_second`** in the final log line give you throughput.
- **VRAM per GPU**: ~28–34 GB for LoRA r=16 + grad-ckpt on 12B at seq_len=1024. Well under GB200's ~186 GB.

If `train_loss` plateaus above 1.6 after 200 steps, your prompt formatting probably
doesn't match Gemma 4's chat template. The `train.py` in this folder uses
`tokenizer.apply_chat_template`, which is the correct path for `-it` models.

---

## After training

The adapter is saved under `$OUTPUT_DIR` (default `/work/p3-out` if PVC, else
`/tmp/p3-out`). LoRA adapters for 12B are typically **~80–200 MB** (much smaller than the base).

Quick inference smoke test (run in the same workspace pod):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_id = "google/gemma-4-12B-it"
adapter_dir = "/work/p3-out"

tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
model = PeftModel.from_pretrained(base, adapter_dir).eval()

inputs = tok.apply_chat_template(
    [{"role": "user", "content": "Write a haiku about NVIDIA Blackwell GPUs."}],
    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
).to("cuda")
out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

---

## Going to full SFT (no LoRA)

For 4× GB200, full bf16 SFT of 12B with grad-ckpt + FSDP is comfortable. Two paths:

1. Edit `train.py`: drop `peft_config=peft_cfg`, set `gradient_checkpointing=True` and `optim="adamw_torch_fused"`, then `torchrun --nproc_per_node=4 train.py`. PyTorch's default DDP + grad-ckpt suffices for 12B on 4× GB200.
2. For 31B, jump to [Phase 4](../phase4-fsdp/notes.md) — same recipe with FSDP `FULL_SHARD`.

When loss is decreasing and the adapter saved, move on to Phase 4 (or stop here — Phase 4 is optional).
