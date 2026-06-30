# Phase 3 — Portal Training-workload guide (Gemma 4 12B-it LoRA)

When CLI `training submit` is blocked by an admin policy on annotations (the
case in the original lab's `ri-fm` project — substitute your own project), drive
Phase 3 from the Run:AI portal. Same pattern as Phase 1 (NCCL/DDP) and the
nanoGPT pretraining demo.

Total wall-clock: ~30–60 minutes for 200 steps on 4× GB200 (most of that is
the first-time weights download). The training itself is fast.

---

## 0. Prereqs (one-time)

1. Sign in to <https://huggingface.co> and open <https://huggingface.co/google/gemma-4-12B-it>.
2. Click **"Access this gated model"** and accept Google's Gemma terms.
3. Generate a token at <https://huggingface.co/settings/tokens> with **read access to gated repos**. Copy it as `hf_xxx...`.
4. (Optional but recommended) confirm the same account can also access `tatsu-lab/alpaca` — that one is not gated, so this should "just work."

---

## 1. Pre-stage the weights on the PVC (saves 5–10 min later)

Spin up the **2-GPU Workspace** you already have from Phase 2c (or a fresh one)
and run this once. After that, every Training workload reuses the cache.

```bash
# In the workspace terminal:
export HF_HOME=/data/hf-cache
mkdir -p $HF_HOME
export HF_TOKEN=hf_xxx_your_token

# Activate the venv from Phase 2c (we'll reuse it).
source /data/training/gpt/venv-nanogpt/bin/activate

pip install --quiet --upgrade \
  'transformers>=4.50' 'trl>=0.11' 'peft>=0.13' \
  'datasets>=3.0' 'accelerate>=1.0' 'bitsandbytes>=0.43' wandb \
  'huggingface_hub[cli]>=0.25'

# Pull Gemma 4 12B-it weights to the PVC (~24 GB; ~3-5 min on a decent link).
huggingface-cli download google/gemma-4-12B-it \
  --local-dir $HF_HOME/hub/google--gemma-4-12B-it \
  --local-dir-use-symlinks False

# Smoke test that it loads in bf16 on one GPU.
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "google/gemma-4-12B-it"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                         attn_implementation="sdpa", device_map="cuda:0")
inputs = tok.apply_chat_template([{"role":"user","content":"Say hello."}],
                                 tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors="pt").to("cuda:0")
out = m.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
PY
```

When that prints a "hello" reply you are ready to launch the 4-GPU Training
workload.

> If the workspace doesn't have the venv from Phase 2c anymore, just create a
> fresh one: `python3 -m venv /data/training/venv-p3 && source /data/training/venv-p3/bin/activate && pip install --upgrade pip 'torch>=2.6' 'transformers>=4.50' 'trl>=0.11' 'peft>=0.13' 'datasets>=3.0' 'accelerate>=1.0' 'bitsandbytes>=0.43' wandb`.

---

## 2. Stage the training script on the PVC

Still in the 2-GPU workspace terminal:

```bash
mkdir -p /data/training/p3-gemma4 && cd /data/training/p3-gemma4
```

Then paste the contents of `phase3-lora/train.py` (in this repo) into
`/data/training/p3-gemma/train.py`. Or — if `git` is available on the pod and
this repo is reachable — clone it:

```bash
git clone <this repo URL> /data/training/runai-repo
cp /data/training/runai-repo/phase3-lora/train.py /data/training/p3-gemma/
```

---

## 3. Launch the 4-GPU Training workload

1. Open your Run:AI portal → **Workloads** → **+ NEW WORKLOAD** → **Training**.
2. **Project**: your project (the original lab used `ri-fm`).
3. **Name**: `p3-gemma4-12b-4gpu`
4. **Cluster**: the GB200 cluster (same one you used for Phase 2c).
5. **Environment (image)**:
   - **Image URL**: `nvcr.io/nvidia/pytorch:25.04-py3` (same as Phase 2c — Blackwell-aware).
   - **No JupyterLab tool needed** — Training workloads run-to-completion.
6. **Compute resource**:
   - **GPU devices**: `4`
   - **Large /dev/shm**: ON
   - CPU / RAM: defaults are fine.
7. **Data sources**:
   - **+ Add data source** → your existing PVC.
   - Mount path: `/data` (same as Phase 2c so the staged venv + weights are visible).
8. **Environment variables** (very important — without these the workload fails):
   - `HF_TOKEN` = your `hf_xxx_...` token
   - `HF_HOME` = `/data/hf-cache`
   - `TRANSFORMERS_NO_ADVISORY_WARNINGS` = `1`
   - `TIKTOKEN_CACHE_DIR` = `/data/tiktoken-cache`  *(belt-and-suspenders, in case any dep pulls tiktoken)*
   - *(optional)* `WANDB_API_KEY`, `WANDB_PROJECT=runai-learn` for live metrics.
9. **Scheduling / Node pool**: GB200 pool.
10. **Annotations / Labels**: **leave empty** (this is what triggered the CLI block).
11. **Command** — paste this into the **Command** field (and leave **Arguments** empty):

```bash
bash -lc 'set -euo pipefail
source /data/training/gpt/venv-nanogpt/bin/activate
nvidia-smi -L
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

mkdir -p /data/training/p3-gemma4/out
cd /data/training/p3-gemma4

export OUTPUT_DIR=/data/training/p3-gemma4/out
export MAX_STEPS=200
export PER_DEVICE_BS=4
export GRAD_ACCUM=4
export LR=2e-4
export LORA_R=16
export MAX_SEQ_LEN=1024

torchrun --standalone --nproc_per_node=4 train.py 2>&1 | tee out/train.log'
```

12. **CREATE TRAINING**.

The workload will go `Pending` → `Running` within ~30 s. Open it and click
**Logs** to follow.

---

## 4. What "good" looks like

### Early logs (first ~30 s)

```
GPU 0: NVIDIA GB200 (UUID: ...)
GPU 1: NVIDIA GB200 (UUID: ...)
GPU 2: NVIDIA GB200 (UUID: ...)
GPU 3: NVIDIA GB200 (UUID: ...)
2.12.1+cu130 True 4
model=google/gemma-4-12B-it  dataset=tatsu-lab/alpaca  output=/data/training/p3-gemma4/out
steps=200  per_device_bs=4  grad_accum=4  lr=0.0002  lora_r=16  seq_len=1024
```

### Loading + tokenizing (~30–90 s)

```
Loading checkpoint shards: 100%|##########| 6/6 [00:08<00:00]
Map: 100%|##########| 52002/52002 [00:11<00:00]
trainable params: 51,019,776 || all params: 12,036,xxx,xxx || trainable%: 0.42
```

That `trainable%: ~0.4%` is the LoRA signature — you're updating only the
adapter weights, not the full 12B.

### Training (~5–10 min for 200 steps)

```
{'loss': 1.92, 'learning_rate': 4e-5, 'epoch': 0.02, 'step': 10}
{'loss': 1.41, 'learning_rate': 1.6e-4, 'epoch': 0.05, 'step': 30}
{'loss': 1.18, 'learning_rate': 1.9e-4, 'epoch': 0.10, 'step': 60}
{'loss': 1.06, 'learning_rate': 1.4e-4, 'epoch': 0.16, 'step': 100}
{'loss': 0.94, 'learning_rate': 5e-5, 'epoch': 0.27, 'step': 170}
{'loss': 0.91, 'learning_rate': 1e-6, 'epoch': 0.32, 'step': 200}
{'train_runtime': 480.1, 'train_samples_per_second': 26.6, 'train_steps_per_second': 0.42, 'epoch': 0.32}
saved LoRA adapter to /data/training/p3-gemma4/out
```

Headline numbers to expect on 4× GB200, seq_len=1024, LoRA r=16:
- `train_loss` drops from ~2.0 → ~0.9 over 200 steps.
- `train_samples_per_second` ≈ 25–35 (effective batch = 4 GPUs × 4 BS × 4 accum = 64).
- VRAM: ~28–34 GB / GPU. (`nvidia-smi` from a separate workspace into the same node.)
- Adapter size on disk: ~100–200 MB under `out/`.

If `train_loss` stays > 1.6 after 100 steps:
- Double-check `tok.chat_template` is non-empty (Gemma 4 -it sets this by default). Add `print(repr(tok.chat_template)[:200])` in `train.py` to confirm.
- Check `nvidia-smi` shows all 4 GPUs at >70% util. If one is idle, NCCL fell back to PCIe — set `NCCL_DEBUG=INFO` and resubmit.

---

## 5. After it finishes

The workload status flips to **Completed**. The adapter is at
`/data/training/p3-gemma4/out` on the PVC and survives the pod going away.

Test it from any 1-GPU workspace:

```bash
source /data/training/gpt/venv-nanogpt/bin/activate
export HF_HOME=/data/hf-cache
python - <<'PY'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "google/gemma-4-12B-it"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16,
                                            attn_implementation="sdpa", device_map="cuda")
model = PeftModel.from_pretrained(base, "/data/training/p3-gemma/out").eval()

prompts = [
    "Write a haiku about NVIDIA Blackwell GPUs.",
    "Explain LoRA fine-tuning in one short paragraph.",
    "Give me 3 bullet tips for debugging NCCL hangs.",
]
for p in prompts:
    inputs = tok.apply_chat_template([{"role":"user","content":p}],
                                     tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    print("Q:", p)
    print("A:", tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
    print("---")
PY
```

Compare a few of those outputs to the base model (load without `PeftModel.from_pretrained`) — you should see noticeably tighter, more "Alpaca-style" responses from the LoRA-tuned version.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: ... is a gated repo` | `HF_TOKEN` not set or hasn't accepted license | Set `HF_TOKEN` env var on the workload **and** accept the license in browser. |
| First step takes >5 min, then fails | Downloading 12B over a slow link | Pre-stage weights on the PVC first (Step 1). |
| `AssertionError: gradient_accumulation_steps ... world_size` | Same as Phase 2c | We set `GRAD_ACCUM=4` and `GPUS=4`. If you change `GPUS`, keep `GRAD_ACCUM` divisible by it. |
| `trainable params: 0` | Wrong `target_modules` for the model | Print `model` once to inspect layer names; Gemma 4 uses `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`. |
| `RuntimeError: FlashAttention only supports ...` | Image picked up FA2 anyway | Force `attn_implementation="sdpa"` (already the default in `train.py`); also `export TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa`. |
| Loss = NaN at step 1 | bf16 + LoRA on Gemma sometimes needs scaled init | Lower `lr` to `1e-4`, or upgrade `peft>=0.14`. |
| One GPU sitting idle | Misconfigured workload (e.g. command launched plain `python` not `torchrun`) | Confirm the command uses `torchrun --nproc_per_node=4`. |

When `train_loss` is descending smoothly and the adapter saved, **Phase 3 is
complete**. Next stop is [Phase 4](../phase4-fsdp/notes.md) — Gemma 4 31B
with FSDP, or you can stop here and call it a day. You've now used the cluster
end-to-end for pretrain (Phase 2c) and instruction-tune (Phase 3).
