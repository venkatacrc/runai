"""LoRA SFT of Gemma 4 with TRL.

Defaults: google/gemma-4-12B-it + tatsu-lab/alpaca, bf16, LoRA(r=16), seq_len=1024.
Single GPU is fine; multi-GPU via `torchrun --nproc_per_node=N`.

Gemma 4 specifics:
- Gated repo — set HF_TOKEN in the env (and accept the license on huggingface.co).
- attn_implementation="sdpa"  (FA2 is not recommended for Gemma's softcap).
- Uses the model's native chat template via tokenizer.apply_chat_template,
  which is correct for instruction-tuned ("-it") variants.

Override anything via env vars (no argparse here — this is a learning sandbox):

  MODEL_ID        default google/gemma-4-12B-it
  DATASET_ID      default tatsu-lab/alpaca
  OUTPUT_DIR      default /tmp/p3-out
  MAX_STEPS       default 200
  PER_DEVICE_BS   default 4
  GRAD_ACCUM      default 4
  LR              default 2e-4
  LORA_R          default 16
  MAX_SEQ_LEN     default 1024
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    model_id = env("MODEL_ID", "google/gemma-4-12B-it")
    dataset_id = env("DATASET_ID", "tatsu-lab/alpaca")
    output_dir = env("OUTPUT_DIR", "/tmp/p3-out")
    max_steps = int(env("MAX_STEPS", "200"))
    per_device_bs = int(env("PER_DEVICE_BS", "4"))
    grad_accum = int(env("GRAD_ACCUM", "4"))
    lr = float(env("LR", "2e-4"))
    lora_r = int(env("LORA_R", "16"))
    max_seq_len = int(env("MAX_SEQ_LEN", "1024"))

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"model={model_id}  dataset={dataset_id}  output={output_dir}", flush=True)
        print(
            f"steps={max_steps}  per_device_bs={per_device_bs}  "
            f"grad_accum={grad_accum}  lr={lr}  lora_r={lora_r}  seq_len={max_seq_len}",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if Path(dataset_id).is_dir():
        ds = load_from_disk(dataset_id)
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]
    else:
        ds = load_dataset(dataset_id, split="train")

    has_chat_template = bool(getattr(tok, "chat_template", None))

    def format_example(ex: dict) -> dict:
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")
        user = f"{instr}\n\n{inp}".rstrip() if inp else instr
        if has_chat_template:
            text = tok.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": out},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt = (
                f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
                if inp
                else f"### Instruction:\n{instr}\n\n### Response:\n"
            )
            text = prompt + out + tok.eos_token
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    cfg_kwargs = dict(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=max_steps,
        save_total_limit=1,
        gradient_checkpointing=True,
        packing=False,
        dataset_text_field="text",
        report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
        ddp_find_unused_parameters=False,
    )
    import inspect
    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig:
        cfg_kwargs["max_length"] = max_seq_len
    elif "max_seq_length" in sig:
        cfg_kwargs["max_seq_length"] = max_seq_len
    cfg = SFTConfig(**cfg_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    if rank == 0:
        print(f"saved LoRA adapter to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
