# Phase 5 — Optimization playbook (findings → fixes)

Profiling is only useful if it leads to a fix. This is the cookbook: for each
symptom you might see in `torch.profiler` / `nsys` / `ncu`, the candidate
optimizations to try and the expected speedup on GB200.

The two real targets in this lab:

| Workload | Baseline today | Target after Phase 5 |
|---|---|---|
| **Phase 2c** — nanoGPT GPT-2 124M, 4× GB200 | 2.6 s/iter, MFU ~28% | < 1.5 s/iter, MFU > 45% |
| **Phase 3**  — Gemma 4 12B LoRA, 4× GB200   | 4.0 s/step, 15.9 samp/s | < 2.5 s/step, > 25 samp/s |

---

## Symptom → fix table

### 1. Big white gaps in the `nsys` CUDA HW row between steps (GPU is idle)

**Likely causes:**
- Dataloader is single-threaded or hits disk on every step.
- Python overhead between steps (graph rebuilds, parameter copies).
- A `torch.compile` recompile happening mid-run.

**Fixes (try in order):**
1. `dataloader_num_workers >= 4`, `pin_memory=True`, `persistent_workers=True`.
2. Pre-tokenize / cache the dataset (already done for Phase 3 via `save_to_disk`).
3. If using `torch.compile`, mark inputs with `dynamic=False` or set `TORCH_LOGS=recompiles` to see recompile causes.

**Phase 3 specifically:** `SFTConfig` defaults to 0 dataloader workers. Bump to 4:

```python
cfg_kwargs["dataloader_num_workers"] = 4
cfg_kwargs["dataloader_pin_memory"] = True
```

**Expected win:** 5–15% on Phase 3 if dataloader was the gap.

---

### 2. NCCL all-reduce dominates a big slice of step time

**Symptom in nsys:** the `NCCL` row has all-reduce blocks that don't overlap with backward; they sit at the *end* of each step.

**Likely cause:** DDP isn't bucketing right, or there's a forced sync (e.g. unnecessary `.item()` / `.cpu()` call in the training loop).

**Fixes:**
1. Confirm `find_unused_parameters=False` (already set in our `train.py`).
2. Increase DDP bucket size: `DDP(..., bucket_cap_mb=200)`.
3. For full SFT (no LoRA), switch to FSDP with `BACKWARD_PRE` — overlaps comm with previous-layer backward.
4. Grep your training loop for `.item()`, `.cpu()`, `.tolist()` — every one is a sync point.

**Expected win:** 10–25% on multi-GPU runs where comm wasn't overlapped.

---

### 3. SDPA / matmul kernels show < 30% Tensor-Core utilization in `ncu`

**Symptom in ncu:** "Tensor Active" or "Tensor Pipe Active" metric is low for a matmul kernel.

**Likely causes:**
- Compute is happening in fp32 (TF32 or full fp32), not bf16/fp8.
- Tensor shapes aren't multiples of 8 (or 16/64 on Blackwell), preventing the most efficient tensor-core paths.

**Fixes:**
1. Confirm `bf16=True` in `SFTConfig` (already set).
2. `torch.set_float32_matmul_precision('high')` at program start — opts every remaining fp32 matmul into TF32 on Tensor Cores.
3. Ensure sequence length is a multiple of 16 (we use 1024 ✅) and batch sizes are multiples of 8.
4. Try `attn_implementation="flash_attention_3"` if the venv has it (Gemma 4's softcap may need a recent FA3 with softcap support).

**Expected win:** 15–40% on the affected kernels, often 5–15% overall.

---

### 4. `aten::copy_`, `aten::to`, or `aten::layer_norm` appears in the top-5 of `torch.profiler` self CUDA time

**Likely cause:** unintended dtype casts (bf16 ↔ fp32) on every iteration, or non-fused LayerNorm.

**Fixes:**
1. Audit any `.float()` / `.to(torch.float32)` calls inside the model — keep activations in bf16 end-to-end except for the loss reduction.
2. Use `torch.compile` (mode="default" or "reduce-overhead") — it fuses LayerNorm + matmul.
3. Switch LayerNorm → `nn.RMSNorm` when targeting Gemma (Gemma 4 already uses RMSNorm internally; this is for custom models like nanoGPT where LayerNorm dominates).

**Expected win:** 5–10% on transformer training when dtype thrash was real.

---

### 5. `torch.compile` is off because the image has no C compiler

**Symptom:** you tried `compile=True` and got `InductorError: ... gcc not found`.

**Fix path:**
1. Easiest: pre-install `build-essential` in a workspace once (`sudo apt-get install -y build-essential`), then `torch.compile` Just Works.
2. Or: install via conda — `conda install -c conda-forge gcc gxx`.
3. Or: pre-bake a custom image that includes gcc/g++ and have the cluster admin make it an Environment asset.

**Expected win when compile is on:** 1.3–2.0× for Phase 2c (we measured nanoGPT MFU 141% reported = ~28% real; compile + FA can push that to 45–55%).

---

### 6. Phase 2c specifically: MFU is "141%" (which is wrong) — what's real?

The nanoGPT estimator uses a hardcoded peak-flops number for A100/H100. GB200 bf16 peak is ~5× higher per GPU. So real MFU = reported × (1 / ~5) ≈ **28% on GB200**.

Wins available, in order of impact:

| Change | Expected new step time | Notes |
|---|---|---|
| Enable `torch.compile` | 1.6 → 1.8 s/iter | Needs gcc in the image |
| Switch to flash-attn v3 | 1.4 → 1.6 s/iter | bf16 + sm_100 wheels needed |
| Increase `batch_size` × `seq_len` to fully utilize HBM | 1.2 → 1.4 s/iter | Watch VRAM headroom |
| Add gradient bucketing tuning | 0.95× | Small but free |

Stacked: ~1.5–1.8× speedup is realistic for Phase 2c.

---

### 7. Phase 3 specifically: gradient checkpointing dominates

**Symptom in `torch.profiler`:** backward time is ~2× forward time (without grad-ckpt it'd be ~1.5×). With grad-ckpt, much of the forward gets recomputed in backward.

**Why it's currently on:** to fit comfortably with safety margin. But 12B LoRA on 4× GB200 has ~150 GB headroom per GPU — you don't need grad-ckpt.

**Fix:** in `train.py`, set `gradient_checkpointing=False` and try `per_device_train_batch_size=8` (was 4). VRAM should still stay under 80 GB.

**Expected win:** 1.4–1.6× on Phase 3.

---

### 8. Fractional GPU preloader warnings flooding logs

Already-known from earlier: `LD_PRELOAD` warnings from Run:AI's fractional GPU enforcement. Harmless, doesn't affect perf, but noisy. Mute with:

```bash
export RUNAI_QUOTA_DEBUG=0      # or
unset LD_PRELOAD                # in run.sh if the workload doesn't need fractional GPU quota
```

---

## A 60-minute "speedrun" optimization plan

For maximum learning per unit time, here is the order I'd attack things in:

1. **t = 0**: Run `torch.profiler` on Phase 3 (sub-phase 5a). Look at top 5 ops + GPU idle %.
2. **t = 10**: Apply fix #1 (`dataloader_num_workers=4`) + fix #7 (`gradient_checkpointing=False`, `per_device_train_batch_size=8`). Rerun the 200-step Phase 3. Record new wall-clock.
3. **t = 30**: Run `nsys` on the optimized Phase 3 (5b). Verify NCCL overlap, look for residual gaps.
4. **t = 45**: If gcc available, install build-essential, enable `torch.compile`, rerun. Verify with `torch.profiler` that compiled regions show up as fused kernels.
5. **t = 60**: Final 200-step run, capture the new `train_samples_per_second`. Compare to baseline 15.93.

Stretch goals (out of the hour but worth doing later):
- `ncu` deep-dive on the dominant kernel (likely SDPA forward) — see if FA3 helps.
- Repeat on Phase 2c. The wins there are larger because the baseline is further from optimal.

---

## How to write up the results

For each optimization, record in a small table:

| # | Change | Wall-clock | samples/s | %Δ vs baseline |
|---|---|---|---|---|
| 0 | baseline | 803 s | 15.93 | — |
| 1 | +dataloader_workers=4 | 762 s | 16.79 | +5% |
| 2 | +grad-ckpt off, BS=8 | 540 s | 23.71 | +49% |
| 3 | +torch.compile        | 410 s | 31.23 | +96% |

That's your Phase 5 deliverable. Whatever the final speedup is, you'll have *measured* it and *attributed* it to specific changes.
