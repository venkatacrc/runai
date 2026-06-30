#!/usr/bin/env bash
# Phase 5 — Nsight Systems profile of a real Phase 3 (or Phase 2c) training run.
#
# Captures the full system timeline: CUDA streams, NCCL, OS scheduler, Python.
# Output: one .nsys-rep file you scp to your laptop and open in Nsight Systems GUI.
#
# Run inside the Training-workload pod (after env preflight in run.sh):
#
#   PROFILE_TARGET=phase3  bash /data/training/runai-repo/phase5-profile/run_nsys.sh
#   PROFILE_TARGET=phase2c bash /data/training/runai-repo/phase5-profile/run_nsys.sh
#   PROFILE_TARGET=bench   bash /data/training/runai-repo/phase5-profile/run_nsys.sh
#
# Defaults: 4 GPUs, 20 short steps. Trace file: /data/training/p5-profile/<target>-<ts>.nsys-rep

set -euo pipefail

TARGET="${PROFILE_TARGET:-phase3}"
GPUS="${GPUS:-4}"
OUT_DIR="${OUT_DIR:-/data/training/p5-profile}"
TS="$(date +%Y%m%d-%H%M%S)"
REP_BASE="${OUT_DIR}/${TARGET}-${TS}"

mkdir -p "$OUT_DIR"

command -v nsys >/dev/null || {
  echo "ERROR: nsys not on PATH. See phase5-profile/notes.md 'Install if missing'." >&2
  exit 1
}

# Limit trace file size: stop capturing after 60 s, even if the job runs longer.
NSYS_FLAGS=(
  --output "$REP_BASE"
  --force-overwrite=true
  --export=none
  --trace=cuda,nvtx,osrt,cudnn,cublas
  --cuda-memory-usage=true
  --python-sampling=true
  --python-sampling-frequency=1000
  --sample=cpu
  --backtrace=fp
  --duration=60                # max 60 s of trace; plenty for ~15 steps
  --delay=10                   # skip first 10 s (warmup, model load)
  --kill=sigint
)

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

case "$TARGET" in

  phase3)
    cd /data/training/p3-gemma
    export HF_HOME=/data/hf-cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export OUTPUT_DIR=/data/training/p3-gemma/out
    export MAX_STEPS=20
    export PER_DEVICE_BS=4
    export GRAD_ACCUM=4
    export LR=2e-4
    export LORA_R=16
    export MAX_SEQ_LEN=1024
    nsys profile "${NSYS_FLAGS[@]}" \
      torchrun --standalone --nproc_per_node="$GPUS" train.py
    ;;

  phase2c)
    cd /data/training/gpt/nanoGPT
    export TIKTOKEN_CACHE_DIR=/data/tiktoken-cache
    nsys profile "${NSYS_FLAGS[@]}" \
      torchrun --standalone --nproc_per_node="$GPUS" \
        train.py config/train_gpt2.py \
        --max_iters=30 --eval_interval=1000 \
        --gradient_accumulation_steps=4 \
        --out_dir=/tmp/p5-nsys-p2c
    ;;

  bench)
    HERE="$(cd "$(dirname "$0")" && pwd)"
    nsys profile "${NSYS_FLAGS[@]}" \
      torchrun --standalone --nproc_per_node="$GPUS" "$HERE/profile_step.py"
    ;;

  *)
    echo "Unknown PROFILE_TARGET=$TARGET (use: phase3 | phase2c | bench)" >&2
    exit 2
    ;;

esac

echo
echo "=== nsys capture complete ==="
ls -lh "${REP_BASE}.nsys-rep" 2>/dev/null || {
  echo "WARNING: expected ${REP_BASE}.nsys-rep not found — check nsys stderr above"
  exit 3
}
echo
echo "To analyze:"
echo "  1) Copy to your laptop:"
echo "       runai workspace exec <workspace> -- cat ${REP_BASE}.nsys-rep > ~/Downloads/$(basename "$REP_BASE").nsys-rep"
echo "     or (from your laptop, if you have kubectl):"
echo "       kubectl -n runai-<your-project> cp <pod-name>:${REP_BASE}.nsys-rep ~/Downloads/"
echo "  2) Open in Nsight Systems GUI (download from https://developer.nvidia.com/nsight-systems)."
