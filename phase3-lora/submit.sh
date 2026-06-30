#!/usr/bin/env bash
# Phase 3: LoRA fine-tune Gemma 4 12B-it on Alpaca with TRL.
# Default: 4 GPUs (matches Phase 2c setup). Bump GPUS=8 if free.
#
# Gemma 4 is gated — export HF_TOKEN before running (or set it in scripts/env.sh)
# and accept the license at https://huggingface.co/google/gemma-4-12B-it
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../scripts/env.sh"

NAME="${NAME:-p3-gemma4-12b-$(date +%H%M%S)}"
GPUS="${GPUS:-4}"
MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Gemma 4 is a gated repo — download will fail" >&2
  echo "         unless the workload pod already has cached weights on the PVC." >&2
fi

TRAIN_B64="$(base64 < "$HERE/train.py" | tr -d '\n')"

PVC_ARGS=()
WORKROOT="/tmp/work"
if [[ -n "${RUNAI_PVC:-}" ]]; then
  PVC_ARGS=(--existing-pvc "claimname=${RUNAI_PVC},path=/work")
  WORKROOT="/work"
fi

read -r -d '' INNER <<EOF || true
set -euo pipefail
nvidia-smi -L
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'

export HOME=$WORKROOT
export HF_HOME=$WORKROOT/hf
mkdir -p \$HF_HOME
cd $WORKROOT

pip install --quiet --upgrade \
  'transformers>=4.50' \
  'trl>=0.11' 'peft>=0.13' 'datasets>=3.0' 'accelerate>=1.0' \
  'bitsandbytes>=0.43' wandb

echo "$TRAIN_B64" | base64 -d > train.py

export MODEL_ID="$MODEL_ID"
export OUTPUT_DIR=$WORKROOT/p3-out
mkdir -p \$OUTPUT_DIR

if [[ "$GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node=$GPUS train.py
else
  python train.py
fi
EOF

echo ">>> submitting $NAME (gpus=$GPUS, model=$MODEL_ID)"
runai training submit "$NAME" \
  --image "$RUNAI_IMAGE" \
  ${RUNAI_PROJECT:+--project "$RUNAI_PROJECT"} \
  ${RUNAI_NODE_POOL:+--node-pools "$RUNAI_NODE_POOL"} \
  --gpu-devices-request "$GPUS" \
  --large-shm \
  --backoff-limit 0 \
  ${HF_TOKEN:+--environment "HF_TOKEN=$HF_TOKEN"} \
  ${WANDB_API_KEY:+--environment "WANDB_API_KEY=$WANDB_API_KEY"} \
  ${WANDB_PROJECT:+--environment "WANDB_PROJECT=$WANDB_PROJECT"} \
  "${PVC_ARGS[@]}" \
  --command -- bash -lc "$INNER"

echo
echo "Tail with: runai workload logs $NAME -f"
