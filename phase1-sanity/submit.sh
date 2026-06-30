#!/usr/bin/env bash
# Phase 1: NCCL all-reduce micro-benchmark + DDP smoke test on 8x GB200.
# Run from repo root:   bash phase1-sanity/submit.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../scripts/env.sh"

NAME="${NAME:-p1-sanity-$(date +%H%M%S)}"
GPUS="${GPUS:-8}"

# Encode the two Python scripts so we don't need a PVC or a custom image.
ALLREDUCE_B64="$(base64 < "$HERE/allreduce_check.py" | tr -d '\n')"
DDP_B64="$(base64 < "$HERE/ddp_smoke.py" | tr -d '\n')"

read -r -d '' INNER <<EOF || true
set -euo pipefail
echo "=== Hostname / GPUs ==="
hostname
nvidia-smi -L
echo "=== Python / Torch ==="
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())'
mkdir -p /tmp/p1 && cd /tmp/p1
echo "$ALLREDUCE_B64" | base64 -d > allreduce_check.py
echo "$DDP_B64"        | base64 -d > ddp_smoke.py
echo "=== NCCL all-reduce ==="
torchrun --standalone --nproc_per_node=$GPUS allreduce_check.py
echo "=== DDP smoke ==="
torchrun --standalone --nproc_per_node=$GPUS ddp_smoke.py
echo "=== DONE ==="
EOF

echo ">>> submitting $NAME (gpus=$GPUS, image=$RUNAI_IMAGE)"
runai training submit "$NAME" \
  --image "$RUNAI_IMAGE" \
  ${RUNAI_PROJECT:+--project "$RUNAI_PROJECT"} \
  ${RUNAI_NODE_POOL:+--node-pools "$RUNAI_NODE_POOL"} \
  --gpu-devices-request "$GPUS" \
  --large-shm \
  --command -- bash -lc "$INNER"

echo
echo "Tail logs with:"
echo "  runai workload logs $NAME -f"
echo "Clean up when done:"
echo "  runai workload delete $NAME"
