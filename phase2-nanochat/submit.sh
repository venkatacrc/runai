#!/usr/bin/env bash
# Phase 2: run Karpathy's nanochat speedrun (d20, ~560M) end-to-end on 8x GB200.
# Pipeline = tokenizer training -> base pretrain -> midtrain -> SFT -> RL -> eval.
# Expected wall clock: ~1.5-2 hours on 8x GB200 (~3 h on 8x H100 for reference).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../scripts/env.sh"

NAME="${NAME:-p2-nanochat-$(date +%H%M%S)}"
GPUS="${GPUS:-8}"
DEPTH="${DEPTH:-20}"   # d20 = the ~560M "speedrun" model.
NANOCHAT_REF="${NANOCHAT_REF:-master}"

# If you have a PVC, mount it at /work so checkpoints + HF cache survive the pod.
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
export TORCH_HOME=$WORKROOT/torch
mkdir -p \$HF_HOME \$TORCH_HOME
cd $WORKROOT

if [[ ! -d nanochat ]]; then
  git clone --depth 1 -b $NANOCHAT_REF https://github.com/karpathy/nanochat
fi
cd nanochat

# nanochat uses 'uv'. The NGC pytorch image ships pip but not uv.
pip install --quiet uv
uv sync --extra gpu
source .venv/bin/activate
python -c 'import torch; print("venv torch", torch.__version__)'

# The reference speedrun trains the d20 (~560M) model end to end.
# If you want to override depth or shorten things, edit runs/speedrun.sh or
# call scripts/base_train.py directly.
echo "=== starting nanochat speedrun (depth=$DEPTH) ==="
time bash runs/speedrun.sh
echo "=== speedrun finished ==="
ls -la
EOF

echo ">>> submitting $NAME (gpus=$GPUS, depth=$DEPTH, image=$RUNAI_IMAGE)"
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

cat <<EOF

Submitted. Useful follow-ups:
  runai workload list
  runai workload logs $NAME -f
  runai workload describe $NAME

When done (and you persisted to a PVC), grab the chat checkpoint from
  \$RUNAI_PVC:/work/nanochat/checkpoints/
EOF
