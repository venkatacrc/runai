# Source this file in every shell you use for Run:AI work.
# Fill in the blanks after Phase 0.
#
#   source scripts/env.sh
#
# All values are placeholders. Override either by editing this file or by
# `export RUNAI_URL=... ; source scripts/env.sh`.

# Run:AI control-plane / portal URL.
# Example used while writing this lab: https://runai.prod.walmart.com
# Replace with your own org's Run:AI URL.
export RUNAI_URL="${RUNAI_URL:-https://<your-runai-portal>}"

# Run:AI project (namespace) that owns your GPU quota.
# Example used while writing this lab: ri-fm
export RUNAI_PROJECT="${RUNAI_PROJECT:-}"

# Run:AI cluster name (only matters if you have more than one).
export RUNAI_CLUSTER="${RUNAI_CLUSTER:-}"

# Node pool that contains the GB200 nodes. Leave empty to use project default.
export RUNAI_NODE_POOL="${RUNAI_NODE_POOL:-}"

# Container image used by every phase. Blackwell needs CUDA >= 12.4.
# Swap the registry prefix if your cluster mirrors NGC internally.
export RUNAI_IMAGE="${RUNAI_IMAGE:-nvcr.io/nvidia/pytorch:25.04-py3}"

# Persistent volume claim name for checkpoints/datasets (optional).
# Discover with: kubectl -n <ns> get pvc
export RUNAI_PVC="${RUNAI_PVC:-}"

# HuggingFace token for gated models and dataset downloads (Phases 2-4).
# Get one at https://huggingface.co/settings/tokens
export HF_TOKEN="${HF_TOKEN:-}"

# WandB key (optional, nanochat and trl write metrics here if set).
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-runai-learn}"

# Helper: prints a one-line summary of the current env.
runai_env_summary() {
  echo "project=${RUNAI_PROJECT:-<unset>}  cluster=${RUNAI_CLUSTER:-<unset>}  pool=${RUNAI_NODE_POOL:-<default>}"
  echo "image=${RUNAI_IMAGE}"
  echo "pvc=${RUNAI_PVC:-<none>}  hf_token=${HF_TOKEN:+set}  wandb=${WANDB_API_KEY:+set}"
}
