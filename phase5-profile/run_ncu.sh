#!/usr/bin/env bash
# Phase 5 — Nsight Compute (ncu) kernel-level capture.
#
# IMPORTANT: ncu profiles ONE process at a time and runs each kernel many times
# under heavy instrumentation. It is NOT suited to multi-GPU training. We
# capture single-GPU on rank 0 of a tiny benchmark or on a single-GPU shim of
# the real training step.
#
# Run inside the Training-workload pod (after env preflight in run.sh):
#
#   KERNEL_FILTER="aten::scaled_dot_product_attention" \
#     bash /data/training/runai-repo/phase5-profile/run_ncu.sh
#
# Defaults: profile the first 5 invocations of the top-3 matmul/attention kernels
# inside the standalone profile_step.py benchmark on 1 GPU.
#
# Output: /data/training/p5-profile/ncu-<timestamp>.ncu-rep
# Then scp to your laptop and open in Nsight Compute GUI.

set -euo pipefail

TARGET="${PROFILE_TARGET:-bench}"           # bench | phase3-1gpu
OUT_DIR="${OUT_DIR:-/data/training/p5-profile}"
TS="$(date +%Y%m%d-%H%M%S)"
REP="${OUT_DIR}/ncu-${TARGET}-${TS}.ncu-rep"

mkdir -p "$OUT_DIR"

command -v ncu >/dev/null || {
  echo "ERROR: ncu not on PATH. See phase5-profile/notes.md 'Install if missing'." >&2
  exit 1
}

# Skip warmup kernels; capture only the most informative ones.
NCU_FLAGS=(
  -o "${REP%.ncu-rep}"           # ncu appends .ncu-rep
  --force-overwrite
  --set full                     # full set of metrics; ~20x kernel slowdown but fine on 5 kernels
  --target-processes all
  --launch-skip "${LAUNCH_SKIP:-200}"   # skip first 200 kernels (warmup + setup)
  --launch-count "${LAUNCH_COUNT:-20}"  # capture 20 kernels total
  --replay-mode kernel           # safe for stateless GPU kernels; faster than 'application'
)

# Optional: only profile kernels whose name matches a regex.
# e.g. KERNEL_FILTER="gemm|attention|softmax"
if [[ -n "${KERNEL_FILTER:-}" ]]; then
  NCU_FLAGS+=(--kernel-name "regex:${KERNEL_FILTER}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$TARGET" in

  bench)
    HERE="$(cd "$(dirname "$0")" && pwd)"
    # Single GPU, no DDP.
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      ncu "${NCU_FLAGS[@]}" python "$HERE/profile_step.py"
    ;;

  phase3-1gpu)
    cd /data/training/p3-gemma
    export HF_HOME=/data/hf-cache HF_HUB_OFFLINE=1
    export MAX_STEPS=5 PER_DEVICE_BS=2 GRAD_ACCUM=1 MAX_SEQ_LEN=1024
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      ncu "${NCU_FLAGS[@]}" python train.py
    ;;

  *)
    echo "Unknown PROFILE_TARGET=$TARGET (use: bench | phase3-1gpu)" >&2
    exit 2
    ;;

esac

echo
echo "=== ncu capture complete ==="
ls -lh "$REP" 2>/dev/null || {
  echo "WARNING: expected $REP not found — check ncu stderr above"
  exit 3
}
echo
echo "To analyze:"
echo "  1) Copy to your laptop:"
echo "       kubectl -n runai-<your-project> cp <pod>:$REP ~/Downloads/"
echo "  2) Open in Nsight Compute GUI (download from https://developer.nvidia.com/nsight-compute)."
echo "     Or summary on the pod:    ncu --import $REP --page details | less"
