"""Phase 5 — wrap Phase 3's TRL SFTTrainer with torch.profiler.

This is a transparent shim: import this module BEFORE the real Phase 3 train.py
runs, and a torch.profiler context will be attached to the trainer's training
loop via a TrainerCallback. Steps 5..9 (after warmup) are captured to
$OUT_DIR/trace_torch_phase3/ as tensorboard event files + a Chrome trace JSON.

Usage in your run.sh on the workload pod:

    cd /data/training/p3-gemma
    export MAX_STEPS=20          # short run, profiling overhead is ~30% per traced step
    export PROFILE_OUT=/data/training/p3-gemma/trace_torch_phase3
    mkdir -p "$PROFILE_OUT"
    PYTHONPATH=/data/training/runai-repo/phase5-profile:$PYTHONPATH \
      PYTHONSTARTUP=/data/training/runai-repo/phase5-profile/profile_phase3.py \
      torchrun --standalone --nproc_per_node=4 train.py

The PYTHONSTARTUP trick injects our callback registration before train.py runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import TrainerCallback


class TorchProfilerCallback(TrainerCallback):
    """Attaches torch.profiler around a small window of training steps."""

    def __init__(
        self,
        out_dir: str,
        warmup: int = 5,
        active: int = 5,
    ) -> None:
        self.out_dir = out_dir
        self.warmup = warmup
        self.active = active
        self.profiler: profile | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.out_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self.profiler.__enter__()
        print(f"[profile] torch.profiler attached, output -> {self.out_dir}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.profiler is not None:
            self.profiler.step()
        if state.global_step >= (self.warmup + self.active + 1) and self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            print("[profile] traced window finished; tear-down done", flush=True)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None


def _inject() -> None:
    """Monkey-patch SFTTrainer so the callback is added on construction."""
    try:
        from trl import SFTTrainer
    except ImportError:
        return

    orig_init = SFTTrainer.__init__

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        cb = TorchProfilerCallback(
            out_dir=os.environ.get("PROFILE_OUT", "./trace_torch_phase3"),
            warmup=int(os.environ.get("PROFILE_WARMUP", "5")),
            active=int(os.environ.get("PROFILE_ACTIVE", "5")),
        )
        callbacks.append(cb)
        kwargs["callbacks"] = callbacks
        return orig_init(self, *args, **kwargs)

    SFTTrainer.__init__ = patched_init
    print("[profile] SFTTrainer patched to attach torch.profiler", flush=True)


_inject()
