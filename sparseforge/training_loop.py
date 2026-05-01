"""Training loop documentation and stage definitions for SparseForge.

This module documents the multi-stage training loop architecture used by
``main_llama.py`` and ``main_universal.py``.  The actual training loop remains
in those scripts (deeply coupled to module-level state), but this module
provides:

1. An enumeration of training stages for programmatic reference.
2. Documentation of the loop's control flow and stage transitions.
3. Helper predicates for stage detection (useful for logging/callbacks).

Training Loop Architecture
--------------------------
The SparseForge training loop consists of the following stages:

1. **Warm-up** (iter 0 → warmup_iters):
   - Linear LR warm-up from 0 to ``learning_rate``.
   - Sparsity penalty warm-up (``sparsity_warmup_steps``).
   - Mask updates at ``mask_update_period_before`` interval.

2. **Main Training** (warmup_iters → mask_hardening_start):
   - Cosine LR decay.
   - Mask updates switch to ``mask_update_period_after`` at ``mask_update_switch_step``.
   - Hutchinson Hessian estimation for importance scoring.
   - Sparsity penalties (block-level Lagrangian multiplier).

3. **Hardening** (mask_hardening_start → mask_hardening_start + mask_hardening_duration):
   - Progressive mask freezing toward N:M structure.
   - Temperature annealing (``temp_decay``).
   - Sparsity penalty LR scheduling.

4. **Final Fine-tuning** (last ``final_finetune_iters`` iterations):
   - Mask fully frozen (binary).
   - Pure fine-tuning of remaining weights.
   - Best checkpoint selection by WikiText-2 PPL.

Public API
----------
- ``TrainingStage``       : Enum of training stages.
- ``detect_stage``        : Determine current stage from iteration number and args.
- ``STAGE_DESCRIPTIONS``  : Human-readable descriptions of each stage.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any


__all__ = [
    "TrainingStage",
    "detect_stage",
    "STAGE_DESCRIPTIONS",
]


class TrainingStage(Enum):
    """Enumeration of SparseForge training stages."""

    WARMUP = auto()
    MAIN_TRAINING = auto()
    HARDENING = auto()
    FINAL_FINETUNE = auto()


STAGE_DESCRIPTIONS: dict[TrainingStage, str] = {
    TrainingStage.WARMUP: (
        "Linear LR warm-up with sparsity penalty ramp-up. "
        "Mask updates at coarse interval (mask_update_period_before)."
    ),
    TrainingStage.MAIN_TRAINING: (
        "Cosine LR decay with Hutchinson Hessian-guided mask updates. "
        "Block-level sparsity penalties enforce target ratio."
    ),
    TrainingStage.HARDENING: (
        "Progressive mask freezing toward N:M structure. "
        "Temperature annealing drives soft masks to binary."
    ),
    TrainingStage.FINAL_FINETUNE: (
        "Mask fully frozen (binary). Pure weight fine-tuning. "
        "Best checkpoint selected by WikiText-2 PPL."
    ),
}


def detect_stage(
    iter_num: int,
    *,
    max_iters: int,
    warmup_iters: int = 1000,
    mask_hardening_start: int = 2000,
    mask_hardening_duration: int = 15000,
    final_finetune_iters: int = 3000,
) -> TrainingStage:
    """Determine the current training stage from iteration number.

    Args:
        iter_num: Current training iteration.
        max_iters: Total number of training iterations.
        warmup_iters: Number of warm-up iterations.
        mask_hardening_start: Iteration at which hardening begins.
        mask_hardening_duration: Duration of the hardening phase.
        final_finetune_iters: Number of final fine-tuning iterations.

    Returns:
        The ``TrainingStage`` corresponding to the current iteration.
    """
    finetune_start = max_iters - final_finetune_iters

    if iter_num < warmup_iters:
        return TrainingStage.WARMUP
    elif iter_num >= finetune_start:
        return TrainingStage.FINAL_FINETUNE
    elif iter_num >= mask_hardening_start:
        return TrainingStage.HARDENING
    else:
        return TrainingStage.MAIN_TRAINING
