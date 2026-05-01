"""Training state management for SparseForge.

This module defines the ``TrainState`` dataclass that encapsulates all mutable
training state threaded across stages in the training loop.  It serves as the
canonical reference for what constitutes "training state" in SparseForge.

Currently, ``main_llama.py`` and ``main_universal.py`` manage these fields as
module-level variables.  This dataclass documents the contract and can be used
for future refactoring to pass state explicitly.

Public API
----------
- ``TrainState`` : Dataclass holding all mutable training loop state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


__all__ = ["TrainState"]


@dataclass
class TrainState:
    """Encapsulates all mutable state for a SparseForge training run.

    This dataclass documents the full set of state variables that are:
    - Initialized before the training loop
    - Mutated during training
    - Saved/restored from checkpoints

    Attributes:
        iter_num: Current training iteration (0-indexed).
        best_val_loss: Best validation loss seen so far.
        best_wiki_ppl: Best WikiText-2 perplexity seen so far.
        best_lm_eval_mean: Best mean accuracy from lm_eval benchmarks.
        eval_count: Number of evaluation rounds completed.
        mask_update_count: Number of mask update steps performed.
        hardening_active: Whether mask hardening is currently active.
        finalization_active: Whether the final fine-tuning stage is active.
        current_lr: Current learning rate.
        current_sparsity: Current measured sparsity ratio.
        run_id: Unique identifier for this training run.
        out_dir: Output directory for checkpoints and logs.
    """

    iter_num: int = 0
    best_val_loss: Optional[float] = None
    best_wiki_ppl: float = 1e9
    best_lm_eval_mean: float = 0.0
    eval_count: int = 0
    mask_update_count: int = 0
    hardening_active: bool = False
    finalization_active: bool = False
    current_lr: float = 0.0
    current_sparsity: float = 0.0
    run_id: str = ""
    out_dir: str = ""

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize state fields that should be saved in a checkpoint.

        Returns:
            Dictionary of state fields suitable for ``torch.save``.
        """
        return {
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "best_wiki_ppl": self.best_wiki_ppl,
            "best_lm_eval_mean": self.best_lm_eval_mean,
            "eval_count": self.eval_count,
        }

    @classmethod
    def from_checkpoint_dict(cls, d: dict[str, Any]) -> "TrainState":
        """Restore state from a checkpoint dictionary.

        Args:
            d: Dictionary loaded from a checkpoint file.

        Returns:
            A new ``TrainState`` instance with restored fields.
        """
        return cls(
            iter_num=int(d.get("iter_num", 0)),
            best_val_loss=d.get("best_val_loss"),
            best_wiki_ppl=float(d.get("best_wiki_ppl", 1e9)),
            best_lm_eval_mean=float(d.get("best_lm_eval_mean", 0.0)),
            eval_count=int(d.get("eval_count", 0)),
        )
