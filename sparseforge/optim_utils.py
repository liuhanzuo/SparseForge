"""
Optimizer / scheduler helpers for SparseForge.

Pure functions (no globals) that replicate the legacy ``get_lr`` and
``get_decay`` closures in ``main_llama_legacy.py`` and
``main_universal_legacy.py``.

The original code captured ``warmup_iters`` / ``lr_decay_iters`` /
``learning_rate`` / ``min_lr`` / ``args.increase_step`` / ``args.srste_decay``
from the enclosing script scope.  Here we make these explicit parameters
(or bake them into a closure via ``make_lr_schedule`` / ``make_decay_schedule``).
"""
from __future__ import annotations

import math
from typing import Callable


# ---------------------------------------------------------------------------
# Learning-rate: linear warmup + cosine decay to ``min_lr``
# ---------------------------------------------------------------------------
def get_lr(
    it: int,
    *,
    warmup_iters: int,
    lr_decay_iters: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """Cosine LR schedule with linear warmup (bit-identical to legacy)."""
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    decay_ratio = min(1.0, max(0.0, float(decay_ratio)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def make_lr_schedule(
    *,
    warmup_iters: int,
    lr_decay_iters: int,
    learning_rate: float,
    min_lr: float,
) -> Callable[[int], float]:
    """Build a closure with the schedule hyper-params already bound."""

    def _lr(it: int) -> float:
        return get_lr(
            it,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
            learning_rate=learning_rate,
            min_lr=min_lr,
        )

    return _lr


# ---------------------------------------------------------------------------
# SR-STE decay ramp: 0 → srste_decay across ``increase_step`` steps
# ---------------------------------------------------------------------------
def get_decay(it: int, *, increase_step: int, srste_decay: float) -> float:
    """Linear ramp for the SR-STE sparsity penalty coefficient."""
    inc = int(increase_step)
    dmax = float(srste_decay)
    if dmax <= 0.0:
        return 0.0
    if it < inc:
        return dmax / max(1, inc) * it
    return dmax


def make_decay_schedule(
    *, increase_step: int, srste_decay: float
) -> Callable[[int], float]:
    """Build a closure with the decay hyper-params already bound."""

    def _decay(it: int) -> float:
        return get_decay(it, increase_step=increase_step, srste_decay=srste_decay)

    return _decay


__all__ = [
    "get_lr",
    "make_lr_schedule",
    "get_decay",
    "make_decay_schedule",
]
