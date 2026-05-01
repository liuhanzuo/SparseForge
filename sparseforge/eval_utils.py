"""
Evaluation helper for SparseForge.

This module exposes :func:`make_estimate_loss`, a closure factory that
replaces the legacy ``estimate_loss`` function in both entry-point scripts.

The legacy function pulled ``model``, ``args``, ``ctx``, ``get_batch``,
``master_process`` (and, in the universal variant, ``block_size``) from
enclosing scope.  We rebuild the exact same logic as a closure over
explicit parameters so the two entry points can share a single
implementation.

The ``variant`` switch preserves the two distinct debugging behaviours:
- ``variant='llama'``   : minimal (no FSDP/shape debug, no CUDA sync)
- ``variant='universal'``: prints FSDP/lm_head shape once per rank-0,
  also calls ``torch.cuda.synchronize()`` after eval (required for
  HYBRID_SHARD process groups to finish before the next collective).
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, Dict, Optional

import torch


def make_estimate_loss(
    *,
    model,
    distill_model: bool,
    eval_iters: int,
    get_batch: Callable[[str], tuple],
    ctx: AbstractContextManager,
    master_process: bool = True,
    block_size: Optional[int] = None,
    variant: str = "llama",
) -> Callable[[], Dict[str, float]]:
    """Build an ``estimate_loss()`` callable.

    Parameters
    ----------
    model
        The wrapped training model (DDP/FSDP container). The legacy code
        noted that we must NOT unwrap to ``.module`` under FSDP, otherwise
        parameter views become sharded/flat and forward breaks.
    distill_model
        Mirror of ``args.distill_model``. When ``True`` we evaluate only
        the student submodule (``container.student``) so we do not waste
        compute on teacher forwards.
    eval_iters
        Number of iterations per split (mirrors ``args.eval_iters``).
    get_batch
        Closure from :mod:`sparseforge.data_pipeline`.
    ctx
        Autocast / no-autocast context manager used throughout training.
    master_process
        Rank-0 flag (only used for DEBUG printing in the universal variant).
    block_size
        Sequence length; only used by the universal variant for a one-time
        DEBUG print. Safe to leave ``None`` for the llama variant.
    variant
        ``'llama'`` or ``'universal'``.  Controls whether we emit DEBUG
        prints and whether we call ``torch.cuda.synchronize()`` at the end.
    """
    _universal = variant == "universal"

    @torch.no_grad()
    def estimate_loss() -> Dict[str, float]:
        out: Dict[str, float] = {}
        # IMPORTANT: under FSDP, calling the unwrapped module (`.module`) will see sharded/flattened
        # parameter views (e.g. embedding weights become 1-D), which breaks forward. Always run the
        # wrapped `model` for eval when using FSDP/DDP.
        # For distill training, evaluate student-only (no teacher forward needed).
        if distill_model:
            container = model.module if hasattr(model, "module") else model
            eval_model = container.student

            # (universal only) DEBUG: print model structure info once
            if _universal and master_process and not hasattr(estimate_loss, "_debug_printed"):
                estimate_loss._debug_printed = True  # type: ignore[attr-defined]
                print(f"[DEBUG estimate_loss] container type: {type(container)}")
                print(f"[DEBUG estimate_loss] eval_model type: {type(eval_model)}")
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP_Check
                    print(
                        f"[DEBUG estimate_loss] eval_model is FSDP: "
                        f"{isinstance(eval_model, FSDP_Check)}"
                    )
                except Exception:
                    pass
                try:
                    inner = eval_model.module if hasattr(eval_model, "module") else eval_model
                    if hasattr(inner, "model") and hasattr(inner.model, "lm_head"):
                        lm_head = inner.model.lm_head
                        print(f"[DEBUG estimate_loss] lm_head.weight.shape: {lm_head.weight.shape}")
                        print(f"[DEBUG estimate_loss] lm_head type: {type(lm_head)}")
                except Exception as e:
                    print(f"[DEBUG estimate_loss] Could not get lm_head info: {e}")
        else:
            eval_model = model

        eval_model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)

                # (universal only) DEBUG: print first batch shape once
                if (
                    _universal
                    and k == 0
                    and master_process
                    and not hasattr(estimate_loss, "_batch_debug_printed")
                ):
                    estimate_loss._batch_debug_printed = True  # type: ignore[attr-defined]
                    print(f"[DEBUG estimate_loss] X.shape: {X.shape}, Y.shape: {Y.shape}")
                    if block_size is not None:
                        print(f"[DEBUG estimate_loss] block_size: {block_size}")

                with ctx:
                    logits, loss, _ = eval_model(X, Y)
                losses[k] = loss.item()
                # Explicitly delete large tensors to free memory during eval loop
                del logits, loss, X, Y
            out[split] = losses.mean().item()
            # Clear cache between train/val splits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        eval_model.train()

        if _universal and torch.cuda.is_available():
            # CRITICAL: Synchronize CUDA before returning to ensure all GPU operations are
            # complete.  This is especially important for HYBRID_SHARD mode where
            # mesh_shard process group requires all ranks to complete their operations
            # before starting new collectives.
            torch.cuda.synchronize()

        return out

    return estimate_loss


__all__ = ["make_estimate_loss"]
