"""Checkpoint save/resume utilities for SparseForge.

This module documents the checkpoint format and provides helper functions for
saving and loading training checkpoints.  The actual checkpoint logic in
``main_llama.py`` / ``main_universal.py`` handles FSDP state-dict contexts
and multi-rank synchronization inline; this module provides a clean reference
implementation and utilities that can be used by external tools.

Checkpoint Format
-----------------
Each checkpoint directory contains:
  - ``model.pt``   : torch.save dict with keys:
      - ``model_state_dict``     : Full model state dict (FSDP FULL_STATE_DICT)
      - ``optimizer_state_dict`` : Optimizer state (optional, for resume)
      - ``scaler_state_dict``    : GradScaler state (fp16 only)
      - ``iter_num``             : Current iteration number
      - ``eval_count``           : Number of evaluations completed
      - ``best_wiki_ppl``        : Best WikiText-2 perplexity
      - ``best_lm_eval_mean``    : Best lm_eval mean accuracy
      - ``args``                 : Training arguments namespace (as dict)
  - ``eval.json``  : Evaluation metrics at save time
  - ``last``       : Symlink to the most recent checkpoint directory

Public API
----------
- ``CHECKPOINT_KEYS``     : Expected keys in a checkpoint dict.
- ``find_latest_ckpt``    : Locate the most recent checkpoint in an output dir.
- ``load_checkpoint_meta``: Load only metadata (iter_num, metrics) without model weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


__all__ = [
    "CHECKPOINT_KEYS",
    "find_latest_ckpt",
    "load_checkpoint_meta",
]

# Expected top-level keys in a SparseForge checkpoint file.
CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "iter_num",
    "eval_count",
    "best_wiki_ppl",
    "best_lm_eval_mean",
    "args",
)


def find_latest_ckpt(out_dir: str) -> Optional[str]:
    """Find the most recent checkpoint directory under ``out_dir``.

    Checks for a ``last`` symlink or ``last_dir.txt`` file, then verifies
    that ``model.pt`` exists within.

    Args:
        out_dir: Root output directory for the training run.

    Returns:
        Absolute path to the checkpoint directory, or ``None`` if not found.
    """
    last_link = os.path.join(out_dir, "last")
    last_dir_file = os.path.join(out_dir, "last_dir.txt")

    ckpt_dir: Optional[str] = None
    if os.path.islink(last_link) or os.path.isdir(last_link):
        ckpt_dir = os.path.realpath(last_link)
    elif os.path.exists(last_dir_file):
        with open(last_dir_file, "r") as f:
            ckpt_dir = f.read().strip()

    if ckpt_dir and os.path.isdir(ckpt_dir):
        model_pt = os.path.join(ckpt_dir, "model.pt")
        if os.path.exists(model_pt):
            return ckpt_dir
    return None


def load_checkpoint_meta(ckpt_dir: str) -> dict[str, Any]:
    """Load only metadata from a checkpoint (no model weights).

    Useful for inspecting training progress without loading the full model.

    Args:
        ckpt_dir: Path to a checkpoint directory containing ``model.pt``.

    Returns:
        Dictionary with metadata keys (iter_num, eval_count, best_wiki_ppl, etc.).

    Raises:
        FileNotFoundError: If ``model.pt`` does not exist in ``ckpt_dir``.
    """
    import torch

    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Checkpoint not found: {model_pt}")

    # Load with weights_only=False to handle arbitrary Python objects in args.
    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)

    meta = {}
    for key in ("iter_num", "eval_count", "best_wiki_ppl", "best_lm_eval_mean", "args"):
        if key in ckpt:
            meta[key] = ckpt[key]

    # Also try to load eval.json if present.
    eval_json = os.path.join(ckpt_dir, "eval.json")
    if os.path.exists(eval_json):
        try:
            with open(eval_json, "r") as f:
                meta["eval_metrics"] = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return meta
