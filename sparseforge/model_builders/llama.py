"""Model builder utilities for LLaMA sparse training.

This module documents and exposes the model construction pipeline used by
``main_llama.py``.  The actual build sequence is tightly coupled to the
training script's module-level state (args, DDP/FSDP context, device), so
the heavy lifting remains in ``main_llama.py``.  This module provides:

1. A reference to the core model class (``LlamaSparse``).
2. Helper constants and documentation of the FSDP wrapping strategy.
3. Utility for building the auto-wrap policy used by FSDP.

Public API
----------
- ``LlamaSparse``          : The sparse LLaMA model class (re-exported).
- ``get_fsdp_wrap_policy`` : Build a transformer_auto_wrap_policy for LLaMA layers.
- ``SUPPORTED_FSDP_MODES`` : Valid values for ``--fsdp_mode``.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

# Re-export the core model class for convenient access.
from model_llama import LlamaSparse

__all__ = [
    "LlamaSparse",
    "get_fsdp_wrap_policy",
    "SUPPORTED_FSDP_MODES",
]

# Valid FSDP sharding modes accepted by the training scripts.
SUPPORTED_FSDP_MODES = (
    "fully_sharded",   # FULL_SHARD — shard across ALL ranks globally
    "hybrid_sharded",  # HYBRID_SHARD — shard within node, replicate across nodes
    "shard_grad_op",   # SHARD_GRAD_OP — simpler collectives
    "no_shard",        # NO_SHARD — DDP-like behavior
)


def get_fsdp_wrap_policy(
    model: torch.nn.Module,
    min_num_params: int = 100_000_000,
) -> Optional[Any]:
    """Build an FSDP auto-wrap policy for a LlamaSparse model.

    Attempts ``transformer_auto_wrap_policy`` first (wraps each decoder layer);
    falls back to ``size_based_auto_wrap_policy`` if the layer class cannot be
    detected.

    Args:
        model: A ``LlamaSparse`` instance (or any model with nested layers).
        min_num_params: Minimum parameter count for size-based fallback.

    Returns:
        A callable suitable for FSDP's ``auto_wrap_policy`` kwarg, or ``None``
        if no policy could be constructed.
    """
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError:
        transformer_auto_wrap_policy = None

    # Try transformer-layer-based wrapping first.
    if transformer_auto_wrap_policy is not None:
        try:
            if (
                hasattr(model, "model")
                and hasattr(model.model, "model")
                and hasattr(model.model.model, "layers")
            ):
                block_cls = model.model.model.layers[0].__class__
                return functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={block_cls},
                )
        except Exception:
            pass

    # Fallback: size-based wrapping.
    try:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
    except ImportError:
        return None
