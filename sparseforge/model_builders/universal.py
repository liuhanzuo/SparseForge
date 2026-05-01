"""Model builder utilities for the universal (multi-architecture) sparse trainer.

This module documents and exposes the model construction pipeline used by
``main_universal.py``.  It supports multiple architectures (LLaMA, OPT, GPT-2,
Qwen, Mistral, DeepSeek-MoE, Hunyuan) via the ``model_factory`` dispatch layer.

The actual build sequence is tightly coupled to the training script's
module-level state (args, DDP/FSDP context, device), so the heavy lifting
remains in ``main_universal.py``.  This module provides:

1. Re-exports from ``model_factory`` for convenient programmatic access.
2. A reference list of supported architectures.
3. Helper for building FSDP auto-wrap policies across different model families.

Public API
----------
- ``get_sparse_model``     : Factory function that returns the correct sparse model.
- ``detect_model_type``    : Auto-detect architecture from model name/path.
- ``SUPPORTED_MODEL_TYPES``: Tuple of supported architecture strings.
- ``get_fsdp_wrap_policy`` : Build a transformer_auto_wrap_policy for any supported model.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

# Re-export the model factory for convenient access.
from model_factory import get_sparse_model, detect_model_type, SUPPORTED_MODEL_TYPES

__all__ = [
    "get_sparse_model",
    "detect_model_type",
    "SUPPORTED_MODEL_TYPES",
    "get_fsdp_wrap_policy",
]


def get_fsdp_wrap_policy(
    model: torch.nn.Module,
    min_num_params: int = 100_000_000,
) -> Optional[Any]:
    """Build an FSDP auto-wrap policy for any supported sparse model.

    Attempts ``transformer_auto_wrap_policy`` first by detecting the decoder
    layer class; falls back to ``size_based_auto_wrap_policy``.

    Args:
        model: A sparse model instance (LlamaSparse, OPTSparse, QwenSparse, etc.).
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
        block_cls = _detect_layer_class(model)
        if block_cls is not None:
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={block_cls},
            )

    # Fallback: size-based wrapping.
    try:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
    except ImportError:
        return None


def _detect_layer_class(model: torch.nn.Module) -> Optional[type]:
    """Heuristically detect the transformer decoder layer class.

    Walks common nesting patterns used by HuggingFace model wrappers:
      model.model.model.layers[0]  (LLaMA/Qwen/Mistral)
      model.model.decoder.layers[0]  (OPT)
      model.transformer.h[0]  (GPT-2)
    """
    candidates = [
        lambda m: m.model.model.layers[0],
        lambda m: m.model.decoder.layers[0],
        lambda m: m.transformer.h[0],
    ]
    for accessor in candidates:
        try:
            layer = accessor(model)
            return layer.__class__
        except (AttributeError, IndexError, TypeError):
            continue
    return None
