"""Model builder sub-package for SparseForge.

Provides architecture-specific model construction and FSDP wrapping utilities.

Submodules
----------
- ``llama``     : LLaMA-specific model builder (LlamaSparse + FSDP policy).
- ``universal`` : Multi-architecture model builder via model_factory dispatch.
"""

from sparseforge.model_builders.llama import (
    LlamaSparse,
    get_fsdp_wrap_policy as get_llama_fsdp_wrap_policy,
    SUPPORTED_FSDP_MODES,
)
from sparseforge.model_builders.universal import (
    get_sparse_model,
    detect_model_type,
    SUPPORTED_MODEL_TYPES,
    get_fsdp_wrap_policy as get_universal_fsdp_wrap_policy,
)

__all__ = [
    "LlamaSparse",
    "get_llama_fsdp_wrap_policy",
    "get_universal_fsdp_wrap_policy",
    "get_sparse_model",
    "detect_model_type",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_FSDP_MODES",
]
