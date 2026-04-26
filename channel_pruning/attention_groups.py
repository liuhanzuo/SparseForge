"""
Attention Groups: Define and extract attention head groups from Transformer layers.

For each Transformer layer's attention, a head group consists of:
- q_proj.weight[I_h, :] (output rows for head h)
- k_proj.weight[I_kv, :] (output rows for KV head, shared in GQA)
- v_proj.weight[I_kv, :] (output rows for KV head, shared in GQA)
- o_proj.weight[:, I_h] (input columns for head h)

Where I_h = [h * head_dim, (h+1) * head_dim) is the slice for query head h.
For GQA, KV heads are shared: I_kv = [h_kv * head_dim, (h_kv+1) * head_dim)
where h_kv = h // group_size.

These must be pruned together to maintain structural consistency.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn


class AttentionProjections(NamedTuple):
    """Container for attention projection layers."""
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    layer_idx: int


@dataclass
class AttentionConfig:
    """Attention configuration extracted from model."""
    hidden_size: int      # D: model dimension
    num_heads: int        # n_heads: number of query heads
    num_kv_heads: int     # n_kv: number of KV heads (for GQA)
    head_dim: int         # d_head = hidden_size // num_heads
    
    @property
    def group_size(self) -> int:
        """Number of query heads per KV head (GQA group size)."""
        return self.num_heads // self.num_kv_heads
    
    @property
    def is_gqa(self) -> bool:
        """Whether this is Grouped Query Attention."""
        return self.num_kv_heads < self.num_heads
    
    @property
    def is_mqa(self) -> bool:
        """Whether this is Multi-Query Attention (single KV head)."""
        return self.num_kv_heads == 1
    
    def get_query_head_slice(self, head_idx: int) -> Tuple[int, int]:
        """Get the slice indices for a query head.
        
        Args:
            head_idx: Query head index [0, num_heads)
            
        Returns:
            (start, end) indices for this head in q_proj/o_proj
        """
        start = head_idx * self.head_dim
        end = (head_idx + 1) * self.head_dim
        return start, end
    
    def get_kv_head_for_query(self, query_head_idx: int) -> int:
        """Get the KV head index for a query head (GQA mapping).
        
        Args:
            query_head_idx: Query head index [0, num_heads)
            
        Returns:
            Corresponding KV head index [0, num_kv_heads)
        """
        return query_head_idx // self.group_size
    
    def get_kv_head_slice(self, kv_head_idx: int) -> Tuple[int, int]:
        """Get the slice indices for a KV head.
        
        Args:
            kv_head_idx: KV head index [0, num_kv_heads)
            
        Returns:
            (start, end) indices for this KV head in k_proj/v_proj
        """
        start = kv_head_idx * self.head_dim
        end = (kv_head_idx + 1) * self.head_dim
        return start, end
    
    def get_query_heads_for_kv(self, kv_head_idx: int) -> List[int]:
        """Get all query head indices that share a KV head.
        
        Args:
            kv_head_idx: KV head index [0, num_kv_heads)
            
        Returns:
            List of query head indices
        """
        start = kv_head_idx * self.group_size
        end = (kv_head_idx + 1) * self.group_size
        return list(range(start, end))


@dataclass
class AttentionHeadGroup:
    """Represents a single head group in an attention layer.
    
    A head group contains weight slices that must be pruned together:
    - q_proj: rows I_h (output channels for this head)
    - k_proj: rows I_kv (output channels for corresponding KV head)
    - v_proj: rows I_kv (output channels for corresponding KV head)
    - o_proj: columns I_h (input channels for this head)
    
    For GQA, multiple query heads share the same KV head, so KV scores
    are distributed across the sharing query heads.
    """
    layer_idx: int
    head_idx: int         # Query head index
    kv_head_idx: int      # Corresponding KV head index
    attn_config: AttentionConfig
    
    def get_q_slice(self) -> Tuple[int, int]:
        """Get slice for q_proj.weight rows."""
        return self.attn_config.get_query_head_slice(self.head_idx)
    
    def get_kv_slice(self) -> Tuple[int, int]:
        """Get slice for k_proj/v_proj.weight rows."""
        return self.attn_config.get_kv_head_slice(self.kv_head_idx)
    
    def get_o_slice(self) -> Tuple[int, int]:
        """Get slice for o_proj.weight columns."""
        return self.attn_config.get_query_head_slice(self.head_idx)


def get_attention_config(model: nn.Module, model_type: str = "qwen") -> AttentionConfig:
    """Extract attention configuration from model.
    
    Args:
        model: The model
        model_type: Type of model architecture
        
    Returns:
        AttentionConfig with head dimensions
    """
    # Get config
    config = None
    if hasattr(model, 'config'):
        config = model.config
    elif hasattr(model, 'model') and hasattr(model.model, 'config'):
        config = model.model.config
    
    if config is None:
        raise ValueError("Cannot find model config")
    
    # Extract attention parameters
    hidden_size = getattr(config, 'hidden_size', None)
    num_heads = getattr(config, 'num_attention_heads', None)
    
    # GQA parameters (different naming conventions)
    num_kv_heads = getattr(config, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, 'num_kv_heads', None)
    if num_kv_heads is None:
        # Default to MHA (same as num_heads)
        num_kv_heads = num_heads
    
    # Head dimension
    head_dim = getattr(config, 'head_dim', None)
    if head_dim is None:
        head_dim = hidden_size // num_heads
    
    return AttentionConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )


def get_attention_projections(model: nn.Module, model_type: str = "qwen") -> List[AttentionProjections]:
    """Extract attention projection layers from a model.
    
    Supports different model architectures:
    - qwen/llama/mistral: model.model.layers[i].self_attn.{q_proj, k_proj, v_proj, o_proj}
    - opt: model.model.decoder.layers[i].self_attn.{q_proj, k_proj, v_proj, out_proj}
    
    Args:
        model: The model to extract projections from
        model_type: Type of model architecture
        
    Returns:
        List of AttentionProjections for each layer
    """
    projections = []
    
    # Unwrap FSDP / DDP / Module wrappers recursively
    def _unwrap(m):
        if hasattr(m, '_fsdp_wrapped_module'):
            return _unwrap(m._fsdp_wrapped_module)
        if hasattr(m, 'module'):
            return _unwrap(m.module)
        return m
    
    base_model = _unwrap(model)
    
    # Further unwrap model-specific wrappers
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
        base_model = base_model.model.model
    elif hasattr(base_model, 'model'):
        base_model = base_model.model
    
    # Find layers
    if model_type in ["qwen", "qwen2", "qwen3", "llama", "mistral"]:
        if hasattr(base_model, 'layers'):
            layers = base_model.layers
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        else:
            raise ValueError(f"Cannot find layers in model for type {model_type}")
        
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            
            # Standard naming
            if hasattr(attn, 'q_proj'):
                q_proj = attn.q_proj
                k_proj = attn.k_proj
                v_proj = attn.v_proj
                o_proj = attn.o_proj
            # Alternative naming (some implementations)
            elif hasattr(attn, 'qkv_proj'):
                # Fused QKV - need special handling
                raise NotImplementedError("Fused QKV projection not yet supported")
            else:
                raise ValueError(f"Unknown attention structure in layer {i}")
            
            projections.append(AttentionProjections(
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                o_proj=o_proj,
                layer_idx=i
            ))
    
    elif model_type == "opt":
        if hasattr(base_model, 'decoder'):
            layers = base_model.decoder.layers
        else:
            layers = base_model.layers
        
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            
            projections.append(AttentionProjections(
                q_proj=attn.q_proj,
                k_proj=attn.k_proj,
                v_proj=attn.v_proj,
                o_proj=attn.out_proj,
                layer_idx=i
            ))
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return projections


class AttentionHeadGroupManager:
    """Manager for all attention head groups in a model.
    
    Handles GQA (Grouped Query Attention) where multiple query heads
    share the same KV heads.
    
    Provides utilities for:
    - Iterating over all head groups
    - Getting/setting head masks
    - Computing aggregate statistics
    - GQA-aware KV head management
    """
    
    def __init__(self, model: nn.Module, model_type: str = "qwen"):
        self.model = model
        self.model_type = model_type
        self.projections = get_attention_projections(model, model_type)
        self.attn_config = get_attention_config(model, model_type)
        self.num_layers = len(self.projections)
    
    def get_all_head_groups(self) -> List[AttentionHeadGroup]:
        """Get all head groups across all layers."""
        groups = []
        for layer_idx in range(self.num_layers):
            groups.extend(self.get_layer_head_groups(layer_idx))
        return groups
    
    def get_layer_head_groups(self, layer_idx: int) -> List[AttentionHeadGroup]:
        """Get all head groups for a specific layer."""
        groups = []
        for head_idx in range(self.attn_config.num_heads):
            kv_head_idx = self.attn_config.get_kv_head_for_query(head_idx)
            groups.append(AttentionHeadGroup(
                layer_idx=layer_idx,
                head_idx=head_idx,
                kv_head_idx=kv_head_idx,
                attn_config=self.attn_config
            ))
        return groups
    
    def get_projection(self, layer_idx: int) -> AttentionProjections:
        """Get the attention projections for a specific layer."""
        return self.projections[layer_idx]
    
    def total_heads(self) -> int:
        """Total number of query heads across all layers."""
        return self.num_layers * self.attn_config.num_heads
    
    def is_kv_head_fully_pruned(
        self, 
        kv_head_idx: int, 
        head_mask: torch.Tensor
    ) -> bool:
        """Check if all query heads sharing a KV head are pruned.
        
        Used to determine if a KV head can be physically removed.
        
        Args:
            kv_head_idx: KV head index
            head_mask: Query head mask [num_heads]
            
        Returns:
            True if all query heads for this KV head are pruned (mask=0)
        """
        query_heads = self.attn_config.get_query_heads_for_kv(kv_head_idx)
        for qh in query_heads:
            if head_mask[qh] > 0.5:  # Threshold for "kept"
                return False
        return True
    
    def get_prunable_kv_heads(self, head_mask: torch.Tensor) -> List[int]:
        """Get list of KV heads that can be pruned (all their query heads are pruned).
        
        Args:
            head_mask: Query head mask [num_heads]
            
        Returns:
            List of KV head indices that can be removed
        """
        prunable = []
        for kv_idx in range(self.attn_config.num_kv_heads):
            if self.is_kv_head_fully_pruned(kv_idx, head_mask):
                prunable.append(kv_idx)
        return prunable
    
    def __repr__(self) -> str:
        cfg = self.attn_config
        return (f"AttentionHeadGroupManager(num_layers={self.num_layers}, "
                f"num_heads={cfg.num_heads}, num_kv_heads={cfg.num_kv_heads}, "
                f"head_dim={cfg.head_dim}, group_size={cfg.group_size}, "
                f"is_gqa={cfg.is_gqa})")
