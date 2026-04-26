"""
Attention Head Mask: Manage head-level masks for attention pruning.

Each layer has a mask vector of shape [num_heads] for query heads.
The mask follows HASAST's soft-to-hard mechanism.

For GQA, we track query head masks, and KV heads are optionally pruned
when all their sharing query heads are pruned.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType

from .config import ChannelPruningConfig
from .attention_groups import (
    get_attention_projections,
    get_attention_config,
    AttentionProjections,
    AttentionConfig,
    AttentionHeadGroupManager
)


@dataclass
class LayerHeadMaskState:
    """Mask state for attention heads in a single layer."""
    layer_idx: int
    mask: torch.Tensor          # [num_heads], query head mask values in [0, 1]
    kv_mask: torch.Tensor       # [num_kv_heads], KV head mask (derived from query mask)
    frozen: torch.Tensor        # [num_heads], bool - frozen heads don't update
    temperature: float          # Current temperature for soft gating
    hardening_x: float          # Hardening progress: 1.0 = fully soft, 0.0 = fully hard


class AttentionMaskState:
    """Manages attention head masks for all layers with GQA support.
    
    Each layer has an independent mask vector for query heads.
    KV head masks are derived: a KV head is kept if any of its
    sharing query heads are kept.
    
    The mask update follows HASAST's mechanism:
    1. Compute target mask G from importance scores
    2. EMA update: mask <- (1-lr) * mask + lr * G
    3. Temperature annealing for softer->harder decisions
    4. Hardening: gradual transition from soft to binary masks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ChannelPruningConfig,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Extract attention info
        self.projections = get_attention_projections(model, config.model_type)
        self.attn_config = get_attention_config(model, config.model_type)
        self.num_layers = len(self.projections)
        self.num_heads = self.attn_config.num_heads
        self.num_kv_heads = self.attn_config.num_kv_heads
        self.group_size = self.attn_config.group_size
        
        # Initialize masks - all 1s (keep all heads initially)
        self.masks: Dict[int, LayerHeadMaskState] = {}
        for layer_idx in range(self.num_layers):
            self.masks[layer_idx] = LayerHeadMaskState(
                layer_idx=layer_idx,
                mask=torch.ones(self.num_heads, device=self.device),
                kv_mask=torch.ones(self.num_kv_heads, device=self.device),
                frozen=torch.zeros(self.num_heads, dtype=torch.bool, device=self.device),
                temperature=config.temp_init,
                hardening_x=1.0
            )
        
        # Per-layer keep ratios
        if hasattr(config, 'per_layer_attn_keep_ratio') and config.per_layer_attn_keep_ratio is not None:
            self.per_layer_keep_ratio = config.per_layer_attn_keep_ratio
        else:
            ratio = getattr(config, 'attention_keep_ratio', 0.75)
            self.per_layer_keep_ratio = {i: ratio for i in range(self.num_layers)}
    
    def get_keep_k(self, layer_idx: int) -> int:
        """Get the number of query heads to keep for a layer."""
        ratio = self.per_layer_keep_ratio.get(layer_idx, 0.75)
        return max(1, int(round(self.num_heads * ratio)))
    
    def get_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the current query head mask for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Mask tensor [num_heads] with values in [0, 1]
        """
        return self.masks[layer_idx].mask
    
    def get_kv_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the KV head mask derived from query head mask.
        
        A KV head is kept if any of its sharing query heads are kept.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            KV mask tensor [num_kv_heads]
        """
        return self.masks[layer_idx].kv_mask
    
    def _compute_kv_mask(self, query_mask: torch.Tensor) -> torch.Tensor:
        """Compute KV head mask from query head mask.
        
        A KV head mask value is the max of its sharing query heads.
        This ensures a KV head is kept if any query head needs it.
        
        Args:
            query_mask: Query head mask [num_heads]
            
        Returns:
            KV head mask [num_kv_heads]
        """
        # Reshape to [num_kv_heads, group_size]
        query_reshaped = query_mask.view(self.num_kv_heads, self.group_size)
        # Take max across group (if any query head is kept, keep KV head)
        kv_mask = query_reshaped.max(dim=1).values
        return kv_mask
    
    def get_hard_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the hard (binary) query head mask based on top-K scores.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Binary mask [num_heads] with exactly keep_k ones
        """
        mask = self.masks[layer_idx].mask
        keep_k = self.get_keep_k(layer_idx)
        
        _, top_indices = torch.topk(mask, k=keep_k, dim=0, largest=True)
        hard_mask = torch.zeros_like(mask)
        hard_mask.scatter_(0, top_indices, 1.0)
        
        return hard_mask
    
    def get_hard_kv_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the hard KV mask derived from hard query mask.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Binary KV mask [num_kv_heads]
        """
        hard_query_mask = self.get_hard_mask(layer_idx)
        return self._compute_kv_mask(hard_query_mask)
    
    def get_effective_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the effective query head mask (blend of soft and hard).
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Effective mask for forward pass [num_heads]
        """
        state = self.masks[layer_idx]
        soft_mask = state.mask
        hard_mask = self.get_hard_mask(layer_idx)
        
        hx = state.hardening_x
        return hx * soft_mask + (1 - hx) * hard_mask
    
    def get_effective_kv_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the effective KV head mask.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Effective KV mask [num_kv_heads]
        """
        effective_query_mask = self.get_effective_mask(layer_idx)
        return self._compute_kv_mask(effective_query_mask)
    
    @torch.no_grad()
    def compute_target_mask(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """Compute target mask from importance scores.
        
        Uses soft gating with temperature annealing:
        G[h] = sigmoid((score[h] - tau) / T)
        
        Args:
            layer_idx: Layer index
            scores: Head importance scores [num_heads]
            step: Current training step
            
        Returns:
            Target mask G [num_heads]
        """
        state = self.masks[layer_idx]
        keep_k = self.get_keep_k(layer_idx)
        
        # Effective sparsity with warmup
        warmup = self.config.sparsity_warmup_steps
        if step < warmup:
            progress = step / max(1, warmup)
            effective_keep_k = int(self.num_heads - progress * (self.num_heads - keep_k))
        else:
            effective_keep_k = keep_k
        
        effective_keep_k = max(1, min(self.num_heads, effective_keep_k))
        
        # Normalize scores
        scores_float = scores.float()
        mu = scores_float.mean()
        sigma = scores_float.std() + 1e-6
        scores_norm = (scores_float - mu) / sigma
        
        # Get threshold
        sorted_scores, _ = torch.sort(scores_norm, descending=True)
        tau = sorted_scores[effective_keep_k - 1]
        
        # Soft gating
        T = state.temperature
        G = torch.sigmoid((scores_norm - tau) / max(T, 1e-8))
        
        # Respect frozen heads
        frozen = state.frozen
        if frozen.any():
            G = G.clone()
            G[frozen] = state.mask[frozen]
        
        return G.to(scores.dtype)
    
    @torch.no_grad()
    def update_mask(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        step: int
    ):
        """Update the mask for a layer based on importance scores.
        
        Args:
            layer_idx: Layer index
            scores: Head importance scores [num_heads]
            step: Current training step
        """
        state = self.masks[layer_idx]
        config = self.config
        
        if step % config.mask_update_period != 0:
            return
        
        if step < config.sparsity_warmup_steps:
            return
        
        # Compute target mask
        G = self.compute_target_mask(layer_idx, scores, step)
        
        # EMA update
        lr = config.mask_lr
        state.mask.mul_(1 - lr).add_(G, alpha=lr).clamp_(0.0, 1.0)
        
        # Update KV mask
        state.kv_mask = self._compute_kv_mask(state.mask)
        
        # Temperature decay
        state.temperature = max(
            config.temp_min,
            state.temperature * config.temp_decay
        )
        
        # Update hardening progress
        start = config.hardening_start_step
        duration = config.hardening_duration
        if duration <= 0:
            state.hardening_x = 0.0 if step >= start else 1.0
        else:
            if step < start:
                state.hardening_x = 1.0
            elif step >= start + duration:
                state.hardening_x = 0.0
            else:
                progress = (step - start) / duration
                state.hardening_x = max(0.0, 1.0 - progress)
    
    @torch.no_grad()
    def update_all_masks(
        self,
        all_scores: List[torch.Tensor],
        step: int
    ):
        """Update masks for all layers.
        
        Args:
            all_scores: List of score tensors, one per layer
            step: Current training step
        """
        for layer_idx, scores in enumerate(all_scores):
            self.update_mask(layer_idx, scores, step)
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics for logging."""
        all_masks = [self.masks[i].mask for i in range(self.num_layers)]
        stacked = torch.stack(all_masks)
        
        avg_mask = stacked.mean().item()
        min_mask = stacked.min().item()
        max_mask = stacked.max().item()
        
        hard_masks = [(m > 0.5).float() for m in all_masks]
        hard_stacked = torch.stack(hard_masks)
        hard_sparsity = 1.0 - hard_stacked.mean().item()
        
        # KV head stats
        all_kv_masks = [self.masks[i].kv_mask for i in range(self.num_layers)]
        kv_stacked = torch.stack(all_kv_masks)
        kv_sparsity = 1.0 - (kv_stacked > 0.5).float().mean().item()
        
        return {
            "attn_avg_mask": avg_mask,
            "attn_min_mask": min_mask,
            "attn_max_mask": max_mask,
            "attn_hard_sparsity": hard_sparsity,
            "attn_kv_sparsity": kv_sparsity,
        }
    
    def finalize_masks(self):
        """Finalize masks to binary (for export)."""
        for layer_idx in range(self.num_layers):
            hard_mask = self.get_hard_mask(layer_idx)
            self.masks[layer_idx].mask = hard_mask
            self.masks[layer_idx].kv_mask = self._compute_kv_mask(hard_mask)
            self.masks[layer_idx].hardening_x = 0.0
    
    def get_kept_head_indices(self, layer_idx: int) -> torch.Tensor:
        """Get indices of kept query heads.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tensor of kept head indices
        """
        mask = self.get_hard_mask(layer_idx)
        return torch.where(mask > 0.5)[0]
    
    def get_kept_kv_indices(self, layer_idx: int) -> torch.Tensor:
        """Get indices of kept KV heads.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tensor of kept KV head indices
        """
        kv_mask = self.get_hard_kv_mask(layer_idx)
        return torch.where(kv_mask > 0.5)[0]


def create_masked_attention_forward(
    original_forward,
    mask_state: AttentionMaskState,
    layer_idx: int,
    attn_config: AttentionConfig
):
    """Create a masked forward function for an attention layer.
    
    This applies head masking at the attention output level:
    attn_output = attn_output * mask[None, :, None, None]
    
    Args:
        original_forward: The original attention forward method
        mask_state: AttentionMaskState instance
        layer_idx: Index of this layer
        attn_config: Attention configuration
        
    Returns:
        New forward function with masking
    """
    def masked_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        # Get original output
        # Note: Different models have different attention implementations
        # This is a generic approach that intercepts at the output level
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to heads
        num_heads = attn_config.num_heads
        num_kv_heads = attn_config.num_kv_heads
        head_dim = attn_config.head_dim
        
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if present
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV for GQA
        if num_kv_heads < num_heads:
            group_size = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(group_size, dim=1)
            value_states = value_states.repeat_interleave(group_size, dim=1)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout if in training
        if hasattr(self, 'attention_dropout') and self.training:
            attn_weights = self.attention_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)  # [B, num_heads, T, head_dim]
        
        # Apply head mask
        head_mask = mask_state.get_effective_mask(layer_idx)
        # mask is [num_heads], broadcast to [1, num_heads, 1, 1]
        mask_broadcast = head_mask.view(1, num_heads, 1, 1)
        attn_output = attn_output * mask_broadcast
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    return masked_forward


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings (helper function)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def patch_model_attention_forward(
    model: nn.Module,
    mask_state: AttentionMaskState,
    config: ChannelPruningConfig
):
    """Patch all attention modules to use masked forward.
    
    Note: Due to complexity of different attention implementations,
    we use a simpler approach - apply mask at attn_output level
    via a forward hook instead of replacing forward entirely.
    
    Args:
        model: The model to patch
        mask_state: AttentionMaskState instance
        config: Configuration
        
    Returns:
        List of hooks (for removal later)
    """
    hooks = []
    attn_config = mask_state.attn_config
    
    # Get the underlying model
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        base_model = model.model.model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # Find layers
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        raise ValueError("Cannot find layers in model")
    
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        
        # Create hook that masks attention output
        def create_hook(idx):
            def hook_fn(module, input, output):
                # output can be tuple: (attn_output, attn_weights, past_kv) or just attn_output
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                
                # Get mask
                mask = mask_state.get_effective_mask(idx)
                
                # attn_output is [batch, seq, hidden_size]
                # We need to mask by heads: hidden_size = num_heads * head_dim
                # Reshape, apply mask, reshape back
                batch, seq, _ = attn_output.shape
                num_heads = attn_config.num_heads
                head_dim = attn_config.head_dim
                
                # Reshape to [batch, seq, num_heads, head_dim]
                reshaped = attn_output.view(batch, seq, num_heads, head_dim)
                # Apply mask [num_heads] broadcast to [1, 1, num_heads, 1]
                masked = reshaped * mask.view(1, 1, num_heads, 1)
                # Reshape back
                masked_output = masked.view(batch, seq, -1)
                
                if isinstance(output, tuple):
                    return (masked_output,) + output[1:]
                else:
                    return masked_output
            return hook_fn
        
        hook = attn.o_proj.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)
    
    return hooks


def remove_attention_hooks(hooks: List):
    """Remove all attention hooks."""
    for hook in hooks:
        hook.remove()
