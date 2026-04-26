"""
Attention Head Score Computation: Compute importance scores for attention heads.

Uses SlimLLM-style importance scoring with GQA-aware aggregation:
- Per-weight score: s_w = (H + eps) * W^2  (Hessian-OBD style)
- Aggregate to head: s_head[h] = s_q[h] + s_k[h] + s_v[h] + s_o[h]

For GQA (Grouped Query Attention):
- Query heads use their own q_proj/o_proj slices
- KV scores are distributed across sharing query heads: s_k[h] = s_kv[h_kv] / group_size

The aggregation ensures that pruning a head considers all affected weights:
- q_proj rows I_h
- k_proj rows I_kv (distributed to sharing query heads)
- v_proj rows I_kv (distributed to sharing query heads)
- o_proj columns I_h
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .attention_groups import (
    get_attention_projections,
    get_attention_config,
    AttentionProjections,
    AttentionConfig,
    AttentionHeadGroupManager
)
from .config import ChannelPruningConfig


@dataclass
class LayerHeadScores:
    """Head scores for a single layer."""
    layer_idx: int
    scores: torch.Tensor      # [num_heads] - total head scores
    q_scores: torch.Tensor    # [num_heads] - contribution from q_proj
    k_scores: torch.Tensor    # [num_heads] - contribution from k_proj (GQA distributed)
    v_scores: torch.Tensor    # [num_heads] - contribution from v_proj (GQA distributed)
    o_scores: torch.Tensor    # [num_heads] - contribution from o_proj


class AttentionScoreComputer:
    """Computes importance scores for attention heads with GQA support.
    
    Uses Hessian-based scoring similar to HASAST/SlimLLM:
    - Per-weight importance: s = (H + eps) * W^2
    - Head importance: aggregate over all weights in the head group
    
    For GQA, KV scores are distributed to query heads:
    - s_k[h] = (1/g) * sum(S_k[I_kv, :]) where g = group_size
    - s_v[h] = (1/g) * sum(S_v[I_kv, :])
    
    This ensures fair comparison between query heads that share KV heads.
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
        self.model_type = config.model_type
        
        # Extract attention info
        self.projections = get_attention_projections(model, config.model_type)
        self.attn_config = get_attention_config(model, config.model_type)
        self.num_layers = len(self.projections)
        self.num_heads = self.attn_config.num_heads
        self.num_kv_heads = self.attn_config.num_kv_heads
        self.head_dim = self.attn_config.head_dim
        self.group_size = self.attn_config.group_size
        
        # Score aggregation weights (default all 1.0)
        self.alpha = getattr(config, 'attn_score_alpha', 1.0)  # q weight
        self.beta = getattr(config, 'attn_score_beta', 1.0)    # k weight
        self.gamma = getattr(config, 'attn_score_gamma', 1.0)  # v weight
        self.delta = getattr(config, 'attn_score_delta', 1.0)  # o weight
        
        # EMA for score smoothing
        self.score_ema_beta = config.score_ema_beta
        self.score_ema: Optional[Dict[int, torch.Tensor]] = None
        
        # Hessian diagonal estimates (per layer, per projection)
        self.hessian_q: Dict[int, torch.Tensor] = {}
        self.hessian_k: Dict[int, torch.Tensor] = {}
        self.hessian_v: Dict[int, torch.Tensor] = {}
        self.hessian_o: Dict[int, torch.Tensor] = {}
    
    def _get_weight_importance(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-weight importance scores.
        
        Args:
            weight: Weight tensor
            hessian: Hessian diagonal (same shape as weight)
            
        Returns:
            Importance scores (same shape as weight)
        """
        eps = 1e-8
        metric = self.config.importance_metric
        
        W = weight.detach().float()
        
        if metric == "hessian_obd":
            if hessian is not None and hessian.shape == W.shape:
                H = hessian.detach().float()
                scores = (H + eps) * (W * W)
            else:
                scores = W * W
        elif metric == "magnitude":
            scores = W.abs()
        else:
            scores = W * W
        
        return scores
    
    def _aggregate_q_scores(
        self,
        q_scores: torch.Tensor,  # [num_heads * head_dim, hidden_size]
    ) -> torch.Tensor:
        """Aggregate q_proj scores to per-head scores.
        
        For query head h:
        s_q[h] = sum(q_scores[I_h, :]) where I_h = [h*d, (h+1)*d)
        
        Args:
            q_scores: Per-weight scores [num_heads * head_dim, hidden_size]
            
        Returns:
            Per-head q scores [num_heads]
        """
        # Reshape to [num_heads, head_dim, hidden_size]
        scores_reshaped = q_scores.view(self.num_heads, self.head_dim, -1)
        # Sum over head_dim and hidden_size
        head_scores = scores_reshaped.sum(dim=(1, 2))  # [num_heads]
        return head_scores
    
    def _aggregate_o_scores(
        self,
        o_scores: torch.Tensor,  # [hidden_size, num_heads * head_dim]
    ) -> torch.Tensor:
        """Aggregate o_proj scores to per-head scores.
        
        For query head h:
        s_o[h] = sum(o_scores[:, I_h]) where I_h = [h*d, (h+1)*d)
        
        Args:
            o_scores: Per-weight scores [hidden_size, num_heads * head_dim]
            
        Returns:
            Per-head o scores [num_heads]
        """
        # Reshape to [hidden_size, num_heads, head_dim]
        scores_reshaped = o_scores.view(-1, self.num_heads, self.head_dim)
        # Sum over hidden_size and head_dim
        head_scores = scores_reshaped.sum(dim=(0, 2))  # [num_heads]
        return head_scores
    
    def _aggregate_kv_scores_gqa(
        self,
        kv_scores: torch.Tensor,  # [num_kv_heads * head_dim, hidden_size]
    ) -> torch.Tensor:
        """Aggregate k_proj or v_proj scores to per-query-head scores with GQA distribution.
        
        For GQA, each KV head is shared by `group_size` query heads.
        We distribute the KV score evenly:
        
        s_kv[h] = (1/g) * sum(kv_scores[I_kv, :])
        where h_kv = h // g, I_kv = [h_kv*d, (h_kv+1)*d)
        
        Args:
            kv_scores: Per-weight scores [num_kv_heads * head_dim, hidden_size]
            
        Returns:
            Per-query-head distributed scores [num_heads]
        """
        # First compute per-KV-head scores
        # Reshape to [num_kv_heads, head_dim, hidden_size]
        scores_reshaped = kv_scores.view(self.num_kv_heads, self.head_dim, -1)
        # Sum over head_dim and hidden_size
        kv_head_scores = scores_reshaped.sum(dim=(1, 2))  # [num_kv_heads]
        
        # Distribute to query heads (divide by group_size for fair comparison)
        # Each query head gets (1/g) of its KV head's score
        distributed_scores = torch.zeros(self.num_heads, device=kv_head_scores.device, dtype=kv_head_scores.dtype)
        
        for h in range(self.num_heads):
            kv_idx = h // self.group_size
            distributed_scores[h] = kv_head_scores[kv_idx] / self.group_size
        
        return distributed_scores
    
    def _aggregate_head_scores(
        self,
        q_scores: torch.Tensor,   # [num_heads * head_dim, hidden_size]
        k_scores: torch.Tensor,   # [num_kv_heads * head_dim, hidden_size]
        v_scores: torch.Tensor,   # [num_kv_heads * head_dim, hidden_size]
        o_scores: torch.Tensor,   # [hidden_size, num_heads * head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate per-weight scores to head scores with GQA handling.
        
        Final score:
        s_head[h] = α * s_q[h] + β * s_k[h] + γ * s_v[h] + δ * s_o[h]
        
        Where s_k and s_v are GQA-distributed.
        
        Args:
            q_scores: Q projection per-weight scores
            k_scores: K projection per-weight scores
            v_scores: V projection per-weight scores
            o_scores: O projection per-weight scores
            
        Returns:
            (total_scores, s_q, s_k, s_v, s_o) all [num_heads]
        """
        # Aggregate each projection
        s_q = self._aggregate_q_scores(q_scores)
        s_o = self._aggregate_o_scores(o_scores)
        
        # GQA-aware KV aggregation
        s_k = self._aggregate_kv_scores_gqa(k_scores)
        s_v = self._aggregate_kv_scores_gqa(v_scores)
        
        # Optional normalization
        if self.config.normalize_scores:
            hidden_size = q_scores.shape[1]
            norm_factor = hidden_size * self.head_dim
            s_q = s_q / norm_factor
            s_k = s_k / norm_factor
            s_v = s_v / norm_factor
            s_o = s_o / norm_factor
        
        # Weighted sum
        total = self.alpha * s_q + self.beta * s_k + self.gamma * s_v + self.delta * s_o
        
        return total, s_q, s_k, s_v, s_o
    
    @torch.no_grad()
    def compute_layer_scores(self, layer_idx: int) -> LayerHeadScores:
        """Compute head scores for a single layer.
        
        Args:
            layer_idx: Index of the transformer layer
            
        Returns:
            LayerHeadScores with per-head importance scores
        """
        proj = self.projections[layer_idx]
        
        # Get weights
        q_weight = proj.q_proj.weight  # [num_heads * head_dim, hidden_size]
        k_weight = proj.k_proj.weight  # [num_kv_heads * head_dim, hidden_size]
        v_weight = proj.v_proj.weight  # [num_kv_heads * head_dim, hidden_size]
        o_weight = proj.o_proj.weight  # [hidden_size, num_heads * head_dim]
        
        # Get Hessian estimates (if available)
        q_hessian = self.hessian_q.get(layer_idx)
        k_hessian = self.hessian_k.get(layer_idx)
        v_hessian = self.hessian_v.get(layer_idx)
        o_hessian = self.hessian_o.get(layer_idx)
        
        # Compute per-weight importance
        q_scores = self._get_weight_importance(q_weight, q_hessian)
        k_scores = self._get_weight_importance(k_weight, k_hessian)
        v_scores = self._get_weight_importance(v_weight, v_hessian)
        o_scores = self._get_weight_importance(o_weight, o_hessian)
        
        # Aggregate to head scores
        total, s_q, s_k, s_v, s_o = self._aggregate_head_scores(
            q_scores, k_scores, v_scores, o_scores
        )
        
        return LayerHeadScores(
            layer_idx=layer_idx,
            scores=total,
            q_scores=s_q,
            k_scores=s_k,
            v_scores=s_v,
            o_scores=s_o
        )
    
    @torch.no_grad()
    def compute_all_scores(self) -> List[LayerHeadScores]:
        """Compute head scores for all layers.
        
        Returns:
            List of LayerHeadScores, one per layer
        """
        all_scores = []
        for layer_idx in range(self.num_layers):
            scores = self.compute_layer_scores(layer_idx)
            all_scores.append(scores)
        return all_scores
    
    @torch.no_grad()
    def update_hessian_ema(self, layer_idx: int):
        """Update Hessian diagonal estimate using gradient squared EMA.
        
        H ≈ E[grad^2]
        
        Args:
            layer_idx: Index of the layer to update
        """
        proj = self.projections[layer_idx]
        beta = self.config.score_ema_beta
        
        for name, weight, hessian_dict in [
            ('q', proj.q_proj.weight, self.hessian_q),
            ('k', proj.k_proj.weight, self.hessian_k),
            ('v', proj.v_proj.weight, self.hessian_v),
            ('o', proj.o_proj.weight, self.hessian_o),
        ]:
            if weight.grad is None:
                continue
            
            grad_sq = (weight.grad.detach() ** 2).float()
            
            if layer_idx not in hessian_dict:
                hessian_dict[layer_idx] = grad_sq.clone()
            else:
                hessian_dict[layer_idx].mul_(beta).add_(grad_sq, alpha=1 - beta)
    
    @torch.no_grad()
    def update_all_hessian_ema(self):
        """Update Hessian estimates for all layers."""
        for layer_idx in range(self.num_layers):
            self.update_hessian_ema(layer_idx)
    
    @torch.no_grad()
    def update_score_ema(self, layer_scores: List[LayerHeadScores]):
        """Update the running EMA of head scores.
        
        Args:
            layer_scores: Current layer scores
        """
        beta = self.score_ema_beta
        
        if self.score_ema is None:
            self.score_ema = {}
            for ls in layer_scores:
                self.score_ema[ls.layer_idx] = ls.scores.clone()
        else:
            for ls in layer_scores:
                if ls.layer_idx in self.score_ema:
                    self.score_ema[ls.layer_idx].mul_(beta).add_(ls.scores, alpha=1 - beta)
                else:
                    self.score_ema[ls.layer_idx] = ls.scores.clone()
    
    @torch.no_grad()
    def get_smoothed_scores(self, layer_idx: int) -> torch.Tensor:
        """Get the smoothed (EMA) scores for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Smoothed head scores [num_heads]
        """
        if self.score_ema is not None and layer_idx in self.score_ema:
            return self.score_ema[layer_idx]
        else:
            return self.compute_layer_scores(layer_idx).scores
    
    def get_kv_head_scores(self, layer_idx: int) -> torch.Tensor:
        """Compute raw KV head scores (not distributed).
        
        Used to determine if a KV head can be pruned.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            KV head scores [num_kv_heads]
        """
        proj = self.projections[layer_idx]
        
        k_weight = proj.k_proj.weight
        v_weight = proj.v_proj.weight
        
        k_hessian = self.hessian_k.get(layer_idx)
        v_hessian = self.hessian_v.get(layer_idx)
        
        k_scores = self._get_weight_importance(k_weight, k_hessian)
        v_scores = self._get_weight_importance(v_weight, v_hessian)
        
        # Aggregate to KV heads
        k_reshaped = k_scores.view(self.num_kv_heads, self.head_dim, -1)
        v_reshaped = v_scores.view(self.num_kv_heads, self.head_dim, -1)
        
        kv_head_scores = k_reshaped.sum(dim=(1, 2)) + v_reshaped.sum(dim=(1, 2))
        
        return kv_head_scores
