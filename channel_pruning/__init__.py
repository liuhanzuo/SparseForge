"""
Channel-level Structured Pruning for LLM FFN layers and Attention Heads.

This module implements:
1. FFN Channel Pruning: Reduce intermediate dimensions for real speedup
2. Attention Head Pruning: Reduce number of heads with GQA support

Both use SlimLLM-style importance computation combined with HASAST's 
mask-pushing mechanism (soft→hard, no gradient on masks).

Key Features:
- Hessian-OBD importance scoring for both FFN and attention
- GQA-aware head pruning with KV head distribution
- Soft-to-hard mask annealing
- Export support for truly reduced model dimensions

Usage:
    # FFN only
    from channel_pruning import ChannelPruningTrainer
    trainer = ChannelPruningTrainer(config)
    trainer.train()
    
    # FFN + Attention (unified)
    from channel_pruning import UnifiedPruningTrainer
    trainer = UnifiedPruningTrainer(config)
    trainer.train()
"""

from .config import ChannelPruningConfig

# FFN Channel Pruning
from .channel_groups import get_mlp_projections, MLPChannelGroup, MLPChannelGroupManager
from .channel_score import ChannelScoreComputer, LayerChannelScores, PCAManager, LayerPCAState
from .channel_mask import (
    ChannelMaskState, 
    ChannelMaskApplier,
    ChannelRegressionRecovery,
    LoRABypassModule,
    patch_model_mlp_forward,
    restore_model_mlp_forward
)

# Attention Head Pruning (GQA-aware)
from .attention_groups import (
    get_attention_projections,
    get_attention_config,
    AttentionConfig,
    AttentionProjections,
    AttentionHeadGroup,
    AttentionHeadGroupManager
)
from .attention_score import AttentionScoreComputer, LayerHeadScores
from .attention_mask import (
    AttentionMaskState,
    patch_model_attention_forward,
    remove_attention_hooks
)

# Export utilities
from .export_pruned import (
    export_pruned_model,
    export_pruned_attention,
    export_fully_pruned_model,
    compute_param_reduction,
    compute_attention_param_reduction
)

# Trainers
from .train_channel_pruning import ChannelPruningTrainer

__all__ = [
    # Config
    'ChannelPruningConfig',
    
    # FFN Channel Pruning
    'get_mlp_projections',
    'MLPChannelGroup',
    'MLPChannelGroupManager',
    'ChannelScoreComputer',
    'LayerChannelScores',
    'PCAManager',
    'LayerPCAState',
    'ChannelMaskState',
    'ChannelMaskApplier',
    'ChannelRegressionRecovery',
    'LoRABypassModule',
    'patch_model_mlp_forward',
    'restore_model_mlp_forward',
    
    # Attention Head Pruning
    'get_attention_projections',
    'get_attention_config',
    'AttentionConfig',
    'AttentionProjections',
    'AttentionHeadGroup',
    'AttentionHeadGroupManager',
    'AttentionScoreComputer',
    'LayerHeadScores',
    'AttentionMaskState',
    'patch_model_attention_forward',
    'remove_attention_hooks',
    
    # Export
    'export_pruned_model',
    'export_pruned_attention',
    'export_fully_pruned_model',
    'compute_param_reduction',
    'compute_attention_param_reduction',
    
    # Trainers
    'ChannelPruningTrainer',
]
