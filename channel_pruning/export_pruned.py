"""
Export Pruned Model: Rebuild model with reduced FFN dimensions.

After training, convert the masked model to a truly pruned model
with smaller intermediate dimensions for real speedup.

This involves:
1. Identify kept channels (mask > threshold)
2. Rebuild Linear layers with reduced dimensions
3. Update model config
4. Save in HuggingFace format
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import torch
import torch.nn as nn
from copy import deepcopy

from .channel_groups import get_mlp_projections, MLPProjections
from .channel_mask import ChannelMaskState
from .config import ChannelPruningConfig


def get_kept_channels(mask: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Get indices of channels to keep based on mask.
    
    Args:
        mask: Channel mask [intermediate_size]
        keep_k: Number of channels to keep
        
    Returns:
        Indices of kept channels [keep_k]
    """
    # Top-K by mask value
    _, indices = torch.topk(mask, k=keep_k, dim=0, largest=True)
    # Sort indices for deterministic ordering
    indices, _ = torch.sort(indices)
    return indices


def prune_linear_layer(
    layer: nn.Linear,
    keep_indices: torch.Tensor,
    dim: int
) -> nn.Linear:
    """Prune a Linear layer along specified dimension.
    
    Args:
        layer: The Linear layer to prune
        keep_indices: Indices to keep
        dim: Dimension to prune (0=output, 1=input)
        
    Returns:
        New Linear layer with reduced dimension
    """
    weight = layer.weight.data
    bias = layer.bias.data if layer.bias is not None else None
    
    if dim == 0:
        # Prune output dimension: weight[keep_indices, :]
        new_weight = weight.index_select(0, keep_indices)
        new_out_features = len(keep_indices)
        new_in_features = weight.shape[1]
        new_bias = bias.index_select(0, keep_indices) if bias is not None else None
    elif dim == 1:
        # Prune input dimension: weight[:, keep_indices]
        new_weight = weight.index_select(1, keep_indices)
        new_out_features = weight.shape[0]
        new_in_features = len(keep_indices)
        new_bias = bias  # Bias doesn't change for input pruning
    else:
        raise ValueError(f"Invalid dim: {dim}")
    
    # Create new Linear layer
    new_layer = nn.Linear(
        new_in_features,
        new_out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype
    )
    new_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        new_layer.bias.data.copy_(new_bias)
    
    return new_layer


def prune_mlp_layer(
    proj: MLPProjections,
    keep_indices: torch.Tensor
) -> Tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Prune all projections in an MLP layer.
    
    Args:
        proj: MLP projections (up_proj, gate_proj, down_proj)
        keep_indices: Channel indices to keep
        
    Returns:
        Tuple of pruned (up_proj, gate_proj, down_proj)
    """
    # up_proj: [intermediate_size, hidden_size] -> prune dim=0
    new_up = prune_linear_layer(proj.up_proj, keep_indices, dim=0)
    
    # gate_proj: [intermediate_size, hidden_size] -> prune dim=0
    # Note: For OPT, gate_proj == up_proj, so check for this
    if proj.gate_proj is proj.up_proj:
        new_gate = new_up
    else:
        new_gate = prune_linear_layer(proj.gate_proj, keep_indices, dim=0)
    
    # down_proj: [hidden_size, intermediate_size] -> prune dim=1
    new_down = prune_linear_layer(proj.down_proj, keep_indices, dim=1)
    
    return new_up, new_gate, new_down


def export_pruned_model(
    model: nn.Module,
    mask_state: ChannelMaskState,
    config: ChannelPruningConfig,
    output_path: str,
    save_format: str = "safetensors"  # or "pytorch"
) -> nn.Module:
    """Export a pruned model with reduced FFN dimensions.
    
    This creates a new model where:
    - FFN intermediate dimensions are reduced based on masks
    - Weights are physically smaller (real memory/compute savings)
    - Config is updated with new dimensions
    
    Args:
        model: The trained model with masks
        mask_state: ChannelMaskState with final masks
        config: Configuration
        output_path: Path to save the pruned model
        save_format: "safetensors" or "pytorch"
        
    Returns:
        The pruned model
    """
    print(f"[export_pruned_model] Exporting to {output_path}")
    
    # Finalize masks to binary
    mask_state.finalize_masks()
    
    # Get projections
    projections = get_mlp_projections(model, config.model_type)
    num_layers = len(projections)
    
    # Collect per-layer keep indices
    layer_keep_indices = {}
    layer_new_sizes = {}
    
    for layer_idx in range(num_layers):
        mask = mask_state.get_mask(layer_idx)
        keep_k = mask_state.get_keep_k(layer_idx)
        keep_indices = get_kept_channels(mask, keep_k)
        layer_keep_indices[layer_idx] = keep_indices
        layer_new_sizes[layer_idx] = len(keep_indices)
        print(f"  Layer {layer_idx}: {mask_state.intermediate_size} -> {len(keep_indices)} channels")
    
    # Clone model for modification
    # Note: For large models, consider modifying in-place
    pruned_model = deepcopy(model)
    
    # Get the underlying model structure
    if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'model'):
        base_model = pruned_model.model.model
    elif hasattr(pruned_model, 'model'):
        base_model = pruned_model.model
    else:
        base_model = pruned_model
    
    # Find layers
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        raise ValueError("Cannot find layers in model")
    
    # Prune each layer's MLP
    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        keep_indices = layer_keep_indices[layer_idx].to(mlp.up_proj.weight.device)
        
        # Prune projections
        if hasattr(mlp, 'up_proj'):
            mlp.up_proj = prune_linear_layer(mlp.up_proj, keep_indices, dim=0)
            if hasattr(mlp, 'gate_proj') and mlp.gate_proj is not mlp.up_proj:
                mlp.gate_proj = prune_linear_layer(mlp.gate_proj, keep_indices, dim=0)
            mlp.down_proj = prune_linear_layer(mlp.down_proj, keep_indices, dim=1)
        elif hasattr(mlp, 'w1'):
            mlp.w1 = prune_linear_layer(mlp.w1, keep_indices, dim=0)
            mlp.w3 = prune_linear_layer(mlp.w3, keep_indices, dim=0)
            mlp.w2 = prune_linear_layer(mlp.w2, keep_indices, dim=1)
        elif hasattr(mlp, 'fc1'):  # OPT
            mlp.fc1 = prune_linear_layer(mlp.fc1, keep_indices, dim=0)
            mlp.fc2 = prune_linear_layer(mlp.fc2, keep_indices, dim=1)
    
    # Update config
    model_config = None
    if hasattr(pruned_model, 'config'):
        model_config = pruned_model.config
    elif hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'config'):
        model_config = pruned_model.model.config
    
    if model_config is not None:
        # For uniform pruning, update intermediate_size
        if len(set(layer_new_sizes.values())) == 1:
            # All layers have same size
            new_intermediate = list(layer_new_sizes.values())[0]
            if hasattr(model_config, 'intermediate_size'):
                model_config.intermediate_size = new_intermediate
            elif hasattr(model_config, 'ffn_dim'):
                model_config.ffn_dim = new_intermediate
    
    # Save the model
    os.makedirs(output_path, exist_ok=True)
    
    if save_format == "safetensors":
        # Use HuggingFace's save_pretrained if available
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'save_pretrained'):
            pruned_model.model.save_pretrained(
                output_path,
                safe_serialization=True
            )
        else:
            # Manual save
            state_dict = pruned_model.state_dict()
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(output_path, "model.safetensors"))
    else:
        # PyTorch format
        state_dict = pruned_model.state_dict()
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
    
    # Save config
    if model_config is not None:
        config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else vars(model_config)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Save pruning info
    pruning_info = {
        "original_intermediate_size": mask_state.intermediate_size,
        "per_layer_new_sizes": layer_new_sizes,
        "ffn_keep_ratio": config.ffn_keep_ratio,
        "per_layer_keep_ratio": config.per_layer_keep_ratio,
        "importance_metric": config.importance_metric,
    }
    with open(os.path.join(output_path, "pruning_info.json"), "w") as f:
        json.dump(pruning_info, f, indent=2)
    
    print(f"[export_pruned_model] Saved to {output_path}")
    print(f"  Original intermediate size: {mask_state.intermediate_size}")
    print(f"  New intermediate sizes: {layer_new_sizes}")
    
    return pruned_model


def compute_param_reduction(
    model: nn.Module,
    mask_state: ChannelMaskState,
    config: ChannelPruningConfig
) -> Dict[str, float]:
    """Compute parameter reduction statistics.
    
    Args:
        model: The model
        mask_state: ChannelMaskState with masks
        config: Configuration
        
    Returns:
        Dict with reduction statistics
    """
    projections = get_mlp_projections(model, config.model_type)
    num_layers = len(projections)
    
    original_params = 0
    pruned_params = 0
    
    for layer_idx in range(num_layers):
        proj = projections[layer_idx]
        keep_k = mask_state.get_keep_k(layer_idx)
        
        # Original MLP params
        up_params = proj.up_proj.weight.numel()
        gate_params = proj.gate_proj.weight.numel() if proj.gate_proj is not proj.up_proj else 0
        down_params = proj.down_proj.weight.numel()
        
        original_layer = up_params + gate_params + down_params
        original_params += original_layer
        
        # Pruned MLP params
        hidden_size = proj.up_proj.weight.shape[1]
        pruned_up = keep_k * hidden_size
        pruned_gate = keep_k * hidden_size if proj.gate_proj is not proj.up_proj else 0
        pruned_down = hidden_size * keep_k
        
        pruned_layer = pruned_up + pruned_gate + pruned_down
        pruned_params += pruned_layer
    
    reduction = 1.0 - (pruned_params / original_params)
    
    return {
        "original_ffn_params": original_params,
        "pruned_ffn_params": pruned_params,
        "ffn_param_reduction": reduction,
        "ffn_remaining_ratio": 1.0 - reduction,
    }


# =============================================================================
# Attention Head Pruning Export
# =============================================================================

def prune_attention_layer(
    attn_module,
    keep_head_indices: torch.Tensor,
    keep_kv_indices: torch.Tensor,
    attn_config,
    model_type: str = "qwen"
) -> None:
    """Prune attention projections in-place.
    
    This reduces:
    - q_proj: out_features from num_heads*head_dim to keep_heads*head_dim
    - k_proj: out_features from num_kv_heads*head_dim to keep_kv*head_dim
    - v_proj: out_features from num_kv_heads*head_dim to keep_kv*head_dim
    - o_proj: in_features from num_heads*head_dim to keep_heads*head_dim
    
    Args:
        attn_module: The attention module to prune
        keep_head_indices: Query head indices to keep [keep_heads]
        keep_kv_indices: KV head indices to keep [keep_kv]
        attn_config: AttentionConfig
        model_type: Model architecture type
    """
    head_dim = attn_config.head_dim
    device = attn_module.q_proj.weight.device
    dtype = attn_module.q_proj.weight.dtype
    
    # Compute weight indices for query heads
    # Each head corresponds to head_dim consecutive rows/columns
    q_keep_rows = []
    for h in keep_head_indices:
        start = h.item() * head_dim
        end = (h.item() + 1) * head_dim
        q_keep_rows.extend(range(start, end))
    q_keep_rows = torch.tensor(q_keep_rows, device=device, dtype=torch.long)
    
    # Compute weight indices for KV heads
    kv_keep_rows = []
    for h in keep_kv_indices:
        start = h.item() * head_dim
        end = (h.item() + 1) * head_dim
        kv_keep_rows.extend(range(start, end))
    kv_keep_rows = torch.tensor(kv_keep_rows, device=device, dtype=torch.long)
    
    # Prune q_proj: [num_heads * head_dim, hidden_size] -> [keep_heads * head_dim, hidden_size]
    attn_module.q_proj = prune_linear_layer(attn_module.q_proj, q_keep_rows, dim=0)
    
    # Prune k_proj: [num_kv_heads * head_dim, hidden_size] -> [keep_kv * head_dim, hidden_size]
    attn_module.k_proj = prune_linear_layer(attn_module.k_proj, kv_keep_rows, dim=0)
    
    # Prune v_proj: same as k_proj
    attn_module.v_proj = prune_linear_layer(attn_module.v_proj, kv_keep_rows, dim=0)
    
    # Prune o_proj: [hidden_size, num_heads * head_dim] -> [hidden_size, keep_heads * head_dim]
    attn_module.o_proj = prune_linear_layer(attn_module.o_proj, q_keep_rows, dim=1)
    
    # Update num_heads attribute if present
    if hasattr(attn_module, 'num_heads'):
        attn_module.num_heads = len(keep_head_indices)
    if hasattr(attn_module, 'num_key_value_heads'):
        attn_module.num_key_value_heads = len(keep_kv_indices)


def export_pruned_attention(
    model: nn.Module,
    attn_mask_state,  # AttentionMaskState
    config: ChannelPruningConfig,
    output_path: str,
    save_format: str = "safetensors"
) -> nn.Module:
    """Export a model with pruned attention heads.
    
    This creates a new model where:
    - Attention head dimensions are reduced based on masks
    - KV heads are pruned when all their query heads are pruned
    - Weights are physically smaller (real memory/compute savings)
    - Config is updated with new head counts
    
    Args:
        model: The trained model with masks
        attn_mask_state: AttentionMaskState with final masks
        config: Configuration
        output_path: Path to save the pruned model
        save_format: "safetensors" or "pytorch"
        
    Returns:
        The pruned model
    """
    from .attention_groups import get_attention_projections, get_attention_config
    
    print(f"[export_pruned_attention] Exporting to {output_path}")
    
    # Finalize masks to binary
    attn_mask_state.finalize_masks()
    
    attn_config = attn_mask_state.attn_config
    num_layers = attn_mask_state.num_layers
    
    # Collect per-layer keep indices
    layer_keep_heads = {}
    layer_keep_kv = {}
    
    for layer_idx in range(num_layers):
        keep_heads = attn_mask_state.get_kept_head_indices(layer_idx)
        keep_kv = attn_mask_state.get_kept_kv_indices(layer_idx)
        layer_keep_heads[layer_idx] = keep_heads
        layer_keep_kv[layer_idx] = keep_kv
        print(f"  Layer {layer_idx}: {attn_config.num_heads} -> {len(keep_heads)} heads, "
              f"{attn_config.num_kv_heads} -> {len(keep_kv)} KV heads")
    
    # Clone model for modification
    pruned_model = deepcopy(model)
    
    # Get the underlying model structure
    if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'model'):
        base_model = pruned_model.model.model
    elif hasattr(pruned_model, 'model'):
        base_model = pruned_model.model
    else:
        base_model = pruned_model
    
    # Find layers
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        raise ValueError("Cannot find layers in model")
    
    # Prune each layer's attention
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        keep_heads = layer_keep_heads[layer_idx].to(attn.q_proj.weight.device)
        keep_kv = layer_keep_kv[layer_idx].to(attn.k_proj.weight.device)
        
        prune_attention_layer(
            attn,
            keep_heads,
            keep_kv,
            attn_config,
            config.model_type
        )
    
    # Update config
    model_config = None
    if hasattr(pruned_model, 'config'):
        model_config = pruned_model.config
    elif hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'config'):
        model_config = pruned_model.model.config
    
    if model_config is not None:
        # Check if all layers have same head count
        unique_head_counts = set(len(h) for h in layer_keep_heads.values())
        unique_kv_counts = set(len(k) for k in layer_keep_kv.values())
        
        if len(unique_head_counts) == 1:
            new_num_heads = len(layer_keep_heads[0])
            if hasattr(model_config, 'num_attention_heads'):
                model_config.num_attention_heads = new_num_heads
        
        if len(unique_kv_counts) == 1:
            new_num_kv = len(layer_keep_kv[0])
            if hasattr(model_config, 'num_key_value_heads'):
                model_config.num_key_value_heads = new_num_kv
    
    # Save the model
    os.makedirs(output_path, exist_ok=True)
    
    if save_format == "safetensors":
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'save_pretrained'):
            pruned_model.model.save_pretrained(
                output_path,
                safe_serialization=True
            )
        else:
            state_dict = pruned_model.state_dict()
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(output_path, "model.safetensors"))
    else:
        state_dict = pruned_model.state_dict()
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
    
    # Save config
    if model_config is not None:
        config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else vars(model_config)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Save pruning info
    pruning_info = {
        "original_num_heads": attn_config.num_heads,
        "original_num_kv_heads": attn_config.num_kv_heads,
        "per_layer_kept_heads": {k: len(v) for k, v in layer_keep_heads.items()},
        "per_layer_kept_kv_heads": {k: len(v) for k, v in layer_keep_kv.items()},
        "head_dim": attn_config.head_dim,
    }
    with open(os.path.join(output_path, "attention_pruning_info.json"), "w") as f:
        json.dump(pruning_info, f, indent=2)
    
    print(f"[export_pruned_attention] Saved to {output_path}")
    
    return pruned_model


def export_fully_pruned_model(
    model: nn.Module,
    ffn_mask_state,      # ChannelMaskState (can be None)
    attn_mask_state,     # AttentionMaskState (can be None)
    config: ChannelPruningConfig,
    output_path: str,
    save_format: str = "safetensors"
) -> nn.Module:
    """Export a model with both FFN and attention pruning.
    
    This combines FFN channel pruning and attention head pruning.
    
    Args:
        model: The trained model with masks
        ffn_mask_state: ChannelMaskState for FFN (None to skip)
        attn_mask_state: AttentionMaskState for attention (None to skip)
        config: Configuration
        output_path: Path to save the pruned model
        save_format: "safetensors" or "pytorch"
        
    Returns:
        The pruned model
    """
    from .attention_groups import get_attention_config
    
    print(f"[export_fully_pruned_model] Exporting to {output_path}")
    
    # Clone model
    pruned_model = deepcopy(model)
    
    # Get the underlying model structure
    if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'model'):
        base_model = pruned_model.model.model
    elif hasattr(pruned_model, 'model'):
        base_model = pruned_model.model
    else:
        base_model = pruned_model
    
    # Find layers
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        raise ValueError("Cannot find layers in model")
    
    pruning_info = {}
    
    # Prune FFN if mask state provided
    if ffn_mask_state is not None:
        ffn_mask_state.finalize_masks()
        
        layer_ffn_sizes = {}
        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mask = ffn_mask_state.get_mask(layer_idx)
            keep_k = ffn_mask_state.get_keep_k(layer_idx)
            keep_indices = get_kept_channels(mask, keep_k).to(mlp.up_proj.weight.device)
            
            # Prune MLP
            if hasattr(mlp, 'up_proj'):
                mlp.up_proj = prune_linear_layer(mlp.up_proj, keep_indices, dim=0)
                if hasattr(mlp, 'gate_proj') and mlp.gate_proj is not mlp.up_proj:
                    mlp.gate_proj = prune_linear_layer(mlp.gate_proj, keep_indices, dim=0)
                mlp.down_proj = prune_linear_layer(mlp.down_proj, keep_indices, dim=1)
            
            layer_ffn_sizes[layer_idx] = len(keep_indices)
            print(f"  Layer {layer_idx} FFN: {ffn_mask_state.intermediate_size} -> {len(keep_indices)}")
        
        pruning_info["ffn"] = {
            "original_intermediate_size": ffn_mask_state.intermediate_size,
            "per_layer_new_sizes": layer_ffn_sizes,
        }
    
    # Prune attention if mask state provided
    if attn_mask_state is not None:
        attn_mask_state.finalize_masks()
        attn_config = attn_mask_state.attn_config
        
        layer_head_counts = {}
        layer_kv_counts = {}
        
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            keep_heads = attn_mask_state.get_kept_head_indices(layer_idx).to(attn.q_proj.weight.device)
            keep_kv = attn_mask_state.get_kept_kv_indices(layer_idx).to(attn.k_proj.weight.device)
            
            prune_attention_layer(
                attn,
                keep_heads,
                keep_kv,
                attn_config,
                config.model_type
            )
            
            layer_head_counts[layer_idx] = len(keep_heads)
            layer_kv_counts[layer_idx] = len(keep_kv)
            print(f"  Layer {layer_idx} Attn: {attn_config.num_heads} -> {len(keep_heads)} heads, "
                  f"{attn_config.num_kv_heads} -> {len(keep_kv)} KV")
        
        pruning_info["attention"] = {
            "original_num_heads": attn_config.num_heads,
            "original_num_kv_heads": attn_config.num_kv_heads,
            "per_layer_head_counts": layer_head_counts,
            "per_layer_kv_counts": layer_kv_counts,
        }
    
    # Update config
    model_config = None
    if hasattr(pruned_model, 'config'):
        model_config = pruned_model.config
    elif hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'config'):
        model_config = pruned_model.model.config
    
    if model_config is not None:
        # Update FFN size
        if ffn_mask_state is not None:
            unique_sizes = set(pruning_info["ffn"]["per_layer_new_sizes"].values())
            if len(unique_sizes) == 1:
                new_size = list(unique_sizes)[0]
                if hasattr(model_config, 'intermediate_size'):
                    model_config.intermediate_size = new_size
        
        # Update attention heads
        if attn_mask_state is not None:
            unique_heads = set(pruning_info["attention"]["per_layer_head_counts"].values())
            unique_kv = set(pruning_info["attention"]["per_layer_kv_counts"].values())
            
            if len(unique_heads) == 1:
                if hasattr(model_config, 'num_attention_heads'):
                    model_config.num_attention_heads = list(unique_heads)[0]
            
            if len(unique_kv) == 1:
                if hasattr(model_config, 'num_key_value_heads'):
                    model_config.num_key_value_heads = list(unique_kv)[0]
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    
    if save_format == "safetensors":
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'save_pretrained'):
            pruned_model.model.save_pretrained(output_path, safe_serialization=True)
        else:
            state_dict = pruned_model.state_dict()
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(output_path, "model.safetensors"))
    else:
        state_dict = pruned_model.state_dict()
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
    
    # Save config
    if model_config is not None:
        config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else vars(model_config)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Save pruning info
    with open(os.path.join(output_path, "pruning_info.json"), "w") as f:
        json.dump(pruning_info, f, indent=2)
    
    print(f"[export_fully_pruned_model] Saved to {output_path}")
    
    return pruned_model


def compute_attention_param_reduction(
    model: nn.Module,
    attn_mask_state,  # AttentionMaskState
    config: ChannelPruningConfig
) -> Dict[str, float]:
    """Compute attention parameter reduction statistics.
    
    Args:
        model: The model
        attn_mask_state: AttentionMaskState with masks
        config: Configuration
        
    Returns:
        Dict with reduction statistics
    """
    from .attention_groups import get_attention_projections
    
    projections = get_attention_projections(model, config.model_type)
    num_layers = len(projections)
    attn_config = attn_mask_state.attn_config
    
    original_params = 0
    pruned_params = 0
    
    for layer_idx in range(num_layers):
        proj = projections[layer_idx]
        
        # Original attention params
        q_params = proj.q_proj.weight.numel()
        k_params = proj.k_proj.weight.numel()
        v_params = proj.v_proj.weight.numel()
        o_params = proj.o_proj.weight.numel()
        
        original_layer = q_params + k_params + v_params + o_params
        original_params += original_layer
        
        # Pruned params
        keep_heads = len(attn_mask_state.get_kept_head_indices(layer_idx))
        keep_kv = len(attn_mask_state.get_kept_kv_indices(layer_idx))
        hidden_size = attn_config.hidden_size
        head_dim = attn_config.head_dim
        
        pruned_q = keep_heads * head_dim * hidden_size
        pruned_k = keep_kv * head_dim * hidden_size
        pruned_v = keep_kv * head_dim * hidden_size
        pruned_o = hidden_size * keep_heads * head_dim
        
        pruned_layer = pruned_q + pruned_k + pruned_v + pruned_o
        pruned_params += pruned_layer
    
    reduction = 1.0 - (pruned_params / original_params) if original_params > 0 else 0.0
    
    return {
        "original_attn_params": original_params,
        "pruned_attn_params": pruned_params,
        "attn_param_reduction": reduction,
        "attn_remaining_ratio": 1.0 - reduction,
    }
