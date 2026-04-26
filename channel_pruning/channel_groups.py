"""
Channel Groups: Define and extract MLP channel groups from Transformer layers.

For each Transformer layer's MLP (FFN), a channel group consists of:
- up_proj.weight[j, :] (output channel j)
- gate_proj.weight[j, :] (output channel j)  
- down_proj.weight[:, j] (input channel j)

These must be pruned together to maintain structural consistency.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn


class MLPProjections(NamedTuple):
    """Container for MLP projection layers."""
    up_proj: nn.Linear
    gate_proj: nn.Linear
    down_proj: nn.Linear
    layer_idx: int


@dataclass
class MLPChannelGroup:
    """Represents a single channel group in an MLP layer.
    
    A channel group contains weight slices that must be pruned together:
    - up_proj: row j (output channel)
    - gate_proj: row j (output channel)
    - down_proj: column j (input channel)
    """
    layer_idx: int
    channel_idx: int
    intermediate_size: int
    hidden_size: int
    
    def get_weight_slices(
        self, 
        up_weight: torch.Tensor,
        gate_weight: torch.Tensor, 
        down_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract weight slices for this channel.
        
        Args:
            up_weight: [intermediate_size, hidden_size]
            gate_weight: [intermediate_size, hidden_size]
            down_weight: [hidden_size, intermediate_size]
            
        Returns:
            up_slice: [hidden_size] - row j of up_proj
            gate_slice: [hidden_size] - row j of gate_proj
            down_slice: [hidden_size] - column j of down_proj
        """
        j = self.channel_idx
        up_slice = up_weight[j, :]      # [hidden_size]
        gate_slice = gate_weight[j, :]  # [hidden_size]
        down_slice = down_weight[:, j]  # [hidden_size]
        return up_slice, gate_slice, down_slice


def get_mlp_projections(model: nn.Module, model_type: str = "qwen") -> List[MLPProjections]:
    """Extract MLP projection layers from a model.
    
    Supports different model architectures:
    - qwen/qwen2/qwen3: model.model.layers[i].mlp.{up_proj, gate_proj, down_proj}
    - llama: model.model.layers[i].mlp.{up_proj, gate_proj, down_proj}
    - opt: model.model.decoder.layers[i].fc1/fc2 (different structure)
    - mistral: model.model.layers[i].mlp.{up_proj, gate_proj, down_proj}
    
    Args:
        model: The model to extract projections from
        model_type: Type of model architecture
        
    Returns:
        List of MLPProjections for each layer
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
    
    # Further unwrap model-specific wrappers (e.g. QwenSparse -> HF -> transformer)
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
        base_model = base_model.model.model
    elif hasattr(base_model, 'model'):
        base_model = base_model.model
    
    if model_type in ["qwen", "qwen2", "qwen3", "llama", "mistral"]:
        # All these use similar structure with SwiGLU MLP
        if hasattr(base_model, 'layers'):
            layers = base_model.layers
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        else:
            raise ValueError(f"Cannot find layers in model for type {model_type}")
        
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            
            # Different naming conventions
            if hasattr(mlp, 'up_proj'):
                up_proj = mlp.up_proj
                gate_proj = mlp.gate_proj
                down_proj = mlp.down_proj
            elif hasattr(mlp, 'w1'):  # Some implementations use w1/w2/w3
                up_proj = mlp.w1
                gate_proj = mlp.w3
                down_proj = mlp.w2
            else:
                raise ValueError(f"Unknown MLP structure in layer {i}")
            
            projections.append(MLPProjections(
                up_proj=up_proj,
                gate_proj=gate_proj,
                down_proj=down_proj,
                layer_idx=i
            ))
    
    elif model_type == "opt":
        # OPT uses fc1/fc2 without gating
        if hasattr(base_model, 'decoder'):
            layers = base_model.decoder.layers
        else:
            layers = base_model.layers
            
        for i, layer in enumerate(layers):
            # OPT doesn't have gate_proj, treat fc1 as both up and gate
            fc1 = layer.fc1
            fc2 = layer.fc2
            
            # Create a dummy gate_proj that shares weights with fc1
            projections.append(MLPProjections(
                up_proj=fc1,
                gate_proj=fc1,  # OPT: no separate gate
                down_proj=fc2,
                layer_idx=i
            ))
    
    elif model_type == "gpt2":
        # GPT-2 uses c_fc/c_proj without gating (GELU activation, not SwiGLU)
        # Structure: model.transformer.h[i].mlp.{c_fc, c_proj}
        # c_fc: [n_embd] -> [4*n_embd]  (类似 up_proj，无 gate)
        # c_proj: [4*n_embd] -> [n_embd] (类似 down_proj)
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
        elif hasattr(base_model, 'h'):
            layers = base_model.h
        else:
            raise ValueError(f"Cannot find transformer blocks in GPT-2 model")
        
        for i, block in enumerate(layers):
            mlp = block.mlp
            c_fc = mlp.c_fc
            c_proj = mlp.c_proj
            
            # GPT-2 没有 gate_proj，用 c_fc 同时充当 up 和 gate
            projections.append(MLPProjections(
                up_proj=c_fc,
                gate_proj=c_fc,   # GPT-2: no separate gate
                down_proj=c_proj,
                layer_idx=i
            ))
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return projections


def get_intermediate_size(model: nn.Module, model_type: str = "qwen") -> int:
    """Get the intermediate (FFN hidden) size from model config or weights.
    
    Args:
        model: The model
        model_type: Type of model architecture
        
    Returns:
        The intermediate size (number of channels in FFN hidden layer)
    """
    # Unwrap FSDP/DDP first
    def _unwrap(m):
        if hasattr(m, '_fsdp_wrapped_module'):
            return _unwrap(m._fsdp_wrapped_module)
        if hasattr(m, 'module'):
            return _unwrap(m.module)
        return m
    
    base = _unwrap(model)
    
    # Try to get from config first
    config = None
    if hasattr(base, 'config'):
        config = base.config
    elif hasattr(base, 'model') and hasattr(base.model, 'config'):
        config = base.model.config
    
    if config is not None:
        if hasattr(config, 'intermediate_size'):
            return config.intermediate_size
        elif hasattr(config, 'ffn_dim'):
            return config.ffn_dim
        elif hasattr(config, 'n_inner') and config.n_inner is not None:
            return config.n_inner
        elif hasattr(config, 'n_embd'):
            # GPT-2: intermediate_size = 4 * n_embd
            return 4 * config.n_embd
    
    # Fallback: get from first MLP layer weights
    # 注意：FSDP 分片后 weight shape 可能不正确，不应依赖此路径
    projections = get_mlp_projections(model, model_type)
    if projections:
        w = projections[0].up_proj.weight
        # FSDP 可能将 weight 扁平化分片，检查是否合理
        if w.dim() == 2 and w.shape[0] > 0:
            return w.shape[0]
    
    raise ValueError("Cannot determine intermediate_size from model")


def get_hidden_size(model: nn.Module, model_type: str = "qwen") -> int:
    """Get the hidden (model) size from model config or weights.
    
    Args:
        model: The model
        model_type: Type of model architecture
        
    Returns:
        The hidden size (model dimension)
    """
    # Unwrap FSDP/DDP first
    def _unwrap(m):
        if hasattr(m, '_fsdp_wrapped_module'):
            return _unwrap(m._fsdp_wrapped_module)
        if hasattr(m, 'module'):
            return _unwrap(m.module)
        return m
    
    base = _unwrap(model)
    
    config = None
    if hasattr(base, 'config'):
        config = base.config
    elif hasattr(base, 'model') and hasattr(base.model, 'config'):
        config = base.model.config
    
    if config is not None:
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        elif hasattr(config, 'd_model'):
            return config.d_model
        elif hasattr(config, 'n_embd'):
            # GPT-2: hidden_size = n_embd
            return config.n_embd
    
    # Fallback: get from first MLP layer weights
    # 注意：FSDP 分片后 weight shape 可能不正确，不应依赖此路径
    projections = get_mlp_projections(model, model_type)
    if projections:
        w = projections[0].up_proj.weight
        if w.dim() == 2 and w.shape[1] > 0:
            return w.shape[1]
    
    raise ValueError("Cannot determine hidden_size from model")


def get_num_layers(model: nn.Module, model_type: str = "qwen") -> int:
    """Get the number of transformer layers.
    
    Args:
        model: The model
        model_type: Type of model architecture
        
    Returns:
        Number of transformer layers
    """
    projections = get_mlp_projections(model, model_type)
    return len(projections)


class MLPChannelGroupManager:
    """Manager for all MLP channel groups in a model.
    
    Provides utilities for:
    - Iterating over all channel groups
    - Getting/setting channel masks
    - Computing aggregate statistics
    """
    
    def __init__(self, model: nn.Module, model_type: str = "qwen"):
        self.model = model
        self.model_type = model_type
        self.projections = get_mlp_projections(model, model_type)
        self.intermediate_size = get_intermediate_size(model, model_type)
        self.hidden_size = get_hidden_size(model, model_type)
        self.num_layers = len(self.projections)
    
    def get_all_channel_groups(self) -> List[MLPChannelGroup]:
        """Get all channel groups across all layers."""
        groups = []
        for layer_idx in range(self.num_layers):
            for channel_idx in range(self.intermediate_size):
                groups.append(MLPChannelGroup(
                    layer_idx=layer_idx,
                    channel_idx=channel_idx,
                    intermediate_size=self.intermediate_size,
                    hidden_size=self.hidden_size
                ))
        return groups
    
    def get_layer_channel_groups(self, layer_idx: int) -> List[MLPChannelGroup]:
        """Get all channel groups for a specific layer."""
        groups = []
        for channel_idx in range(self.intermediate_size):
            groups.append(MLPChannelGroup(
                layer_idx=layer_idx,
                channel_idx=channel_idx,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size
            ))
        return groups
    
    def get_projection(self, layer_idx: int) -> MLPProjections:
        """Get the MLP projections for a specific layer."""
        return self.projections[layer_idx]
    
    def total_channels(self) -> int:
        """Total number of channels across all layers."""
        return self.num_layers * self.intermediate_size
    
    def __repr__(self) -> str:
        return (f"MLPChannelGroupManager(num_layers={self.num_layers}, "
                f"intermediate_size={self.intermediate_size}, "
                f"hidden_size={self.hidden_size}, "
                f"total_channels={self.total_channels()})")
