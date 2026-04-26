"""
DeepSeek MoE model wrapper with SparseLinear replacement, aligned with GPT pipeline.
Loads HuggingFace DeepSeek-MoE weights and optionally distills using teacher.

Supported models:
- deepseek-ai/deepseek-moe-16b-base
- deepseek-ai/deepseek-moe-16b-chat

Note: DeepSeek MoE uses a Mixture of Experts architecture with:
- 64 routed experts (2 activated per token)
- 2 shared experts (always activated)
- Total parameters: ~16B, Activated parameters: ~2.8B per token
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# DeepSeek MoE uses custom architecture, need to load with trust_remote_code
from transformers import AutoConfig, AutoModelForCausalLM

from sparse_modeling import SparseLinear
from adamw import AdamW


def _patch_deepseek_moe_dtype(model: nn.Module) -> None:
    """
    Monkey-patch DeepSeek MoE's forward methods to fix dtype mismatch issues.
    
    The original DeepSeek MoE code creates output tensor `y` with default dtype (float32),
    but expert outputs may be in bfloat16/float16, causing:
        RuntimeError: Index put requires the source and destination dtypes match
    
    This patch wraps the MoE layer's forward to ensure consistent dtypes.
    """
    # Find all MoE layers (DeepseekMoE class)
    for name, module in model.named_modules():
        class_name = type(module).__name__
        # DeepSeek MoE uses 'DeepseekMoE' class for MoE layers
        if 'DeepseekMoE' in class_name and not getattr(module, '_dtype_patched', False):
            _patch_moe_forward_inplace(module)
            module._dtype_patched = True
            print(f"[DeepSeek MoE] Patched dtype for {name} ({class_name})")


def _patch_moe_forward_inplace(moe_module: nn.Module) -> None:
    """
    Patch a single DeepseekMoE module to handle dtype consistency.
    
    The issue is that DeepSeek's original code creates intermediate tensors:
        y = torch.empty_like(hidden_states)  # Line ~383 in training mode
        expert_cache = torch.zeros_like(x)   # Line ~397 in inference mode
        y = torch.zeros(...)                  # defaults to float32
        torch.ones(...)                       # defaults to float32 (in Gate's aux_loss)
    
    Problem scenarios:
    1. If hidden_states is float32 but expert weights are bfloat16:
       - y = empty_like(hidden_states) creates float32 tensor
       - expert(hidden_states) returns bfloat16 (due to expert.weight being bfloat16)
       - y[mask] = expert(...) fails: "got Float for destination and BFloat16 for source"
    
    2. FSDP may cast tensors to different dtypes during sharding/gathering
    
    Solution: 
    1. Get the target dtype from expert weights (the ground truth for MoE dtype)
    2. Ensure hidden_states is converted to that dtype before forward
    3. Patch torch.zeros/ones/empty to use that dtype during forward
    4. Convert result back to input dtype if needed
    """
    
    original_forward = moe_module.forward
    
    def patched_forward(hidden_states):
        """
        Patched forward that ensures consistent dtype throughout MoE computation.
        """
        input_dtype = hidden_states.dtype
        input_device = hidden_states.device
        
        # Determine target dtype from expert weights (ground truth for MoE computation)
        # This handles cases where FSDP may have cast hidden_states to a different dtype
        target_dtype = input_dtype
        if hasattr(moe_module, 'experts') and len(moe_module.experts) > 0:
            first_expert = moe_module.experts[0]
            for param in first_expert.parameters():
                target_dtype = param.dtype
                break
        
        # If hidden_states dtype doesn't match expert dtype, convert it
        # This ensures y = empty_like(hidden_states) creates correct dtype tensor
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
        
        # Store original functions
        _original_zeros = torch.zeros
        _original_ones = torch.ones
        _original_empty = torch.empty
        
        def patched_zeros(*args, dtype=None, **kwargs):
            """Patched torch.zeros that defaults to target_dtype instead of float32."""
            if dtype is None:
                dtype = target_dtype
            return _original_zeros(*args, dtype=dtype, **kwargs)
        
        def patched_ones(*args, dtype=None, **kwargs):
            """Patched torch.ones that defaults to target_dtype instead of float32."""
            if dtype is None:
                dtype = target_dtype
            return _original_ones(*args, dtype=dtype, **kwargs)
        
        def patched_empty(*args, dtype=None, **kwargs):
            """Patched torch.empty that defaults to target_dtype instead of float32."""
            if dtype is None:
                dtype = target_dtype
            return _original_empty(*args, dtype=dtype, **kwargs)
        
        # Patch the torch module directly
        torch.zeros = patched_zeros
        torch.ones = patched_ones
        torch.empty = patched_empty
        
        try:
            result = original_forward(hidden_states)
        finally:
            # Always restore original functions
            torch.zeros = _original_zeros
            torch.ones = _original_ones
            torch.empty = _original_empty
        
        # Convert result back to input dtype if needed (for gradient flow)
        if isinstance(result, torch.Tensor) and result.dtype != input_dtype:
            result = result.to(input_dtype)
        
        return result
    
    moe_module.forward = patched_forward


def _replace_linear_with_sparse(module: nn.Module, cfg, skip_expert_layers: bool = False) -> None:
    """
    Recursively replace nn.Linear with SparseLinear, preserving weights/bias.
    
    For MoE models, we have special handling:
    - skip_expert_layers=True: Only sparsify non-expert layers (attention, embeddings)
    - skip_expert_layers=False: Sparsify all layers including expert FFN
    
    Expert layers in DeepSeek MoE are already sparse (only 2 out of 64 experts activated),
    so additional weight pruning on expert layers might be redundant or harmful.
    """
    for name, child in list(module.named_children()):
        # Skip expert layers if requested (they're already dynamically sparse)
        if skip_expert_layers and _is_expert_layer(name, child):
            continue
            
        if isinstance(child, nn.Linear):
            new_linear = SparseLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                sparselinear_config=cfg,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                new_linear.weight.copy_(child.weight)
                if child.bias is not None and new_linear.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
        else:
            _replace_linear_with_sparse(child, cfg, skip_expert_layers)


def _is_expert_layer(name: str, module: nn.Module) -> bool:
    """Check if a module is part of the MoE expert layers."""
    # DeepSeek MoE expert layer naming patterns
    expert_patterns = ['experts', 'expert', 'moe', 'gate']
    name_lower = name.lower()
    for pattern in expert_patterns:
        if pattern in name_lower:
            return True
    # Also check module class name
    class_name = type(module).__name__.lower()
    for pattern in expert_patterns:
        if pattern in class_name:
            return True
    return False


class DeepSeekMoESparse(nn.Module):
    def __init__(
        self,
        config,
        *,
        sparselinear_config=None,
        is_teacher: bool = False,
        model=None,
        sparsify_experts: bool = False,
    ) -> None:
        """
        Initialize DeepSeek MoE sparse wrapper.
        
        Args:
            config: Model configuration
            sparselinear_config: SparseLinear configuration for weight pruning
            is_teacher: If True, don't apply sparsification
            model: Pre-loaded HuggingFace model
            sparsify_experts: If True, also apply weight sparsification to expert layers
                            (default False since experts are already dynamically sparse)
        """
        super().__init__()
        self.config = config
        self.model = model
        self.is_teacher = is_teacher
        self.sparsify_experts = sparsify_experts

        if (not is_teacher) and (sparselinear_config is not None):
            # Replace projection layers with SparseLinear to enable pruning.
            # By default, skip expert layers (they're already sparse via MoE routing)
            _replace_linear_with_sparse(
                self.model.model, 
                sparselinear_config, 
                skip_expert_layers=(not sparsify_experts)
            )

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        # IMPORTANT: Do NOT pass labels to HuggingFace model!
        # HuggingFace's internal loss computation does an additional shift:
        #   shift_logits = logits[..., :-1, :]
        #   shift_labels = labels[..., 1:]
        # But our data pipeline already provides shifted targets (y = x[1:]),
        # so HuggingFace would shift again, causing logits/labels misalignment.
        # Instead, we compute loss manually like GPT-2 does.
        outputs = self.model(
            input_ids=idx,
            labels=None,  # Don't let HuggingFace compute loss
            output_hidden_states=getattr(self.config, 'output_hidden_states', False),
        )
        logits = outputs.logits
        
        # Compute loss manually, matching GPT-2's approach
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Same as GPT-2
            )
        else:
            loss = None
            
        hidden_states = outputs.hidden_states if getattr(self.config, 'output_hidden_states', False) else None
        return logits, loss, hidden_states

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        override_args: Optional[dict] = None,
        sparselinear_config=None,
        is_teacher: bool = False,
    ) -> "DeepSeekMoESparse":
        override_args = override_args or {}
        
        # Load config with trust_remote_code for DeepSeek MoE
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Apply overrides
        if "dropout" in override_args:
            if hasattr(config, 'attention_dropout'):
                config.attention_dropout = override_args["dropout"]
            if hasattr(config, 'hidden_dropout'):
                config.hidden_dropout = override_args["dropout"]
        
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False  # disable cache for training

        # Load HuggingFace model with trust_remote_code
        # attn_implementation 跟随 eager_attention 参数：
        #   eager=True  -> 使用 eager attention，支持 Hutchinson 二阶导数
        #   eager=False -> 使用默认 SDPA/Flash Attention，性能更好
        # Use bfloat16 for training (consistent dtype throughout the model)
        # IMPORTANT: Use device_map="cpu" to ensure model is loaded to CPU first.
        # FSDP will handle sharding and device placement. Loading to GPU here causes OOM.
        attn_impl = "eager" if override_args.get("eager_attention", False) else None
        from_pretrained_kwargs = dict(
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,  # Faster loading, uses less peak memory
            device_map="cpu",  # Force load to CPU, FSDP will handle GPU placement
        )
        if attn_impl is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_impl
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **from_pretrained_kwargs,
        )
        
        # Patch MoE layers to fix dtype mismatch issues
        # DeepSeek's original code may create output tensors with default dtype (float32)
        # while expert outputs are bfloat16, causing index_put_ dtype mismatch errors
        _patch_deepseek_moe_dtype(hf_model)
        
        # Whether to sparsify expert layers (default: no, since they're already sparse via routing)
        sparsify_experts = override_args.get("sparsify_experts", False)
        
        wrapper = cls(
            config,
            sparselinear_config=sparselinear_config,
            is_teacher=is_teacher,
            model=hf_model,
            sparsify_experts=sparsify_experts,
        )
        return wrapper

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, adaptive_l1_decay=0.0):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, adaptive_l1_decay=adaptive_l1_decay)
        return optimizer

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.model, 'model') and hasattr(self.model.model, "embed_tokens"):
            n_params -= self.model.model.embed_tokens.weight.numel()
        return n_params
    
    def get_activated_params(self) -> int:
        """
        Estimate activated parameters per forward pass for MoE model.
        DeepSeek MoE activates 2 out of 64 experts per token, plus shared experts.
        """
        total_params = self.get_num_params()
        
        # Rough estimation: attention + embeddings + (2/64 + 2/64) of expert params
        # This is a simplification; actual activation depends on routing
        # DeepSeek MoE: ~16B total, ~2.8B activated
        activation_ratio = 2.8 / 16.0  # Approximate ratio
        return int(total_params * activation_ratio)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate MFU for MoE model.
        Note: MoE models have different FLOP characteristics due to sparse activation.
        """
        # Use activated params for FLOP estimation
        N = self.get_activated_params()
        cfg = self.config
        L = getattr(cfg, 'num_hidden_layers', 28)
        H = getattr(cfg, 'num_attention_heads', 16)
        hidden_size = getattr(cfg, 'hidden_size', 2048)
        Q = hidden_size // H
        T = getattr(cfg, 'max_position_embeddings', 4096)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_promised = 312e12
        return flops_per_iter * (1.0 / dt) / flops_promised
    
    def tie_weights(self):
        """Tie weights for lm_eval compatibility."""
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
