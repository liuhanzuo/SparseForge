"""
LLaMA model wrapper with SparseLinear replacement, aligned with GPT pipeline.
Loads HuggingFace LLaMA-2 weights and optionally distills using teacher.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from sparse_modeling import SparseLinear
from adamw import AdamW


def _replace_linear_with_sparse(module: nn.Module, cfg) -> None:
    """Recursively replace nn.Linear with SparseLinear, preserving weights/bias."""
    for name, child in list(module.named_children()):
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
            _replace_linear_with_sparse(child, cfg)


class LlamaSparse(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        sparselinear_config=None,
        is_teacher: bool = False,
        model: Optional[LlamaForCausalLM] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model if model is not None else LlamaForCausalLM(config)
        self.is_teacher = is_teacher

        if (not is_teacher) and (sparselinear_config is not None):
            # Replace projection layers with SparseLinear to enable pruning.
            _replace_linear_with_sparse(self.model.model, sparselinear_config)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        # IMPORTANT: Do NOT pass labels to HuggingFace LLaMA!
        # HuggingFace's internal loss computation does an additional shift:
        #   shift_logits = logits[..., :-1, :]
        #   shift_labels = labels[..., 1:]
        # But our data pipeline already provides shifted targets (y = x[1:]),
        # so HuggingFace would shift again, causing logits/labels misalignment.
        # Instead, we compute loss manually like GPT-2 does.
        outputs = self.model(
            input_ids=idx,
            labels=None,  # Don't let HuggingFace compute loss
            output_hidden_states=self.config.output_hidden_states,
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
            
        hidden_states = outputs.hidden_states if self.config.output_hidden_states else None
        return logits, loss, hidden_states

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        override_args: Optional[dict] = None,
        sparselinear_config=None,
        is_teacher: bool = False,
    ) -> "LlamaSparse":
        override_args = override_args or {}
        # Load base config and apply overrides (dropout, gradient checkpointing, etc.).
        config = LlamaConfig.from_pretrained(model_name)
        if "dropout" in override_args:
            config.hidden_dropout = override_args["dropout"]
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        # 不再通过 config.gradient_checkpointing 启用（新版 transformers 4.50+ 已重构）。
        # 改为在模型加载后通过 model.gradient_checkpointing_enable() 方法启用，
        # 并显式指定 use_reentrant=False 以确保与 FSDP 兼容。
        config.use_cache = False  # disable cache for training

        # Load HuggingFace weights first (dense), then replace Linear->SparseLinear in-place.
        # This avoids strict state_dict loading errors due to SparseLinear extra buffers.
        # attn_implementation 跟随 eager_attention 参数：
        #   eager=True  -> 使用 eager attention，支持 Hutchinson 二阶导数
        #   eager=False -> 使用默认 SDPA/Flash Attention，性能更好
        # IMPORTANT: Do NOT use device_map="cpu" here! HuggingFace's device_map injects
        # accelerate dispatch hooks (AlignDevicesHook) into model submodules, which are
        # incompatible with FSDP's own parameter shard/unshard/device management.
        # With low_cpu_mem_usage=True, the model loads to CPU by default without hooks.
        # FSDP will handle sharding and device placement.
        attn_impl = "eager" if override_args.get("eager_attention", False) else None
        from_pretrained_kwargs = dict(
            config=config,
            low_cpu_mem_usage=True,  # Faster loading, uses less peak memory
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if attn_impl is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_impl
        hf_model = LlamaForCausalLM.from_pretrained(
            model_name, 
            **from_pretrained_kwargs,
        )
        
        # 如果需要 gradient checkpointing，通过模型方法启用
        # IMPORTANT: Teacher 不需要 gradient checkpointing（只在 no_grad 下做 inference）。
        # 在 FSDP 下，gradient_checkpointing hooks 会导致额外的 all_gather 操作，
        # 与 no_grad 环境结合可能引发 NCCL 集合操作不匹配和死锁。
        # gradient_checkpointing 通过模型方法启用，使用 use_reentrant=False。
        # IMPORTANT: use_reentrant=True（旧版默认）在 FSDP 下会导致 NCCL 集合操作不匹配，
        # 因为 reentrant checkpoint 在 backward recompute 时不保证所有 rank 以相同顺序
        # 执行 all_gather，导致 SeqNum 偏移 → 600s 超时死锁。
        # IMPORTANT: Teacher 不需要 gradient checkpointing（只在 no_grad 下做 inference）。
        if override_args.get("gradient_checkpointing", False) and not is_teacher:
            hf_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        wrapper = cls(
            config,
            sparselinear_config=sparselinear_config,
            is_teacher=is_teacher,
            model=hf_model,
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
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.model.model, "embed_tokens"):
            n_params -= self.model.model.embed_tokens.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        # Approximate using GPT-style formula; LLaMA arch specifics are similar.
        N = self.get_num_params()
        cfg = self.config
        L = cfg.num_hidden_layers
        H = cfg.num_attention_heads
        Q = cfg.hidden_size // cfg.num_attention_heads
        T = cfg.max_position_embeddings
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_promised = 312e12
        return flops_per_iter * (1.0 / dt) / flops_promised
    
    def tie_weights(self):
        """Tie weights for lm_eval compatibility."""
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
