"""
OPT model wrapper with SparseLinear replacement, aligned with LLaMA/GPT pipeline.
Loads HuggingFace OPT weights and optionally distills using teacher.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import OPTConfig, OPTForCausalLM

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


class OPTSparse(nn.Module):
    def __init__(
        self,
        config: OPTConfig,
        *,
        sparselinear_config=None,
        is_teacher: bool = False,
        model: Optional[OPTForCausalLM] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model if model is not None else OPTForCausalLM(config)
        self.is_teacher = is_teacher

        if (not is_teacher) and (sparselinear_config is not None):
            # Replace projection layers with SparseLinear to enable pruning.
            # OPT 的 transformer 层在 model.model.decoder 下
            _replace_linear_with_sparse(self.model.model.decoder, sparselinear_config)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        # IMPORTANT: Do NOT pass labels to HuggingFace OPT!
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
    ) -> "OPTSparse":
        override_args = override_args or {}
        # Load base config and apply overrides (dropout, gradient checkpointing, etc.).
        config = OPTConfig.from_pretrained(model_name)
        if "dropout" in override_args:
            # OPT 使用 dropout 和 attention_dropout
            config.dropout = override_args["dropout"]
            config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        # OPT 不支持原生 gradient_checkpointing 配置，需要通过模型方法启用
        config.use_cache = False  # disable cache for training

        # Load HuggingFace weights first (dense), then replace Linear->SparseLinear in-place.
        # This avoids strict state_dict loading errors due to SparseLinear extra buffers.
        # Use low_cpu_mem_usage=True for faster loading on shared filesystems (cephfs/NFS)
        hf_model = OPTForCausalLM.from_pretrained(
            model_name, 
            config=config,
            low_cpu_mem_usage=True,  # Faster loading, uses less peak memory
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # Load in target dtype directly
        )
        
        # 如果需要 gradient checkpointing，通过模型方法启用
        # IMPORTANT: Teacher 不需要 gradient checkpointing（只在 no_grad 下做 inference）。
        # 在 FSDP 下，gradient_checkpointing hooks 会导致额外的 all_gather 操作，
        # 与 no_grad 环境结合可能引发 NCCL 集合操作不匹配和死锁。
        # gradient_checkpointing 通过模型方法启用，使用 use_reentrant=False。
        # IMPORTANT: use_reentrant=True（旧版默认）在 FSDP 下会导致 NCCL 集合操作不匹配。
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
        # OPT 的 embedding 在 model.model.decoder.embed_tokens
        if non_embedding and hasattr(self.model.model.decoder, "embed_tokens"):
            n_params -= self.model.model.decoder.embed_tokens.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        # Approximate using GPT-style formula; OPT arch specifics are similar.
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
