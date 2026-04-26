"""
Hunyuan model wrapper with SparseLinear replacement, aligned with GPT pipeline.
Loads HuggingFace Hunyuan weights and optionally distills using teacher.

Supported models:
- tencent/Hunyuan-1.8B-Pretrain
- tencent/Hunyuan-1.8B (and other Hunyuan variants)

Note: Hunyuan uses LLaMA architecture internally (model_type="llama", architectures=["LlamaForCausalLM"]),
but we use AutoConfig/AutoModelForCausalLM with trust_remote_code=True
to support any Hunyuan-specific customizations.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from sparse_modeling import SparseLinear
from adamw import AdamW


def _replace_linear_with_sparse(module: nn.Module, cfg) -> None:
    """递归替换 nn.Linear 为 SparseLinear，保留原始权重/偏置。"""
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


class HunyuanSparse(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        sparselinear_config=None,
        is_teacher: bool = False,
        model: Optional[PreTrainedModel] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model if model is not None else AutoModelForCausalLM.from_config(config)
        self.is_teacher = is_teacher

        if (not is_teacher) and (sparselinear_config is not None):
            # 替换投影层为 SparseLinear 以启用剪枝。
            # Hunyuan 内部结构与 LLaMA 一致，model.model 即为 transformer body
            _replace_linear_with_sparse(self.model.model, sparselinear_config)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        # IMPORTANT: 不要传 labels 给 HuggingFace 模型！
        # HuggingFace 内部的 loss 计算会做额外的 shift：
        #   shift_logits = logits[..., :-1, :]
        #   shift_labels = labels[..., 1:]
        # 但我们的数据管线已经提供了 shifted targets (y = x[1:])，
        # 所以 HuggingFace 会再次 shift，导致 logits/labels 不对齐。
        # 因此我们像 GPT-2 一样手动计算 loss。
        outputs = self.model(
            input_ids=idx,
            labels=None,  # 不让 HuggingFace 计算 loss
            output_hidden_states=self.config.output_hidden_states,
        )
        logits = outputs.logits
        
        # 手动计算 loss，与 GPT-2 方式一致
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # 与 GPT-2 相同
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
    ) -> "HunyuanSparse":
        override_args = override_args or {}
        # 加载基础 config 并应用覆盖参数（dropout, gradient checkpointing 等）。
        # 使用 AutoConfig 以支持 Hunyuan 的任何自定义配置
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "dropout" in override_args:
            # Hunyuan 基于 LLaMA，使用相同的 dropout 属性名
            if hasattr(config, 'hidden_dropout'):
                config.hidden_dropout = override_args["dropout"]
            if hasattr(config, 'attention_dropout'):
                config.attention_dropout = override_args["dropout"]
        config.output_hidden_states = bool(override_args.get("output_hidden_state", False))
        config.use_cache = False  # 训练时禁用 cache

        # 先加载 HuggingFace 权重（dense），然后原地替换 Linear->SparseLinear。
        # 这样可以避免由于 SparseLinear 额外 buffer 导致的严格 state_dict 加载错误。
        # attn_implementation 跟随 eager_attention 参数：
        #   eager=True  -> 使用 eager attention，支持 Hutchinson 二阶导数
        #   eager=False -> 使用默认 SDPA/Flash Attention，性能更好
        # IMPORTANT: 模型默认加载到 CPU（不需要 device_map）。
        # FSDP 会处理 sharding 和设备放置。
        attn_impl = "eager" if override_args.get("eager_attention", False) else None
        from_pretrained_kwargs = dict(
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 更快加载，使用更少峰值内存
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if attn_impl is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_impl
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **from_pretrained_kwargs,
        )
        
        # 如果需要 gradient checkpointing，通过模型方法启用
        # IMPORTANT: Teacher 不需要 gradient checkpointing（只在 no_grad 下做 inference）。
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
        if non_embedding and hasattr(self.model.model, "embed_tokens"):
            n_params -= self.model.model.embed_tokens.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        # 使用 GPT 风格的近似公式；Hunyuan/LLaMA 架构类似。
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
