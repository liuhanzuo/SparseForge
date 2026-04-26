"""
Channel Mask: Manage channel-level masks for FFN pruning.

Each layer has a mask vector of shape [intermediate_size].
The mask follows HASAST's soft-to-hard mechanism:
1. Start with soft masks in [0, 1]
2. Periodically update based on importance scores
3. Gradually harden to binary {0, 1}

Masks are NOT learnable parameters - they are updated in the outer loop
based on importance scores, not via gradient descent.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None

from .config import ChannelPruningConfig
from .channel_groups import get_mlp_projections, get_intermediate_size, MLPProjections


@dataclass
class LayerMaskState:
    """Mask state for a single layer."""
    layer_idx: int
    mask: torch.Tensor  # [intermediate_size], values in [0, 1]
    frozen: torch.Tensor  # [intermediate_size], bool - frozen channels don't update
    temperature: float  # Current temperature for soft gating
    hardening_x: float  # Hardening progress: 1.0 = fully soft, 0.0 = fully hard


class ChannelMaskState:
    """Manages channel masks for all layers.
    
    Each layer has an independent mask vector that controls which
    FFN channels are kept (mask=1) or pruned (mask=0).
    
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
        
        # Extract model info
        self.projections = get_mlp_projections(model, config.model_type)
        self.num_layers = len(self.projections)
        
        # 从 config 获取 intermediate size（不从 FSDP 分片后的 weight shape 取，那会不正确）
        self.intermediate_size = get_intermediate_size(model, config.model_type)
        
        # Initialize masks - all 1s (keep all channels initially)
        self.masks: Dict[int, LayerMaskState] = {}
        for layer_idx in range(self.num_layers):
            self.masks[layer_idx] = LayerMaskState(
                layer_idx=layer_idx,
                mask=torch.ones(self.intermediate_size, device=self.device),
                frozen=torch.zeros(self.intermediate_size, dtype=torch.bool, device=self.device),
                temperature=config.temp_init,
                hardening_x=1.0  # Start fully soft
            )
        
        # Per-layer keep ratios
        if config.per_layer_keep_ratio is not None:
            self.per_layer_keep_ratio = config.per_layer_keep_ratio
        else:
            # Same ratio for all layers
            self.per_layer_keep_ratio = {i: config.ffn_keep_ratio for i in range(self.num_layers)}
    
    def get_keep_k(self, layer_idx: int) -> int:
        """Get the number of channels to keep for a layer."""
        ratio = self.per_layer_keep_ratio.get(layer_idx, self.config.ffn_keep_ratio)
        return int(round(self.intermediate_size * ratio))
    
    def get_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the current mask for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Mask tensor [intermediate_size] with values in [0, 1]
        """
        return self.masks[layer_idx].mask
    
    def get_hard_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the hard (binary) mask based on top-K scores.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Binary mask [intermediate_size] with exactly keep_k ones
        """
        mask = self.masks[layer_idx].mask
        keep_k = self.get_keep_k(layer_idx)
        
        # Top-K selection based on mask values
        _, top_indices = torch.topk(mask, k=keep_k, dim=0, largest=True)
        hard_mask = torch.zeros_like(mask)
        hard_mask.scatter_(0, top_indices, 1.0)
        
        return hard_mask
    
    def get_effective_mask(self, layer_idx: int) -> torch.Tensor:
        """Get the effective mask (blend of soft and hard based on hardening_x).
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Effective mask for forward pass
        """
        state = self.masks[layer_idx]
        soft_mask = state.mask
        hard_mask = self.get_hard_mask(layer_idx)
        
        # Convex combination: x * soft + (1-x) * hard
        hx = state.hardening_x
        return hx * soft_mask + (1 - hx) * hard_mask
    
    @torch.no_grad()
    def compute_target_mask(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """Compute target mask from importance scores.
        
        Uses soft gating with temperature annealing:
        G[j] = sigmoid((score[j] - tau) / T)
        where tau is the threshold for top-K selection.
        
        Args:
            layer_idx: Layer index
            scores: Channel importance scores [intermediate_size]
            step: Current training step
            
        Returns:
            Target mask G [intermediate_size]
        """
        state = self.masks[layer_idx]
        keep_k = self.get_keep_k(layer_idx)
        
        # Effective sparsity with warmup
        warmup = self.config.sparsity_warmup_steps
        if step < warmup:
            # Linear warmup: gradually increase sparsity
            progress = step / max(1, warmup)
            effective_keep_k = int(self.intermediate_size - progress * (self.intermediate_size - keep_k))
        else:
            effective_keep_k = keep_k
        
        effective_keep_k = max(1, min(self.intermediate_size, effective_keep_k))
        
        # Normalize scores
        scores_float = scores.float()
        mu = scores_float.mean()
        sigma = scores_float.std() + 1e-6
        scores_norm = (scores_float - mu) / sigma
        
        # Get threshold (score of k-th highest element)
        sorted_scores, _ = torch.sort(scores_norm, descending=True)
        tau = sorted_scores[effective_keep_k - 1]
        
        # Soft gating with temperature
        T = state.temperature
        G = torch.sigmoid((scores_norm - tau) / max(T, 1e-8))
        
        # Respect frozen channels
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
        
        Following HASAST mechanism:
        1. Compute target G from scores
        2. EMA update: mask <- (1-lr) * mask + lr * G
        3. Update temperature (decay)
        4. Update hardening_x (progress toward binary)
        
        Args:
            layer_idx: Layer index
            scores: Channel importance scores [intermediate_size]
            step: Current training step
        """
        state = self.masks[layer_idx]
        config = self.config
        
        # Skip if not time to update
        if step % config.mask_update_period != 0:
            return
        
        # Skip during warmup
        if step < config.sparsity_warmup_steps:
            return
        
        # Compute target mask
        G = self.compute_target_mask(layer_idx, scores, step)
        
        # EMA update
        lr = config.mask_lr
        state.mask.mul_(1 - lr).add_(G, alpha=lr).clamp_(0.0, 1.0)
        
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
        """Get sparsity statistics for logging.
        
        Returns:
            Dict with avg_mask, min_mask, max_mask, hard_sparsity
        """
        all_masks = [self.masks[i].mask for i in range(self.num_layers)]
        stacked = torch.stack(all_masks)
        
        # Soft mask stats
        avg_mask = stacked.mean().item()
        min_mask = stacked.min().item()
        max_mask = stacked.max().item()
        
        # Hard sparsity (based on threshold)
        hard_masks = [(m > 0.5).float() for m in all_masks]
        hard_stacked = torch.stack(hard_masks)
        hard_sparsity = 1.0 - hard_stacked.mean().item()
        
        # Per-layer stats
        per_layer_sparsity = {
            f"layer_{i}_sparsity": 1.0 - (self.masks[i].mask > 0.5).float().mean().item()
            for i in range(self.num_layers)
        }
        
        return {
            "avg_mask": avg_mask,
            "min_mask": min_mask,
            "max_mask": max_mask,
            "hard_sparsity": hard_sparsity,
            **per_layer_sparsity
        }
    
    def finalize_masks(self):
        """Finalize masks to binary (for export).
        
        Converts all masks to hard binary based on top-K.
        """
        for layer_idx in range(self.num_layers):
            hard_mask = self.get_hard_mask(layer_idx)
            self.masks[layer_idx].mask = hard_mask
            self.masks[layer_idx].hardening_x = 0.0


class ChannelRegressionRecovery:
    """在 channel mask hardening 时执行线性回归恢复。
    
    SlimLLM 的 post-pruning recovery:
      min_{A, B} ||Y_orig - (diag(A) * Y_pruned + B)||^2
    
    对每一层 MLP，收集原始输出 Y_orig 和剪枝后输出 Y_pruned，
    然后逐维度做最小二乘拟合 a_d, b_d:
      Y_orig[:, d] ≈ a_d * Y_pruned[:, d] + b_d
    
    拟合完成后，将 a_d 吸收进 down_proj 的权重，b_d 吸收进偏置。
    """
    
    def __init__(
        self,
        model: nn.Module,
        mask_state: 'ChannelMaskState',
        config: ChannelPruningConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.mask_state = mask_state
        self.config = config
        self.device = device or next(model.parameters()).device
        self.projections = get_mlp_projections(model, config.model_type)
        self.num_layers = len(self.projections)
        
        # 检测 FSDP：在 FSDP 下 weight 被 flatten + 分片为 1D tensor，
        # 必须用 summon_full_params 恢复完整形状才能做 in-place 修改
        self._fsdp_target = self._detect_fsdp_target(model)
        
        # 收集的校准数据: {layer_idx: (Y_orig_list, Y_pruned_list)}
        self._calibration_data: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._hooks: List = []
        self._collecting = False
        # 记录上次执行 regression 时的 hardening_x，避免重复执行
        self._last_regression_hx: Dict[int, float] = {i: 1.0 for i in range(self.num_layers)}
    
    @staticmethod
    def _detect_fsdp_target(model: nn.Module):
        """检测模型是否被 FSDP 包裹，返回用于 summon_full_params 的 target。"""
        if FSDP is None:
            return None
        # Distill_Model 结构: model.student 是 FSDP 包裹的模块
        _raw = model.module if hasattr(model, 'module') else model
        if hasattr(_raw, 'student') and isinstance(_raw.student, FSDP):
            return _raw.student
        if isinstance(model, FSDP):
            return model
        if isinstance(_raw, FSDP):
            return _raw
        return None
    
    def _summon_full_params_ctx(self):
        """返回 summon_full_params context manager（FSDP 下）或 nullcontext。"""
        if self._fsdp_target is not None and FSDP is not None:
            return FSDP.summon_full_params(
                self._fsdp_target,
                writeback=True,
                recurse=True,
                offload_to_cpu=False,
            )
        return nullcontext()

    @torch.no_grad()
    def collect_calibration_step(self, layer_idx: int, Y_orig: torch.Tensor, Y_pruned: torch.Tensor):
        """收集一步校准数据。
        
        Args:
            layer_idx: 层索引
            Y_orig: 原始 MLP 输出 [N, D]
            Y_pruned: 剪枝后 MLP 输出 [N, D]
        """
        if layer_idx not in self._calibration_data:
            self._calibration_data[layer_idx] = ([], [])
        
        orig_list, pruned_list = self._calibration_data[layer_idx]
        # 只保留最近几批数据，避免内存过大
        max_batches = 8
        orig_list.append(Y_orig.detach().reshape(-1, Y_orig.shape[-1]).float().cpu())
        pruned_list.append(Y_pruned.detach().reshape(-1, Y_pruned.shape[-1]).float().cpu())
        if len(orig_list) > max_batches:
            orig_list.pop(0)
            pruned_list.pop(0)
    
    @torch.no_grad()
    def fit_regression(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """对层 layer_idx 执行线性回归拟合。
        
        min_{A, B} ||Y_orig - (diag(A) * Y_pruned + B)||^2
        
        逐维度独立拟合:
          a_d = cov(y_orig_d, y_pruned_d) / var(y_pruned_d)
          b_d = mean(y_orig_d) - a_d * mean(y_pruned_d)
        
        Returns:
            (A, B): A ∈ R^D 缩放因子, B ∈ R^D 偏置; 或 None
        """
        if layer_idx not in self._calibration_data:
            return None
        
        orig_list, pruned_list = self._calibration_data[layer_idx]
        if len(orig_list) == 0:
            return None
        
        Y_orig = torch.cat(orig_list, dim=0)    # [N_total, D]
        Y_pruned = torch.cat(pruned_list, dim=0)  # [N_total, D]
        
        D = Y_orig.shape[1]
        
        # 逐维度最小二乘
        mean_orig = Y_orig.mean(dim=0)      # [D]
        mean_pruned = Y_pruned.mean(dim=0)   # [D]
        
        Y_orig_c = Y_orig - mean_orig.unsqueeze(0)
        Y_pruned_c = Y_pruned - mean_pruned.unsqueeze(0)
        
        # cov(orig, pruned) 和 var(pruned)，逐维度
        cov_op = (Y_orig_c * Y_pruned_c).mean(dim=0)  # [D]
        var_p = (Y_pruned_c * Y_pruned_c).mean(dim=0)  # [D]
        
        eps = 1e-8
        A = cov_op / (var_p + eps)  # [D]
        B = mean_orig - A * mean_pruned  # [D]
        
        # 对 A 做 clamp，防止数值不稳定
        A = A.clamp(0.1, 10.0)
        
        return A.to(self.device), B.to(self.device)
    
    @torch.no_grad()
    def apply_regression_to_weights(self, layer_idx: int, A: torch.Tensor, B: torch.Tensor):
        """将回归系数吸收进 down_proj 的权重和偏置。
        
        Y_corrected = diag(A) * (W_down @ intermediate) + B
                    = (diag(A) @ W_down) @ intermediate + B
        
        所以:
          W_down_new = diag(A) @ W_down
          bias_new   = B (如果没有 bias 就新建一个)
        
        Args:
            layer_idx: 层索引
            A: 缩放因子 [D]
            B: 偏置 [D]
        """
        proj = self.projections[layer_idx]
        down_weight = proj.down_proj.weight  # [D, H]
        
        # W_down_new[d, :] = A[d] * W_down[d, :]
        down_weight.data.mul_(A.unsqueeze(1))
        
        # 处理 bias
        if proj.down_proj.bias is not None:
            proj.down_proj.bias.data.mul_(A).add_(B)
        else:
            # 如果模型本身没有 bias，创建一个并注册
            # 注意：大多数 LLM 的 linear 没有 bias，这里只做 weight 修正
            # B 的效果通过 residual connection 在后续训练中自然补偿
            pass
    
    @torch.no_grad()
    def maybe_run_regression(self, step: int, master_process: bool = False) -> bool:
        """检查是否需要在当前步执行回归恢复。
        
        触发条件：hardening_x 发生了显著变化（从 > 0.5 降到 <= 0.5，
        即 mask 从以 soft 为主变为以 hard 为主）。
        
        Args:
            step: 当前训练步
            master_process: 是否是主进程（用于日志输出）
            
        Returns:
            是否执行了回归
        """
        if not self.config.pca_regression_on_hardening:
            return False
        
        # 先收集所有需要做 regression 的 (layer_idx, A, B)，再统一 apply
        # 这样在 FSDP 下只需 summon 一次全参数，避免多次通信
        pending_regressions = []
        for layer_idx in range(self.num_layers):
            state = self.mask_state.masks[layer_idx]
            last_hx = self._last_regression_hx.get(layer_idx, 1.0)
            cur_hx = state.hardening_x
            
            # 当 hardening_x 跨过 0.5 阈值时触发回归
            if last_hx > 0.5 and cur_hx <= 0.5:
                result = self.fit_regression(layer_idx)
                if result is not None:
                    A, B = result
                    pending_regressions.append((layer_idx, A, B))
                self._last_regression_hx[layer_idx] = cur_hx
            elif cur_hx != last_hx:
                self._last_regression_hx[layer_idx] = cur_hx
        
        if not pending_regressions:
            return False
        
        # 在一次 summon_full_params 中完成所有 layer 的权重修改
        # FSDP 下 weight 被 flatten + 分片为 1D，必须 summon 才能拿到完整 [D, H] 形状
        with self._summon_full_params_ctx():
            for layer_idx, A, B in pending_regressions:
                self.apply_regression_to_weights(layer_idx, A, B)
                if master_process:
                    print(f"[Channel Regression] Layer {layer_idx}: "
                          f"A mean={A.mean().item():.4f}, B mean={B.mean().item():.4f}")
        
        return True
    
    def clear_calibration_data(self):
        """清除校准数据以释放内存。"""
        self._calibration_data.clear()


class LoRABypassModule(nn.Module):
    """Channel Pruning LoRA Bypass: 从 hidden_states 直接旁路到 MLP 输出。
    
    用于补偿 channel pruning 带来的 FFN 容量损失。
    
    结构:
        x ─┬── [pruned MLP] ──> y_pruned ──┬── + ──> y_final
           │                                 │
           └── LoRA_A (D→r) ──> LoRA_B (r→D) ┘
    
    初始化:
        - LoRA_A: kaiming uniform
        - LoRA_B: 全 0 初始化（保证训练初期 bypass 输出为 0）
    
    训练结束后可以把 LoRA_B @ LoRA_A 合并进 down_proj（仅当 down_proj 
    的输入 == hidden_states 时才成立，channel pruning 场景下不能直接合并，
    但可以作为独立适配器保留）。
    
    Args:
        hidden_size: 模型隐藏维度 D
        rank: LoRA 低秩维度 r
        alpha: 缩放因子 (output = alpha/rank * LoRA_B(LoRA_A(x)))
        dropout: dropout 比率
    """
    
    def __init__(self, hidden_size: int, rank: int = 64, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA_A: D -> r
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        # LoRA_B: r -> D
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # 初始化: A 用 kaiming，B 全零（初始输出为 0）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, D] 原始 hidden_states
            
        Returns:
            lora_out: [batch, seq, D] LoRA bypass 输出
        """
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class ChannelMaskApplier:
    """Applies channel masks to MLP forward pass.
    
    During training, masks are applied as element-wise multiplication
    on the intermediate activations:
    
    Original: y = down_proj(silu(gate_proj(x)) * up_proj(x))
    Masked:   y = down_proj(silu(gate_proj(x)) * up_proj(x) * mask)
    
    The mask zeros out pruned channels, ensuring they don't contribute
    to the output while allowing gradient flow for importance estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mask_state: ChannelMaskState,
        config: ChannelPruningConfig
    ):
        self.model = model
        self.mask_state = mask_state
        self.config = config
        self.projections = get_mlp_projections(model, config.model_type)
        self.hooks = []
        self.apply_mask = True  # Toggle for ablation
    
    def _create_mlp_hook(self, layer_idx: int):
        """Create a forward hook for MLP masking.
        
        Note: We hook into the MLP module, not individual layers,
        to apply masking at the intermediate activation level.
        """
        def hook_fn(module, input, output):
            if not self.apply_mask:
                return output
            
            # Get current effective mask
            mask = self.mask_state.get_effective_mask(layer_idx)
            
            # Apply mask to intermediate activations
            # For SwiGLU: output of up_proj * silu(gate_proj) before down_proj
            # We need to intercept the intermediate result
            # This is tricky because the MLP forward is fused...
            
            # Alternative: modify the MLP forward directly
            # For now, we'll use a simpler approach: scale the output
            # This is not exactly channel masking but provides similar effect
            
            # TODO: Implement proper intermediate masking via custom MLP forward
            return output
        
        return hook_fn
    
    def register_hooks(self):
        """Register forward hooks on all MLP modules.
        
        Note: Due to the fused nature of SwiGLU, we can't easily
        hook the intermediate activations. Instead, we'll modify
        the MLP forward pass directly in the training loop.
        """
        # Clear existing hooks
        self.remove_hooks()
        
        # For proper channel masking, we need to modify the forward pass
        # This will be done in the training loop via apply_masks_to_forward
        pass
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    @torch.no_grad()
    def apply_masks_to_weights(self):
        """Apply masks directly to weights (for inference/export).
        
        This zeros out the weights of pruned channels:
        - up_proj.weight[pruned_idx, :] = 0
        - gate_proj.weight[pruned_idx, :] = 0
        - down_proj.weight[:, pruned_idx] = 0
        """
        for layer_idx, proj in enumerate(self.projections):
            mask = self.mask_state.get_effective_mask(layer_idx)
            
            # Apply mask to weights
            # up_proj: [intermediate_size, hidden_size]
            proj.up_proj.weight.data *= mask.unsqueeze(1)
            # gate_proj: [intermediate_size, hidden_size]
            proj.gate_proj.weight.data *= mask.unsqueeze(1)
            # down_proj: [hidden_size, intermediate_size]
            proj.down_proj.weight.data *= mask.unsqueeze(0)
            
            # Also mask biases if present
            if proj.up_proj.bias is not None:
                proj.up_proj.bias.data *= mask
            if proj.gate_proj.bias is not None:
                proj.gate_proj.bias.data *= mask


def create_masked_mlp_forward(
    original_forward,
    mask_state: ChannelMaskState,
    layer_idx: int,
    model_type: str = "llama",
    regression_recovery: 'ChannelRegressionRecovery' = None,
    lora_bypass: 'LoRABypassModule' = None,
):
    """Create a masked forward function for an MLP layer.
    
    This wraps the original forward to apply channel masking
    at the intermediate activation level, with optional LoRA bypass
    to compensate for pruned channel information loss.
    
    Architecture:
        x ─┬── [pruned MLP] ──> y_pruned ──┬── + ──> y_final
           │                                 │
           └── LoRA_A (D→r) ──> LoRA_B (r→D) ┘  (if lora_bypass enabled)
    
    Args:
        original_forward: The original MLP forward method
        mask_state: ChannelMaskState instance
        layer_idx: Index of this layer
        model_type: Model type for architecture-specific handling
        regression_recovery: 可选，如果提供则在 hardening 接近 0.5 阈值时收集校准数据
        lora_bypass: 可选，LoRA bypass 模块用于补偿 channel pruning 的信息损失
        
    Returns:
        New forward function with masking
    """
    if model_type == "gpt2":
        # GPT-2: c_fc -> GELU -> c_proj (no gating)
        def masked_forward(self, hidden_states):
            mask = mask_state.get_effective_mask(layer_idx)
            
            # GPT-2 MLP: c_fc(x) -> gelu -> c_proj
            intermediate = self.c_fc(hidden_states)  # [batch, seq, 4*n_embd]
            intermediate = self.gelu(intermediate)    # GELU activation
            
            # 如果需要收集校准数据（在 hardening 接近阈值时）
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    Y_orig = self.c_proj(intermediate)  # 无 mask 的输出
            
            # Apply channel mask
            mask_broadcast = mask.view(1, 1, -1)
            intermediate = intermediate * mask_broadcast
            
            # Final projection
            output = self.c_proj(intermediate)  # [batch, seq, n_embd]
            output = self.dropout(output)
            
            # LoRA bypass: 从原始 hidden_states 旁路补偿
            if lora_bypass is not None:
                output = output + lora_bypass(hidden_states)
            
            # 收集校准数据
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    regression_recovery.collect_calibration_step(layer_idx, Y_orig.detach(), output.detach())
            
            return output
    elif model_type == "opt":
        # OPT: fc1 -> relu -> fc2 (no gating)
        def masked_forward(self, hidden_states):
            mask = mask_state.get_effective_mask(layer_idx)
            
            intermediate = self.fc1(hidden_states)
            intermediate = F.relu(intermediate)
            
            # 如果需要收集校准数据
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    Y_orig = self.fc2(intermediate)
            
            mask_broadcast = mask.view(1, 1, -1)
            intermediate = intermediate * mask_broadcast
            
            output = self.fc2(intermediate)
            
            # LoRA bypass: 从原始 hidden_states 旁路补偿
            if lora_bypass is not None:
                output = output + lora_bypass(hidden_states)
            
            # 收集校准数据
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    regression_recovery.collect_calibration_step(layer_idx, Y_orig.detach(), output.detach())
            
            return output
    else:
        # SwiGLU models: LLaMA, Qwen, Mistral
        def masked_forward(self, hidden_states):
            mask = mask_state.get_effective_mask(layer_idx)
            
            up = self.up_proj(hidden_states)
            gate = self.gate_proj(hidden_states)
            
            intermediate = F.silu(gate) * up
            
            # 如果需要收集校准数据
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    Y_orig = self.down_proj(intermediate)
            
            mask_broadcast = mask.view(1, 1, -1)
            intermediate = intermediate * mask_broadcast
            
            output = self.down_proj(intermediate)
            
            # LoRA bypass: 从原始 hidden_states 旁路补偿
            if lora_bypass is not None:
                output = output + lora_bypass(hidden_states)
            
            # 收集校准数据
            if regression_recovery is not None and regression_recovery._collecting:
                with torch.no_grad():
                    regression_recovery.collect_calibration_step(layer_idx, Y_orig.detach(), output.detach())
            
            return output
    
    return masked_forward


def _find_model_layers(model: nn.Module, model_type: str):
    """Find transformer layers in model, handling FSDP wrapping.
    
    Supports:
    - Raw models (LLaMA, GPT-2, OPT, Qwen, Mistral)
    - FSDP-wrapped models
    - Nested wrapper models (e.g. Distill_Model -> FSDP -> GPT)
    
    Returns:
        Tuple of (layers_iterable, mlp_attr_name)
        - layers_iterable: Iterable of layer modules
        - mlp_attr_name: Attribute name to access MLP from a layer (e.g., 'mlp')
    """
    # Unwrap FSDP / DDP / Module wrappers recursively
    def unwrap(m):
        if hasattr(m, '_fsdp_wrapped_module'):
            return unwrap(m._fsdp_wrapped_module)
        if hasattr(m, 'module'):
            return unwrap(m.module)
        return m
    
    base = unwrap(model)
    
    model_type_lower = model_type.lower()
    
    # GPT-2: transformer.h[i].mlp
    if model_type_lower == "gpt2":
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            return list(base.transformer.h), 'mlp'
        elif hasattr(base, 'h'):
            return list(base.h), 'mlp'
        # FSDP 可能把 transformer.h 平铺了，尝试搜索
        for name, mod in base.named_modules():
            if name.endswith('.h'):
                return list(mod), 'mlp'
    
    # OPT: model.decoder.layers[i]  
    elif model_type_lower == "opt":
        if hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
            return list(base.decoder.layers), 'mlp'
        if hasattr(base, 'model') and hasattr(base.model, 'decoder'):
            return list(base.model.decoder.layers), 'mlp'
    
    # LLaMA / Qwen / Mistral: model.layers[i].mlp
    else:
        if hasattr(base, 'layers'):
            return list(base.layers), 'mlp'
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return list(base.model.layers), 'mlp'
        # Deeper nesting for HF wrappers
        if hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'layers'):
            return list(base.model.model.layers), 'mlp'
    
    # 最后尝试通用搜索
    for name, mod in base.named_modules():
        if hasattr(mod, '__len__') and name.endswith(('layers', '.h')):
            try:
                first_child = list(mod)[0]
                if hasattr(first_child, 'mlp'):
                    return list(mod), 'mlp'
            except (StopIteration, IndexError):
                pass
    
    raise ValueError(f"Cannot find layers in model (type={model_type}). "
                     f"Model class: {type(base).__name__}, "
                     f"Top-level attrs: {[n for n, _ in base.named_children()]}")


def patch_model_mlp_forward(
    model: nn.Module,
    mask_state: ChannelMaskState,
    config: ChannelPruningConfig,
    regression_recovery: 'ChannelRegressionRecovery' = None,
):
    """Patch all MLP modules to use masked forward.
    
    This modifies the model in-place to apply channel masking
    during the forward pass, with optional LoRA bypass for compensation.
    
    Args:
        model: The model to patch
        mask_state: ChannelMaskState instance
        config: Configuration
        regression_recovery: 可选，线性回归恢复实例（用于收集校准数据）
        
    Returns:
        Tuple of (original_forwards, lora_bypass_modules)
        - original_forwards: List of original forward methods (for restoring later)
        - lora_bypass_modules: List of LoRABypassModule (or empty if not enabled)
    """
    from types import MethodType
    from .channel_groups import get_hidden_size
    
    layers, mlp_attr = _find_model_layers(model, config.model_type)
    original_forwards = []
    lora_bypass_modules = []
    
    # 如果启用 LoRA bypass，获取 hidden_size 并创建模块
    if config.enable_lora_bypass:
        hidden_size = get_hidden_size(model, config.model_type)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, mlp_attr)
        original_forwards.append(mlp.forward)
        
        # 创建 LoRA bypass 模块（如果启用）
        lora_bypass = None
        if config.enable_lora_bypass:
            lora_bypass = LoRABypassModule(
                hidden_size=hidden_size,
                rank=config.lora_bypass_rank,
                alpha=config.lora_bypass_alpha,
                dropout=config.lora_bypass_dropout,
            ).to(device=device, dtype=dtype)
            # 注册为 MLP 的子模块，使其参数可被 optimizer 发现
            mlp.add_module(f'_lora_bypass', lora_bypass)
            lora_bypass_modules.append(lora_bypass)
        
        # Create and bind new forward method (model_type-aware)
        new_forward = create_masked_mlp_forward(
            mlp.forward, mask_state, layer_idx, model_type=config.model_type,
            regression_recovery=regression_recovery,
            lora_bypass=lora_bypass,
        )
        mlp.forward = MethodType(new_forward, mlp)
    
    return original_forwards, lora_bypass_modules


def restore_model_mlp_forward(
    model: nn.Module,
    original_forwards: List,
    config: ChannelPruningConfig
):
    """Restore original MLP forward methods.
    
    Args:
        model: The model
        original_forwards: List of original forward methods
        config: Configuration
    """
    try:
        layers, mlp_attr = _find_model_layers(model, config.model_type)
    except ValueError:
        return
    
    for layer_idx, layer in enumerate(layers):
        if layer_idx < len(original_forwards):
            mlp = getattr(layer, mlp_attr)
            mlp.forward = original_forwards[layer_idx]
