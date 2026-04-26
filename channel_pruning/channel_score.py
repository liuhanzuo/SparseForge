"""
Channel Score Computation: Compute importance scores for FFN channels.

Supports two scoring paradigms:

1. Weight-based (HASAST-style):
   - Per-weight score: s_w = (H + eps) * W^2  (Hessian-OBD style)
   - Aggregate to channel: s_channel[j] = sum(scores over channel group)

2. PCA-aware (SlimLLM-style):
   - 对 MLP 输出 Y 做 PCA，得到特征值 M 和特征向量 Q
   - 计算 channel j 的输出方向 w_j = W_down[:, j] 在 PCA 基底上的投影
   - 用特征值加权：I_j^d = ||u_j ⊙ C||_2, u_j = Q^T w_j
   - 结合中间激活幅值：I_j = ||A[:,j]||_2 · I_j^d
   - 最终与 base score 融合：final = (1-λ) * base + λ * pca

The aggregation ensures that pruning a channel considers all affected weights:
- up_proj row j
- gate_proj row j  
- down_proj column j
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
import math

# FSDP 兼容：尝试导入 FSDP
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _has_fsdp = True
except ImportError:
    _has_fsdp = False

from .channel_groups import (
    get_mlp_projections, 
    MLPProjections,
    get_intermediate_size,
    get_hidden_size,
    get_num_layers
)
from .config import ChannelPruningConfig


@dataclass
class LayerChannelScores:
    """Channel scores for a single layer."""
    layer_idx: int
    scores: torch.Tensor  # [intermediate_size]
    up_scores: torch.Tensor  # [intermediate_size] - contribution from up_proj
    gate_scores: torch.Tensor  # [intermediate_size] - contribution from gate_proj
    down_scores: torch.Tensor  # [intermediate_size] - contribution from down_proj


@dataclass
class LayerPCAState:
    """PCA 状态：每层 MLP 输出空间的协方差统计。
    
    对于 MLP 输出 Y ∈ R^{N×D}:
      Ỹ = Y - mean(Y)
      Σ = (1/N) Ỹ^T Ỹ  ∈ R^{D×D}
    特征分解: Σ = Q diag(M) Q^T
    
    Attributes:
        layer_idx: 层索引
        cov_matrix: 协方差矩阵 Σ ∈ R^{D×D}（在线 EMA 估计）
        eigenvalues: 特征值 M ∈ R^D（降序排列）
        eigenvectors: 特征向量 Q ∈ R^{D×D}（列为特征向量）
        n_samples: 已累积的样本数
        importance_weights: 归一化后的特征值权重 C ∈ R^D
    """
    layer_idx: int
    cov_matrix: torch.Tensor  # [D, D]
    eigenvalues: Optional[torch.Tensor] = None  # [D]
    eigenvectors: Optional[torch.Tensor] = None  # [D, D]
    n_samples: int = 0
    importance_weights: Optional[torch.Tensor] = None  # [D]
    mean_y: Optional[torch.Tensor] = None  # [D] 均值


class PCAManager:
    """管理所有层的 PCA 状态和在线协方差估计。
    
    SlimLLM 的核心思想：
    1. 对 MLP 输出 Y 做 PCA，找到输出空间中方差最大的方向（主成分）
    2. 每个 channel j 的输出方向 w_j = W_down[:, j]
    3. w_j 在 PCA 基底上的投影：u_j = Q^T w_j
    4. 用特征值加权的 L2 范数：I_j^d = ||u_j ⊙ C||_2
    5. 结合中间激活幅值：I_j = ||A[:,j]||_2 · I_j^d
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
        
        # 获取 MLP projections
        self.projections = get_mlp_projections(model, config.model_type)
        self.num_layers = len(self.projections)
        
        # 获取维度信息（从 config 获取，不从 FSDP 分片后的 weight shape 取）
        self.hidden_size = get_hidden_size(model, config.model_type)  # D
        self.intermediate_size = get_intermediate_size(model, config.model_type)  # H
        
        # 每层的 PCA 状态
        self.pca_states: Dict[int, LayerPCAState] = {}
        for i in range(self.num_layers):
            self.pca_states[i] = LayerPCAState(
                layer_idx=i,
                cov_matrix=torch.zeros(self.hidden_size, self.hidden_size, 
                                       device=self.device, dtype=torch.float32),
                mean_y=torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32),
                n_samples=0,
            )
        
        # 每层中间激活 A[:,j] 的 L2 norm EMA: [num_layers, intermediate_size]
        self.activation_norms: Dict[int, torch.Tensor] = {}
        
        # Forward hooks
        self._hooks: List = []
        self._collecting = False
    
    @torch.no_grad()
    def update_cov_from_output(self, layer_idx: int, Y: torch.Tensor):
        """用 MLP 输出 Y 更新协方差矩阵的在线估计。
        
        Y ∈ R^{batch*seq, D}
        使用 EMA 或累积均值更新:
          μ ← β * μ + (1-β) * mean(Y)
          Σ ← β * Σ + (1-β) * (1/N) Ỹ^T Ỹ
        
        Args:
            layer_idx: 层索引
            Y: MLP 输出 [N, D]，已经 reshape 为 2D
        """
        state = self.pca_states[layer_idx]
        beta = self.config.pca_ema_beta
        
        Y_flat = Y.reshape(-1, Y.shape[-1]).float()  # [N, D]
        N = Y_flat.shape[0]
        
        if N == 0:
            return
        
        # 计算当前 batch 的均值和去中心化
        batch_mean = Y_flat.mean(dim=0)  # [D]
        Y_centered = Y_flat - batch_mean.unsqueeze(0)  # [N, D]
        
        # 计算当前 batch 的协方差: (1/N) Y_centered^T Y_centered
        batch_cov = Y_centered.t().matmul(Y_centered) / N  # [D, D]
        
        if state.n_samples == 0:
            # 第一次：直接赋值
            state.cov_matrix.copy_(batch_cov)
            state.mean_y = batch_mean.clone()
        else:
            # EMA 更新
            state.cov_matrix.mul_(beta).add_(batch_cov, alpha=1.0 - beta)
            state.mean_y.mul_(beta).add_(batch_mean, alpha=1.0 - beta)
        
        state.n_samples += N
    
    @torch.no_grad()
    def update_activation_norm(self, layer_idx: int, A: torch.Tensor):
        """更新中间激活 A[:,j] 的 L2 norm EMA。
        
        A ∈ R^{N, H} 是 silu(gate(x)) * up(x) 的结果
        
        Args:
            layer_idx: 层索引
            A: 中间激活 [N, H]
        """
        A_flat = A.reshape(-1, A.shape[-1]).float()  # [N, H]
        # ||A[:,j]||_2 for each channel j
        col_norms = A_flat.norm(dim=0)  # [H]
        
        beta = self.config.pca_ema_beta
        if layer_idx not in self.activation_norms:
            self.activation_norms[layer_idx] = col_norms
        else:
            self.activation_norms[layer_idx].mul_(beta).add_(col_norms, alpha=1.0 - beta)
    
    @torch.no_grad()
    def compute_eigendecomposition(self, layer_idx: int):
        """对层 layer_idx 的协方差矩阵做特征分解。
        
        Σ = Q diag(M) Q^T
        
        然后计算归一化的重要性权重 C。
        """
        state = self.pca_states[layer_idx]
        
        if state.n_samples == 0:
            return
        
        # 对称矩阵特征分解
        cov = state.cov_matrix
        # 确保对称性（数值稳定）
        cov = (cov + cov.t()) / 2.0
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except Exception:
            # fallback: 如果 eigh 失败，用 svd
            try:
                U, S, Vh = torch.linalg.svd(cov, full_matrices=True)
                eigenvalues = S
                eigenvectors = U
            except Exception:
                # 最终 fallback：跳过 PCA
                return
        
        # eigh 返回升序排列，我们要降序
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        
        # 截断负特征值（数值误差）
        eigenvalues = eigenvalues.clamp(min=0.0)
        
        state.eigenvalues = eigenvalues
        state.eigenvectors = eigenvectors
        
        # 计算重要性权重 C
        state.importance_weights = self._compute_importance_weights(eigenvalues)
    
    @torch.no_grad()
    def _compute_importance_weights(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """根据特征值计算归一化的重要性权重 C。
        
        SlimLLM 原文: C_i = sigmoid(M_i / mean(M))
        
        Args:
            eigenvalues: 特征值 M ∈ R^D（降序）
            
        Returns:
            importance_weights: C ∈ R^D
        """
        M = eigenvalues.float()
        mode = self.config.pca_normalize_mode
        eps = 1e-8
        
        if mode == "sigmoid":
            # SlimLLM 原文: C_i = sigmoid(M_i / mean(M))
            M_mean = M.mean() + eps
            C = torch.sigmoid(M / M_mean)
        elif mode == "softmax":
            # softmax 归一化
            C = torch.softmax(M, dim=0)
        elif mode == "linear":
            # 线性归一化: C_i = M_i / sum(M)
            M_sum = M.sum() + eps
            C = M / M_sum
        else:
            # 默认 sigmoid
            M_mean = M.mean() + eps
            C = torch.sigmoid(M / M_mean)
        
        # 如果指定了 top-K，只保留前 K 个分量
        top_k = self.config.pca_top_k
        if top_k > 0 and top_k < len(C):
            mask = torch.zeros_like(C)
            mask[:top_k] = 1.0
            C = C * mask
        
        return C
    
    @torch.no_grad()
    def compute_pca_channel_scores(self, layer_idx: int) -> torch.Tensor:
        """计算层 layer_idx 的 PCA-aware channel 重要性分数。
        
        SlimLLM 公式:
          w_j = W_down[:, j] ∈ R^D    （channel j 的输出投影方向）
          u_j = Q^T w_j ∈ R^D          （在 PCA 基底上的投影）
          I_j^d = ||u_j ⊙ C||_2        （方向加权重要性）
          I_j = ||A[:,j]||_2 · I_j^d    （结合激活幅值）
        
        Args:
            layer_idx: 层索引
            
        Returns:
            PCA channel scores [intermediate_size]
        """
        state = self.pca_states[layer_idx]
        proj = self.projections[layer_idx]
        
        # 检查是否有有效的 PCA 分解
        if state.eigenvectors is None or state.importance_weights is None:
            # 没有 PCA 数据，返回全 1（不影响评分）
            return torch.ones(self.intermediate_size, device=self.device)
        
        Q = state.eigenvectors.float()  # [D, D]
        C = state.importance_weights.float()  # [D]
        
        # W_down ∈ R^{D, H}，每列 w_j 是 channel j 的输出投影方向
        W_down = proj.down_proj.weight.detach().float()  # [D, H]
        
        # u = Q^T W_down ∈ R^{D, H}，每列 u_j = Q^T w_j
        u = Q.t().matmul(W_down)  # [D, H]
        
        # I_j^d = ||u_j ⊙ C||_2 = sqrt(sum_i (u_j[i] * C_i)^2)
        # u_j ⊙ C: 每列乘以 C
        u_weighted = u * C.unsqueeze(1)  # [D, H]，广播 C
        I_d = u_weighted.norm(dim=0)  # [H]，每个 channel 的方向加权重要性
        
        # 结合中间激活幅值
        if self.config.pca_use_activation_norm and layer_idx in self.activation_norms:
            act_norm = self.activation_norms[layer_idx].float()  # [H]
            # I_j = ||A[:,j]||_2 · I_j^d
            pca_scores = act_norm * I_d
        else:
            pca_scores = I_d
        
        return pca_scores
    
    @torch.no_grad()
    def compute_all_eigendecompositions(self):
        """对所有层执行特征分解。"""
        for i in range(self.num_layers):
            self.compute_eigendecomposition(i)
    
    def install_hooks(self, model: nn.Module):
        """安装 forward hooks 来收集 MLP 输出和中间激活。
        
        Hook 放在 MLP 模块上，收集:
        1. MLP 输出 Y（用于更新协方差矩阵）
        2. 中间激活 A = silu(gate(x)) * up(x)（用于激活 norm）
        
        Args:
            model: 要安装 hook 的模型
        """
        self.remove_hooks()
        
        # Unwrap FSDP / DDP wrappers recursively
        def _unwrap(m):
            if hasattr(m, '_fsdp_wrapped_module'):
                return _unwrap(m._fsdp_wrapped_module)
            if hasattr(m, 'module'):
                return _unwrap(m.module)
            return m
        
        base_model = _unwrap(model)
        
        # Further unwrap model-specific wrappers
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
            base_model = base_model.model.model
        elif hasattr(base_model, 'model'):
            base_model = base_model.model
        
        # 找到 layers（支持各种架构）
        layers = None
        model_type_lower = self.model_type.lower() if hasattr(self, 'model_type') else ""
        
        if model_type_lower == "gpt2":
            if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
                layers = base_model.transformer.h
            elif hasattr(base_model, 'h'):
                layers = base_model.h
        elif model_type_lower == "opt":
            if hasattr(base_model, 'decoder') and hasattr(base_model.decoder, 'layers'):
                layers = base_model.decoder.layers
        
        if layers is None:
            if hasattr(base_model, 'layers'):
                layers = base_model.layers
            elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                layers = base_model.model.layers
        
        if layers is None:
            print("[PCAManager] 警告: 无法找到 model layers，跳过 hook 安装")
            return
        
        for layer_idx, layer in enumerate(layers):
            if layer_idx >= self.num_layers:
                break
            
            mlp = layer.mlp
            
            # Hook: 收集 MLP 输出 Y
            def make_output_hook(idx):
                def hook_fn(module, input, output):
                    if not self._collecting:
                        return
                    # output 是 MLP 的输出: [batch, seq, D]
                    with torch.no_grad():
                        self.update_cov_from_output(idx, output.detach())
                return hook_fn
            
            h = mlp.register_forward_hook(make_output_hook(layer_idx))
            self._hooks.append(h)
            
            # Hook: 收集中间激活 A (在 down_proj 之前)
            # 我们在 down_proj 上安装输入 hook
            def make_activation_hook(idx):
                def hook_fn(module, input):
                    if not self._collecting:
                        return
                    # input[0] 是 A = silu(gate(x)) * up(x): [batch, seq, H]
                    with torch.no_grad():
                        self.update_activation_norm(idx, input[0].detach())
                return hook_fn
            
            proj = self.projections[layer_idx]
            h2 = proj.down_proj.register_forward_pre_hook(make_activation_hook(layer_idx))
            self._hooks.append(h2)
        
        self._collecting = True
    
    def remove_hooks(self):
        """移除所有 forward hooks。"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._collecting = False
    
    def start_collecting(self):
        """开始收集激活数据。"""
        self._collecting = True
    
    def stop_collecting(self):
        """停止收集激活数据。"""
        self._collecting = False


class ChannelScoreComputer:
    """Computes importance scores for FFN channels.
    
    Uses Hessian-based scoring similar to HASAST/SlimLLM:
    - Per-weight importance: s = (H + eps) * W^2
    - Channel importance: aggregate over all weights in the channel group
    
    Supports multiple importance metrics:
    - hessian_obd: (H + eps) * W^2 (default, most effective)
    - magnitude: |W|
    - taylor: |W * grad|
    - wanda: |W| * activation_norm
    
    可选融合 SlimLLM PCA-aware 方向评分:
    - final_score = (1-λ) * base_score + λ * pca_score
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
        
        # Extract MLP projections
        self.projections = get_mlp_projections(model, config.model_type)
        self.num_layers = len(self.projections)
        self.intermediate_size = get_intermediate_size(model, config.model_type)
        
        # Score aggregation weights
        self.alpha = config.score_alpha
        self.beta = config.score_beta
        self.gamma = config.score_gamma
        
        # EMA for score smoothing
        self.score_ema_beta = config.score_ema_beta
        self.score_ema: Optional[Dict[int, torch.Tensor]] = None  # {layer_idx: scores}
        
        # Hessian diagonal estimates (per layer)
        # Will be populated during calibration or training
        self.hessian_up: Dict[int, torch.Tensor] = {}
        self.hessian_gate: Dict[int, torch.Tensor] = {}
        self.hessian_down: Dict[int, torch.Tensor] = {}
        
        # Activation statistics for WANDA metric
        self.activation_norms: Dict[int, torch.Tensor] = {}
        
        # PCA Manager (SlimLLM-style direction-aware scoring)
        self.pca_manager: Optional[PCAManager] = None
        if config.enable_pca_scoring:
            self.pca_manager = PCAManager(model, config, device)
        
    def _get_weight_importance(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor],
        grad: Optional[torch.Tensor] = None,
        activation_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-weight importance scores.
        
        Args:
            weight: Weight tensor
            hessian: Hessian diagonal (same shape as weight)
            grad: Gradient (same shape as weight, for taylor metric)
            activation_norm: Activation norms (for wanda metric)
            
        Returns:
            Importance scores (same shape as weight)
        """
        eps = 1e-8
        metric = self.config.importance_metric
        
        W = weight.detach().float()
        
        if metric == "hessian_obd":
            # OBD-style: (H + eps) * W^2
            if hessian is not None and hessian.shape == W.shape:
                H = hessian.detach().float()
                scores = (H + eps) * (W * W)
            else:
                # Fallback to magnitude if no Hessian
                scores = W * W
                
        elif metric == "magnitude":
            scores = W.abs()
            
        elif metric == "taylor":
            # Taylor expansion: |W * grad|
            if grad is not None and grad.shape == W.shape:
                g = grad.detach().float()
                scores = (W * g).abs()
            else:
                scores = W.abs()
                
        elif metric == "wanda":
            # WANDA: |W| * sqrt(activation_norm)
            scores = W.abs()
            if activation_norm is not None:
                # activation_norm is [hidden_size], broadcast to weight shape
                if weight.shape[1] == activation_norm.shape[0]:
                    norm = activation_norm.detach().float().reshape(1, -1)
                    scores = scores * torch.sqrt(norm + eps)
        else:
            # Default to magnitude
            scores = W.abs()
            
        return scores
    
    def _aggregate_channel_scores(
        self,
        up_scores: torch.Tensor,  # [intermediate_size, hidden_size]
        gate_scores: torch.Tensor,  # [intermediate_size, hidden_size]
        down_scores: torch.Tensor,  # [hidden_size, intermediate_size]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate per-weight scores to channel scores.
        
        For channel j:
        - s_up[j] = sum(up_scores[j, :])  (sum over hidden dim)
        - s_gate[j] = sum(gate_scores[j, :])
        - s_down[j] = sum(down_scores[:, j])  (sum over hidden dim)
        - total[j] = alpha * s_up[j] + beta * s_gate[j] + gamma * s_down[j]
        
        Args:
            up_scores: [intermediate_size, hidden_size]
            gate_scores: [intermediate_size, hidden_size]
            down_scores: [hidden_size, intermediate_size]
            
        Returns:
            total_scores: [intermediate_size]
            up_channel_scores: [intermediate_size]
            gate_channel_scores: [intermediate_size]
            down_channel_scores: [intermediate_size]
        """
        # FSDP 兼容：如果权重被分片为一维，尝试 reshape 回二维
        intermediate_size = self.intermediate_size
        hidden_size = get_hidden_size(self.model, self.model_type)
        
        if up_scores.dim() == 1:
            # FSDP 分片后权重变为一维 flat_param，尝试 reshape
            expected_numel = intermediate_size * hidden_size
            if up_scores.numel() == expected_numel:
                up_scores = up_scores.reshape(intermediate_size, hidden_size)
            else:
                # 分片后 numel 不等于完整大小，说明是部分分片
                # 此时无法正确 reshape，回退到均匀分配
                print(f"[ChannelScore] WARNING: up_scores is 1D with {up_scores.numel()} elements "
                      f"(expected {expected_numel}), falling back to equal scores")
                s_up = torch.ones(intermediate_size, device=up_scores.device, dtype=up_scores.dtype)
                s_gate = torch.ones(intermediate_size, device=up_scores.device, dtype=up_scores.dtype)
                s_down = torch.ones(intermediate_size, device=up_scores.device, dtype=up_scores.dtype)
                total = self.alpha * s_up + self.beta * s_gate + self.gamma * s_down
                return total, s_up, s_gate, s_down
        
        if gate_scores.dim() == 1:
            expected_numel = intermediate_size * hidden_size
            if gate_scores.numel() == expected_numel:
                gate_scores = gate_scores.reshape(intermediate_size, hidden_size)
        
        if down_scores.dim() == 1:
            expected_numel = hidden_size * intermediate_size
            if down_scores.numel() == expected_numel:
                down_scores = down_scores.reshape(hidden_size, intermediate_size)
        
        # Sum over hidden dimension
        s_up = up_scores.sum(dim=1)  # [intermediate_size]
        s_gate = gate_scores.sum(dim=1)  # [intermediate_size]
        s_down = down_scores.sum(dim=0)  # [intermediate_size]
        
        # Optional normalization by dimension
        if self.config.normalize_scores:
            hidden_size = up_scores.shape[1] if up_scores.dim() == 2 else hidden_size
            s_up = s_up / hidden_size
            s_gate = s_gate / hidden_size
            s_down = s_down / hidden_size
        
        # Weighted aggregation
        total = self.alpha * s_up + self.beta * s_gate + self.gamma * s_down
        
        return total, s_up, s_gate, s_down
    
    @torch.no_grad()
    def compute_layer_scores(self, layer_idx: int) -> LayerChannelScores:
        """Compute channel scores for a single layer.
        
        如果启用了 PCA scoring，会将 base score 与 PCA score 融合:
          final = (1 - pca_lambda) * base_score + pca_lambda * pca_score
        
        Args:
            layer_idx: Index of the transformer layer
            
        Returns:
            LayerChannelScores with per-channel importance scores
        """
        proj = self.projections[layer_idx]
        
        # Get weights
        up_weight = proj.up_proj.weight  # [intermediate_size, hidden_size]
        gate_weight = proj.gate_proj.weight  # [intermediate_size, hidden_size]
        down_weight = proj.down_proj.weight  # [hidden_size, intermediate_size]
        
        # Get Hessian estimates (if available)
        up_hessian = self.hessian_up.get(layer_idx)
        gate_hessian = self.hessian_gate.get(layer_idx)
        down_hessian = self.hessian_down.get(layer_idx)
        
        # Get gradients (if available and metric requires them)
        up_grad = proj.up_proj.weight.grad if hasattr(proj.up_proj.weight, 'grad') else None
        gate_grad = proj.gate_proj.weight.grad if hasattr(proj.gate_proj.weight, 'grad') else None
        down_grad = proj.down_proj.weight.grad if hasattr(proj.down_proj.weight, 'grad') else None
        
        # Get activation norms (for WANDA)
        activation_norm = self.activation_norms.get(layer_idx)
        
        # Compute per-weight importance (base scores)
        up_scores = self._get_weight_importance(up_weight, up_hessian, up_grad, activation_norm)
        gate_scores = self._get_weight_importance(gate_weight, gate_hessian, gate_grad, activation_norm)
        down_scores = self._get_weight_importance(down_weight, down_hessian, down_grad)
        
        # Aggregate to channel scores
        base_total, s_up, s_gate, s_down = self._aggregate_channel_scores(
            up_scores, gate_scores, down_scores
        )
        
        # PCA-aware scoring 融合
        if self.config.enable_pca_scoring and self.pca_manager is not None:
            pca_scores = self.pca_manager.compute_pca_channel_scores(layer_idx)
            
            lam = self.config.pca_lambda
            if lam >= 1.0:
                # λ=1.0: 直接使用 PCA score，不和 base 做融合
                total = pca_scores
            elif lam <= 0.0:
                # λ=0.0: 完全不用 PCA，退化为 base score
                total = base_total
            else:
                # 0 < λ < 1: 对两组分数分别做 z-score 归一化再融合，避免量纲不同
                base_norm = self._normalize_scores(base_total)
                pca_norm = self._normalize_scores(pca_scores)
                total = (1.0 - lam) * base_norm + lam * pca_norm
        else:
            total = base_total
        
        return LayerChannelScores(
            layer_idx=layer_idx,
            scores=total,
            up_scores=s_up,
            gate_scores=s_gate,
            down_scores=s_down
        )
    
    @staticmethod
    @torch.no_grad()
    def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        """Z-score 归一化，使得不同来源的分数可以公平融合。"""
        s = scores.float()
        mu = s.mean()
        sigma = s.std() + 1e-8
        return (s - mu) / sigma
    
    @torch.no_grad()
    def compute_all_scores(self) -> List[LayerChannelScores]:
        """Compute channel scores for all layers.
        
        在 FSDP 下，需要 summon_full_params 来恢复完整的权重参数，
        否则权重是分片后的一维 flat_param，无法正确计算 channel scores。
        
        Returns:
            List of LayerChannelScores, one per layer
        """
        # 检测是否在 FSDP 下
        is_fsdp = _has_fsdp and isinstance(self.model, FSDP)
        
        if is_fsdp:
            # 使用 summon_full_params 恢复完整参数
            with FSDP.summon_full_params(self.model, recurse=True, writeback=False):
                all_scores = []
                for layer_idx in range(self.num_layers):
                    scores = self.compute_layer_scores(layer_idx)
                    all_scores.append(scores)
                return all_scores
        else:
            all_scores = []
            for layer_idx in range(self.num_layers):
                scores = self.compute_layer_scores(layer_idx)
                all_scores.append(scores)
            return all_scores
    
    @torch.no_grad()
    def update_hessian_ema(self, layer_idx: int):
        """Update Hessian diagonal estimate using gradient squared EMA.
        
        This is a simple Fisher Information approximation:
        H ≈ E[grad^2]
        
        Args:
            layer_idx: Index of the layer to update
        """
        proj = self.projections[layer_idx]
        beta = self.config.score_ema_beta
        
        for name, weight, hessian_dict in [
            ('up', proj.up_proj.weight, self.hessian_up),
            ('gate', proj.gate_proj.weight, self.hessian_gate),
            ('down', proj.down_proj.weight, self.hessian_down),
        ]:
            if weight.grad is None:
                continue
            
            grad_sq = (weight.grad.detach() ** 2).float()
            
            if layer_idx not in hessian_dict:
                # Initialize with current grad^2
                hessian_dict[layer_idx] = grad_sq.clone()
            else:
                # EMA update
                hessian_dict[layer_idx].mul_(beta).add_(grad_sq, alpha=1 - beta)
    
    @torch.no_grad()
    def update_all_hessian_ema(self):
        """Update Hessian estimates for all layers.
        
        注意：在 FSDP 下，weight.grad 是 reduce-scattered 后的分片形状，
        因此存储的 Hessian 也是分片形状。在 compute_layer_scores 中，
        如果 summon_full_params 恢复了完整权重，hessian.shape != W.shape，
        会自动 fallback 到 magnitude scoring（W^2）。
        """
        for layer_idx in range(self.num_layers):
            self.update_hessian_ema(layer_idx)
    
    @torch.no_grad()
    def update_score_ema(self, layer_scores: List[LayerChannelScores]):
        """Update the running EMA of channel scores.
        
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
            Smoothed channel scores [intermediate_size]
        """
        if self.score_ema is not None and layer_idx in self.score_ema:
            return self.score_ema[layer_idx]
        else:
            # Compute fresh scores if no EMA available
            return self.compute_layer_scores(layer_idx).scores
    
    def register_activation_hook(self, layer_idx: int):
        """Register a forward hook to collect activation statistics.
        
        Used for WANDA metric: tracks ||X||_2 for input activations.
        
        Args:
            layer_idx: Layer index to hook
        """
        proj = self.projections[layer_idx]
        
        def hook_fn(module, input, output):
            x = input[0]  # [batch, seq_len, hidden_size]
            # Compute L2 norm per hidden dimension across batch and sequence
            x_flat = x.reshape(-1, x.shape[-1])  # [batch*seq, hidden]
            norms = (x_flat ** 2).sum(dim=0)  # [hidden_size]
            
            if layer_idx not in self.activation_norms:
                self.activation_norms[layer_idx] = norms.detach()
            else:
                # Running sum (will normalize later)
                self.activation_norms[layer_idx] += norms.detach()
        
        # Register on up_proj (input to MLP)
        return proj.up_proj.register_forward_hook(hook_fn)
    
    def install_pca_hooks(self, model: nn.Module):
        """安装 PCA 数据收集 hooks（委托给 PCAManager）。
        
        Args:
            model: 模型
        """
        if self.pca_manager is not None:
            self.pca_manager.install_hooks(model)
    
    def remove_pca_hooks(self):
        """移除 PCA hooks。"""
        if self.pca_manager is not None:
            self.pca_manager.remove_hooks()
    
    @torch.no_grad()
    def update_pca(self, step: int):
        """根据收集的数据更新 PCA 分解。
        
        每隔 pca_update_period 步做一次特征分解（开销较大）。
        
        Args:
            step: 当前训练步
        """
        if self.pca_manager is None:
            return
        
        period = self.config.pca_update_period
        if step > 0 and step % period == 0:
            self.pca_manager.compute_all_eigendecompositions()
