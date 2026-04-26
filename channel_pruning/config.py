"""
Configuration for Channel-level Structured Pruning.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ChannelPruningConfig:
    """Configuration for channel-level structured pruning.
    
    This config controls the pruning behavior for FFN intermediate dimensions.
    The mask update mechanism follows HASAST's no-grad soft-to-hard approach.
    """
    
    # ========== Model Settings ==========
    model_name: str = "Qwen/Qwen3-1.7B"  # HuggingFace model name
    model_type: str = "qwen"  # qwen, llama, opt, mistral
    
    # ========== Pruning Target ==========
    # Target keep ratio for FFN intermediate channels (1.0 = no pruning)
    ffn_keep_ratio: float = 0.75  # Keep 75% of FFN channels
    # Per-layer keep ratios (if provided, overrides ffn_keep_ratio)
    # Format: {layer_idx: keep_ratio}
    per_layer_keep_ratio: Optional[dict] = None
    
    # Whether to prune attention heads (Phase 2, not implemented yet)
    prune_attention: bool = False
    attention_keep_ratio: float = 0.75
    # Per-layer attention keep ratios (if provided, overrides attention_keep_ratio)
    per_layer_attn_keep_ratio: Optional[dict] = None
    
    # ========== Attention Score Weights ==========
    # head_score[h] = alpha * s_q[h] + beta * s_k[h] + gamma * s_v[h] + delta * s_o[h]
    attn_score_alpha: float = 1.0  # Weight for q_proj score
    attn_score_beta: float = 1.0   # Weight for k_proj score (GQA distributed)
    attn_score_gamma: float = 1.0  # Weight for v_proj score (GQA distributed)
    attn_score_delta: float = 1.0  # Weight for o_proj score
    
    # ========== Importance Scoring ==========
    # Score aggregation weights for channel importance
    # channel_score[j] = alpha * s_up[j] + beta * s_gate[j] + gamma * s_down[j]
    score_alpha: float = 1.0  # Weight for up_proj score
    score_beta: float = 1.0   # Weight for gate_proj score
    score_gamma: float = 1.0  # Weight for down_proj score
    
    # Whether to normalize scores by dimension
    normalize_scores: bool = True
    
    # Importance metric: "hessian_obd", "magnitude", "wanda", "taylor"
    importance_metric: str = "hessian_obd"
    
    # EMA beta for score smoothing
    score_ema_beta: float = 0.99
    
    # ========== PCA / SlimLLM Settings ==========
    # 是否启用 SlimLLM 风格的 PCA-aware 通道重要性评分
    # 通过对 MLP 输出 Y 做 PCA，找到输出空间的主方向，
    # 评估每个 channel 在重要方向上的投影分量
    enable_pca_scoring: bool = False
    
    # PCA 分数与 base 分数的融合权重:
    #   final_score = (1 - pca_lambda) * base_score + pca_lambda * pca_score
    # base_score 来自 hessian_obd / magnitude / taylor / wanda
    # pca_score 来自 SlimLLM PCA 方向加权
    pca_lambda: float = 0.5
    
    # PCA 使用的 top-K 主成分数量（0 = 使用全部）
    # 只考虑前 K 个最大特征值对应的方向来计算 I_j^d
    pca_top_k: int = 0
    
    # PCA 特征值归一化方式:
    #   "sigmoid": C_i = sigmoid(M_i / mean(M))  （SlimLLM 原文）
    #   "softmax": C_i = softmax(M / temperature)
    #   "linear":  C_i = M_i / sum(M)
    pca_normalize_mode: str = "sigmoid"
    
    # PCA calibration 样本数（用于在线 EMA 估计协方差矩阵）
    pca_ema_beta: float = 0.99
    
    # PCA 中间激活 A[:,j] 的 L2 norm 是否参与评分
    # True: I_j = ||A[:,j]||_2 * I_j^d  (SlimLLM 原文)
    # False: I_j = I_j^d  (只看方向分量)
    pca_use_activation_norm: bool = True
    
    # PCA 协方差矩阵更新周期（每隔多少步更新一次）
    pca_update_period: int = 100
    
    # 在 channel mask hardening 时执行线性回归恢复
    # min_{A,B} ||Y_orig - (diag(A) * Y_pruned + B)||
    # 用最小二乘拟合缩放因子 A 和偏置 B，减小剪枝带来的输出误差
    pca_regression_on_hardening: bool = False
    
    # ========== LoRA Bypass 补偿 ==========
    # 在 channel pruning 的 MLP 输出端加一个 LoRA bypass，
    # 从原始 hidden_states 直接旁路到 MLP 输出，补偿被剪掉 channel 的信息损失。
    # Y_final = MLP_pruned(x) + LoRA_B(LoRA_A(x))
    enable_lora_bypass: bool = False
    # LoRA 低秩维度（rank）
    lora_bypass_rank: int = 64
    # LoRA 初始化缩放因子: LoRA_B 初始化为 0，保证训练初期 bypass 输出为 0
    lora_bypass_alpha: float = 1.0
    # LoRA bypass 的 dropout 比率（0 = 不用 dropout）
    lora_bypass_dropout: float = 0.0
    
    # ========== Mask Update Dynamics ==========
    # Mask update period (update every K steps)
    mask_update_period: int = 50
    
    # Mask learning rate for EMA update: mask <- (1-lr)*mask + lr*target
    mask_lr: float = 0.1
    
    # Temperature for soft gating (lower = harder decisions)
    temp_init: float = 1.0
    temp_min: float = 0.05
    temp_decay: float = 0.97  # Applied per mask update
    
    # Sparsity warmup: linear ramp from 0 to target sparsity
    sparsity_warmup_steps: int = 500
    
    # ========== Hardening Schedule ==========
    # When to start hardening soft masks to binary
    hardening_start_step: int = 2000
    # Duration of hardening transition (steps)
    hardening_duration: int = 5000
    
    # ========== Training Settings ==========
    # Data settings
    dataset: str = "c4_qwen"  # Dataset name (relative to data/)
    block_size: int = 2048
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Optimizer settings
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    
    # Training duration
    max_iters: int = 10000
    warmup_iters: int = 500
    lr_decay_iters: int = 10000  # For cosine decay
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 100
    
    # Logging
    log_interval: int = 10
    wandb_log: bool = True
    wandb_project: str = "channel-pruning"
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    out_dir: str = "outputs/channel_pruning"
    save_interval: int = 1000
    
    # ========== Distillation ==========
    use_distillation: bool = True
    teacher_model: Optional[str] = None  # If None, use same as model_name
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5  # Weight for distillation loss (1-alpha for task loss)
    
    # ========== Hardware ==========
    device: str = "cuda"
    dtype: str = "bfloat16"  # float16, bfloat16, float32
    compile: bool = False  # torch.compile
    
    # ========== Distributed ==========
    ddp: bool = True
    backend: str = "nccl"
    
    # ========== Calibration ==========
    # Number of calibration samples for initial importance estimation
    calibration_samples: int = 128
    calibration_seq_len: int = 2048
    
    # ========== Export ==========
    # Whether to export pruned model after training
    export_pruned: bool = True
    export_path: Optional[str] = None  # If None, use out_dir/pruned_model
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.teacher_model is None:
            self.teacher_model = self.model_name
        
        if self.export_path is None:
            self.export_path = f"{self.out_dir}/pruned_model"
        
        if self.wandb_run_name is None:
            keep_pct = int(self.ffn_keep_ratio * 100)
            self.wandb_run_name = f"{self.model_name.split('/')[-1]}_ffn{keep_pct}pct"
        
        # Validate keep ratios
        assert 0.0 < self.ffn_keep_ratio <= 1.0, "ffn_keep_ratio must be in (0, 1]"
        assert 0.0 < self.attention_keep_ratio <= 1.0, "attention_keep_ratio must be in (0, 1]"
