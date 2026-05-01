"""
Argparse configuration for the two SparseForge entry points.

Public API
----------
- ``str2bool(v)``
    Tolerant boolean parser used as ``type=`` for argparse flags.
- ``build_llama_parser()``
    Returns the parser used by ``main_llama.py`` (LLaMA-only pipeline).
- ``build_universal_parser()``
    Returns the parser used by ``main_universal.py`` (multi-architecture
    pipeline: LLaMA / OPT / GPT-2 / Qwen / Mistral / DeepSeek-MoE / Hunyuan).

The common options are declared once in ``_add_common_args`` to guarantee
that both entry points stay behaviorally identical on the shared flags.
Universal-only options (model auto-detection, always_save_checkpoint,
calibration-source, channel pruning, PCA scoring, LoRA bypass) live in
``_add_universal_only_args``.
"""
from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Type helper
# ---------------------------------------------------------------------------
def str2bool(v):
    """Tolerant boolean parser.

    Accepts common truthy/falsy string spellings; raises
    ``argparse.ArgumentTypeError`` on anything else. ``None`` is treated as
    ``False`` so that flags like ``--skip_eval`` (default=None) remain safe.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


# ---------------------------------------------------------------------------
# Common arguments (shared by main_llama.py and main_universal.py)
# ---------------------------------------------------------------------------
def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add all flags that are identical between the two entry points.

    These defaults MUST match the originals in ``main_llama_legacy.py`` and
    ``main_universal_legacy.py`` exactly, otherwise experiments reproduced
    from the paper will silently diverge.
    """
    # ------------- model / distillation -------------
    parser.add_argument('--student_model', type=str, default='NousResearch/Llama-2-7b-hf')
    parser.add_argument('--teacher_model', type=str, default='NousResearch/Llama-2-7b-hf')
    parser.add_argument('--distill_model', type=str2bool, default=False)
    parser.add_argument('--hardness_task', type=float, default=1.0)
    parser.add_argument('--hardness_kldiv', type=float, default=1.0)
    parser.add_argument('--hardness_squarehead', type=float, default=1.0)

    # ------------- eval -------------
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--skip_wiki_ppl', type=str2bool, default=True)  # Skip WikiText PPL by default (stability)
    # Backward-compat: older launchers used --skip_eval to mean "skip wiki_ppl".
    parser.add_argument('--skip_eval', type=str2bool, default=None)  # DEPRECATED alias of --skip_wiki_ppl

    # lm_eval harness 评估（finalization 阶段）
    parser.add_argument('--finalize_lm_eval', type=str2bool, default=False, help='在 finalization（post-finalize finetune）阶段的每次 eval 时运行 lm_eval harness，并保存 lm_eval mean accuracy 最佳的 checkpoint')
    parser.add_argument('--lm_eval_tasks', type=str, default='boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa', help='lm_eval 评估的任务列表，逗号分隔')
    parser.add_argument('--lm_eval_batch_size', type=int, default=4, help='lm_eval 评估的 batch size')

    # ------------- checkpoint / logging -------------
    parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--output_flip_every', type=int, default=10)

    # ------------- batch / optimization -------------
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--srste_decay', type=float, default=6e-5)
    parser.add_argument('--max_iters', type=int, default=20000)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--lr_decay_iters', type=int, default=20000)
    parser.add_argument('--increase_step', type=int, default=10000)

    # ------------- sparsity / mask core -------------
    parser.add_argument('--mode', choices=['sparse_forward', 'dense_forward'], default='sparse_forward')
    parser.add_argument('--mask_type', choices=['structured', 'unstructured'], default='structured')
    parser.add_argument('--hard_mask_type', choices=['match', 'unstructured', 'structured', 'block16', 'block_sparse16', 'block_sparse32', 'nm_2_4'], default='match')
    parser.add_argument('--mask_metric', choices=['wanda', 'movement', 'hessian_obd', 'hessian_ratio', 'magnitude'], default='magnitude')
    parser.add_argument('--change_mask', type=str2bool, default=False)
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--mask_warmup_steps', type=int, default=0)
    parser.add_argument('--mask_transition_steps', type=int, default=0)
    parser.add_argument('--hybrid_alpha', type=float, default=1.0)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)

    # ------------- SLoRB (low-rank bypass) -------------
    parser.add_argument('--SLoRB_k', type=int, default=64)
    parser.add_argument('--SLoRB', type=str2bool, default=False)
    parser.add_argument('--SLoRB_init_type', choices=['mean', 'sum', 'xavier'], default='mean')
    parser.add_argument('--trainable_projection', type=str2bool, default=False)

    # ------------- misc training options -------------
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=False)
    parser.add_argument('--wandb_logging', type=str2bool, default=False)
    parser.add_argument('--lambda_mid_max', type=float, default=0.01)
    parser.add_argument('--enable_hutchinson', type=str2bool, default=False)

    # ------------- mask hardening -------------
    parser.add_argument('--hardening_period', type=int, default=2000)
    parser.add_argument('--hardening_fraction', type=float, default=0.2)
    parser.add_argument('--hardening_band_low', type=float, default=0.4)
    parser.add_argument('--hardening_band_high', type=float, default=0.6)

    # ------------- wandb / precision / attn -------------
    parser.add_argument('--wandb_project', type=str, default='LLaMA-Sparse')
    parser.add_argument('--wandb_run_name', type=str, default='llama-7b')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--eager_attention', type=str2bool, default=False, help='Use eager (manual) attention instead of Flash Attention. Required for Hutchinson Hessian estimation.')

    # ------------- FSDP -------------
    parser.add_argument('--use_fsdp', type=str2bool, default=True)  # Default to True for multi-GPU to avoid OOM
    parser.add_argument('--fsdp_mode', type=str, default='fully_sharded', choices=['fully_sharded', 'shard_grad_op', 'no_shard', 'hybrid_sharded'])
    parser.add_argument('--fsdp_cpu_offload', type=str2bool, default=False)
    parser.add_argument('--fsdp_mixed_precision', type=str2bool, default=True)
    parser.add_argument('--local_rank', type=int, default=0)

    # ------------- mask update schedule -------------
    parser.add_argument('--mask_update_period', type=int, default=10)
    parser.add_argument('--mask_update_switch_step', type=int, default=0)
    parser.add_argument('--mask_update_period_before', type=int, default=None)
    parser.add_argument('--mask_update_period_after', type=int, default=None)
    parser.add_argument('--mask_lr', type=float, default=0.1)
    parser.add_argument('--mask_penalty_lr', type=float, default=None)
    parser.add_argument('--mask_penalty_lr_min', type=float, default=None, help='If set, enables penalty lr schedule: start at min, grow to mask_penalty_lr')
    parser.add_argument('--mask_penalty_lr_schedule', type=str, choices=['constant', 'linear', 'cosine'], default='constant', help='Schedule for mask_penalty_lr: constant (no change), linear (from min to max), cosine (slow start fast end)')
    parser.add_argument('--mask_penalty_mode', choices=['mid', 'structured_topn', 'block16', 'block_sparse16', 'block_sparse32', 'nm_2_4'], default='mid')

    # ------------- score / temperature EMA -------------
    parser.add_argument('--score_ema_beta', type=float, default=0.99)
    parser.add_argument('--temp_init', type=float, default=1.0)
    parser.add_argument('--temp_min', type=float, default=0.05)
    parser.add_argument('--temp_decay', type=float, default=0.97)
    parser.add_argument('--sparsity_warmup_steps', type=int, default=0)

    # ------------- N:M structured sparsity -------------
    parser.add_argument('--structured_n', type=int, default=2)
    parser.add_argument('--structured_m', type=int, default=4)
    parser.add_argument('--tau_sample_size', type=int, default=262144)
    parser.add_argument('--sparsity_alpha', type=float, default=0.0)
    parser.add_argument('--mask_hardening_start', type=int, default=0)
    parser.add_argument('--mask_hardening_duration', type=int, default=10000)
    parser.add_argument('--structured_exact', type=str2bool, default=False)
    parser.add_argument('--beta_structural_start', type=int, default=0, help='step where β structural mixing starts rising from 0')
    parser.add_argument('--beta_structural_end', type=int, default=0, help='step where β structural mixing reaches 1.0 (0=disabled)')

    # ------------- GLU joint mask / weight scaling / CAST -------------
    parser.add_argument('--glu_joint_mask', type=str2bool, default=False, help='For GLU architectures (LLaMA/Qwen/Mistral), use joint gate/up mask to ensure aligned pruning')
    parser.add_argument('--weight_scaling', type=str2bool, default=False, help='(CAST) Enable weight scaling to compensate energy loss from sparsification. Scale = M/N for structured, 1/(1-sr) for unstructured.')
    parser.add_argument('--adaptive_l1_decay', type=float, default=0.0, help='(CAST AdamS) Adaptive L1 decay strength. Applied as sign(w)*lambda/denom to drive small weights toward zero. Recommended: 1e-4 ~ 1e-3.')

    # ------------- final finetune / param freezing -------------
    parser.add_argument('--final_finetune_iters', type=int, default=1000)
    parser.add_argument('--freeze_low', type=float, default=0.0)
    parser.add_argument('--freeze_high', type=float, default=1.0)

    # ------------- deepspeed (optional, rarely used) -------------
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='configs/deepspeed_xl.json')

    # ------------- I/O -------------
    parser.add_argument('--out_dir', type=str, default='out_llama')
    parser.add_argument('--dataset', type=str, default='c4_dataset')

    # ------------- Resume -------------
    parser.add_argument('--resume', type=str2bool, default=False, help='是否从 checkpoint 恢复训练')
    parser.add_argument('--resume_dir', type=str, default=None, help='checkpoint 目录路径（包含 model.pt）。如果为 None，将尝试从 out_dir/last 恢复')
    parser.add_argument('--resume_optimizer', type=str2bool, default=True, help='是否恢复 optimizer 状态')

    # ------------- PyTorch Profiler -------------
    parser.add_argument('--enable_profiler', type=str2bool, default=False, help='启用 PyTorch Profiler 分析性能瓶颈')
    parser.add_argument('--profiler_start_step', type=int, default=50, help='Profiler 开始记录的步数')
    parser.add_argument('--profiler_warmup_steps', type=int, default=3, help='Profiler 预热步数')
    parser.add_argument('--profiler_active_steps', type=int, default=5, help='Profiler 活跃记录步数')
    parser.add_argument('--profiler_repeat', type=int, default=1, help='Profiler 重复次数')
    parser.add_argument('--profiler_log_interval', type=int, default=500, help='Profiler 统计 log 间隔')


# ---------------------------------------------------------------------------
# Universal-only arguments
# ---------------------------------------------------------------------------
def _add_universal_only_args(parser: argparse.ArgumentParser) -> None:
    """Add the extra flags required by the multi-architecture entry point."""
    # Model auto-detection
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['llama', 'opt', 'gpt2', 'qwen', 'mistral', 'deepseek_moe', 'hunyuan', None],
                        help='Model type (auto-detected if not specified)')

    # Always save
    parser.add_argument('--always_save_checkpoint', type=str2bool, default=True,
                        help='Always save checkpoint at save_interval, regardless of PPL improvement')

    # Calibration 数据源（用于 wanda 等需要校准的 mask metric）
    parser.add_argument('--calib_source', type=str, default='c4', choices=['c4', 'synthetic'],
                        help='校准数据来源: c4 (默认原始C4) 或 synthetic (模型续写+PPL过滤)')
    parser.add_argument('--synthetic_cache_path', type=str, default=None,
                        help='synthetic 校准集的 .pkl 缓存路径 (若不指定则自动搜索 data/synthetic_calibration/)')
    parser.add_argument('--synthetic_num_samples', type=int, default=256,
                        help='synthetic 校准集: 生成的候选样本数 (过滤前)')
    parser.add_argument('--synthetic_gen_len', type=int, default=512,
                        help='synthetic 校准集: 每条样本的续写长度')
    parser.add_argument('--synthetic_prefix_min', type=int, default=1,
                        help='synthetic 校准集: C4 prefix 最短 token 数')
    parser.add_argument('--synthetic_prefix_max', type=int, default=4,
                        help='synthetic 校准集: C4 prefix 最长 token 数')
    parser.add_argument('--synthetic_ppl_filter_percent', type=float, default=0.15,
                        help='synthetic 校准集: 过滤掉 PPL 最高的比例 (0.0~1.0)')
    parser.add_argument('--synthetic_cache_dir', type=str, default=None,
                        help='synthetic 校准集: 缓存输出目录')

    # ========== Channel Pruning (结构化通道剪枝) ==========
    parser.add_argument('--enable_channel_pruning', type=str2bool, default=False, help='启用 FFN channel-level 结构化剪枝（可与 weight-level 稀疏并行使用）')
    parser.add_argument('--ffn_keep_ratio', type=float, default=0.75, help='FFN 中间层保留比例 (0, 1]')
    parser.add_argument('--channel_importance_metric', type=str, default='hessian_obd', choices=['hessian_obd', 'magnitude', 'taylor', 'wanda'], help='Channel 重要性度量方式')
    parser.add_argument('--channel_mask_update_period', type=int, default=50, help='Channel mask 更新周期（步）')
    parser.add_argument('--channel_mask_lr', type=float, default=0.1, help='Channel mask EMA 学习率')
    parser.add_argument('--channel_score_alpha', type=float, default=1.0, help='up_proj 分数权重')
    parser.add_argument('--channel_score_beta', type=float, default=1.0, help='gate_proj 分数权重')
    parser.add_argument('--channel_score_gamma', type=float, default=1.0, help='down_proj 分数权重')
    parser.add_argument('--channel_score_ema_beta', type=float, default=0.99, help='Channel score EMA 平滑系数')
    parser.add_argument('--channel_temp_init', type=float, default=1.0, help='Channel mask 初始温度')
    parser.add_argument('--channel_temp_min', type=float, default=0.05, help='Channel mask 最低温度')
    parser.add_argument('--channel_temp_decay', type=float, default=0.97, help='Channel mask 温度衰减率')
    parser.add_argument('--channel_sparsity_warmup_steps', type=int, default=500, help='Channel 稀疏度预热步数')
    parser.add_argument('--channel_hardening_start_step', type=int, default=0, help='Channel mask 硬化开始步数（0=auto，使用 60%% max_iters）')
    parser.add_argument('--channel_hardening_duration', type=int, default=5000, help='Channel mask 硬化持续步数')
    parser.add_argument('--channel_normalize_scores', type=str2bool, default=True, help='是否对 channel 分数做维度归一化')
    parser.add_argument('--channel_export_path', type=str, default=None, help='Channel pruned 模型导出路径（None=自动）')

    # SlimLLM PCA-aware Channel Scoring
    parser.add_argument('--enable_pca_scoring', type=str2bool, default=False, help='启用 SlimLLM 风格的 PCA-aware 通道重要性评分')
    parser.add_argument('--pca_lambda', type=float, default=0.5, help='PCA 分数融合权重: final = (1-λ)*base + λ*pca')
    parser.add_argument('--pca_top_k', type=int, default=0, help='PCA 使用的 top-K 主成分数（0=全部）')
    parser.add_argument('--pca_normalize_mode', type=str, default='sigmoid', choices=['sigmoid', 'softmax', 'linear'], help='PCA 特征值归一化方式')
    parser.add_argument('--pca_ema_beta', type=float, default=0.99, help='PCA 协方差矩阵 EMA 系数')
    parser.add_argument('--pca_use_activation_norm', type=str2bool, default=True, help='PCA 评分是否结合中间激活幅值')
    parser.add_argument('--pca_update_period', type=int, default=100, help='PCA 特征分解更新周期（步）')
    parser.add_argument('--pca_regression_on_hardening', type=str2bool, default=False, help='在 channel mask hardening 时执行线性回归恢复 min||Y_orig - (diag(A)*Y_pruned + B)||')

    # LoRA Bypass 补偿 Channel Pruning
    parser.add_argument('--enable_lora_bypass', type=str2bool, default=False, help='在 FFN 输出端加 LoRA bypass 补偿 channel pruning 的信息损失')
    parser.add_argument('--lora_bypass_rank', type=int, default=64, help='LoRA bypass 低秩维度 r')
    parser.add_argument('--lora_bypass_alpha', type=float, default=1.0, help='LoRA bypass 缩放因子 alpha')
    parser.add_argument('--lora_bypass_dropout', type=float, default=0.0, help='LoRA bypass dropout 比率')


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_llama_parser() -> argparse.ArgumentParser:
    """Parser used by ``main_llama.py`` (LLaMA-only entry point)."""
    parser = argparse.ArgumentParser(description="Train LLaMA with SparseLinear pruning")
    _add_common_args(parser)
    return parser


def build_universal_parser() -> argparse.ArgumentParser:
    """Parser used by ``main_universal.py`` (multi-architecture entry point)."""
    parser = argparse.ArgumentParser(description="Universal sparse training for LLaMA/OPT/GPT-2")
    _add_common_args(parser)
    _add_universal_only_args(parser)
    return parser


__all__ = [
    "str2bool",
    "build_llama_parser",
    "build_universal_parser",
]
