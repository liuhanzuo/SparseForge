"""
Universal training script supporting multiple model architectures.
Supported models: LLaMA, OPT, GPT-2, Qwen, Mistral, DeepSeek-MoE (auto-detected from model name).

Includes: FSDP, gradient accumulation, Hutchinson Hessian, mask updates, 
mid-penalty, sparsity penalties, and distillation (all losses exactly as GPT).

Examples:
  - LLaMA:   --student_model NousResearch/Llama-2-7b-hf
  - OPT:     --student_model facebook/opt-2.7b
  - GPT-2:   --student_model gpt2-medium
  - Qwen:    --student_model Qwen/Qwen2.5-1.5B
  - Mistral: --student_model mistralai/Mistral-7B-v0.3
  - DeepSeek-MoE: --student_model deepseek-ai/deepseek-moe-16b-base
"""
import os
import sys

# Ensure parent directory is on sys.path so that `sparseforge` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import math
import atexit
import traceback
from contextlib import nullcontext
from typing import Optional
import functools

# PyTorch Profiler for analyzing compute vs communication bottleneck
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from sparse_modeling import Distill_Model, SparseLinearConfig, SparseLinear
from utils import (
    calculate_model_mask,
    calculate_flip_rate,
    set_model_mode,
    initialize_model,
    add_calibration,
    eval_ppl,
    eval_ppl_distributed,
    sync_weight,
    update_model_grad_ema,
    update_hessian_hutchinson,
    mid_penalty,
    harden_fraction,
    log_mask_stats,
    get_raw_model,
    sparsity_penalty,
    nm_2_4_tile_stats,
    update_mask_penalty_lr,
)

# Channel Pruning (结构化通道剪枝)
from channel_pruning import (
    ChannelPruningConfig,
    ChannelScoreComputer,
    PCAManager,
    ChannelMaskState,
    ChannelRegressionRecovery,
    patch_model_mlp_forward,
    restore_model_mlp_forward,
    export_pruned_model,
    export_fully_pruned_model,
    compute_param_reduction,
)
from model_factory import get_sparse_model, detect_model_type, SUPPORTED_MODEL_TYPES
import torch.distributed as dist


def _dist_is_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())

def _safe_barrier() -> None:
    if not _dist_is_ready():
        return
    # Note: timeout parameter not supported in older PyTorch versions
    # Use NCCL_TIMEOUT environment variable (set in train_llama.sh) instead
    dist.barrier()


# ========== MULTI-NODE DEBUG LOGGING ==========
# Write debug messages to a shared file so we can see all nodes' output
# (since SSH redirects stdout to per-node log files)
_DEBUG_LOG_FILE = None
_DEBUG_LOG_ENABLED = False  # Set to True to enable debug file logging

def _init_debug_log():
    """Initialize per-rank debug log file in shared filesystem."""
    global _DEBUG_LOG_FILE
    if not _DEBUG_LOG_ENABLED:
        return
    try:
        trace_dir = os.environ.get('AST_TRACE_DIR', '/apdcephfs/pig_data/Adaptive-Sparse-Trainer/outputs/debug_logs')
        os.makedirs(trace_dir, exist_ok=True)
        rank = dist.get_rank() if _dist_is_ready() else 0
        world_size = dist.get_world_size() if _dist_is_ready() else 1
        node_id = rank // 8
        local_id = rank % 8
        log_path = os.path.join(trace_dir, f"rank_{rank:03d}_node{node_id}_local{local_id}.log")
        _DEBUG_LOG_FILE = open(log_path, 'w', buffering=1)  # Line buffered
        _debug_log(f"=== Debug log initialized: rank={rank}, world_size={world_size}, node={node_id}, local={local_id} ===")
    except Exception as e:
        print(f"[WARNING] Failed to init debug log: {e}", flush=True)

def _debug_log(msg: str):
    """Write debug message to log file only (no stdout to keep console clean)."""
    try:
        if _DEBUG_LOG_FILE is None:
            return  # No log file, skip entirely
        rank = dist.get_rank() if _dist_is_ready() else 0
        node_id = rank // 8
        local_id = rank % 8
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}][NODE {node_id} | RANK {rank} | LOCAL {local_id}] {msg}"
        _DEBUG_LOG_FILE.write(full_msg + "\n")
        _DEBUG_LOG_FILE.flush()
    except Exception:
        pass

def _close_debug_log():
    """Close debug log file."""
    global _DEBUG_LOG_FILE
    if _DEBUG_LOG_FILE is not None:
        try:
            _DEBUG_LOG_FILE.close()
        except:
            pass
        _DEBUG_LOG_FILE = None

# Register cleanup
atexit.register(_close_debug_log)


# Memory monitoring utility
def log_memory(label: str):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        if master_process:
            print(f"[MEMORY] {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from torch.distributed.fsdp import BackwardPrefetch  # For disabling backward prefetch
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except Exception:
        transformer_auto_wrap_policy = None
    # device_mesh for HYBRID_SHARD strategy
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        init_device_mesh = None
except Exception:
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None
    CPUOffload = None
    StateDictType = None
    FullStateDictConfig = None
    transformer_auto_wrap_policy = None
    BackwardPrefetch = None
    init_device_mesh = None

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1","true","t","yes","y","on"}:
        return True
    if s in {"0","false","f","no","n","off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")

parser = argparse.ArgumentParser(description="Universal sparse training for LLaMA/OPT/GPT-2")
parser.add_argument('--student_model', type=str, default='NousResearch/Llama-2-7b-hf')
parser.add_argument('--teacher_model', type=str, default='NousResearch/Llama-2-7b-hf')
parser.add_argument('--model_type', type=str, default=None, choices=['llama', 'opt', 'gpt2', 'qwen', 'mistral', 'deepseek_moe', 'hunyuan', None],
                    help='Model type (auto-detected if not specified)')
parser.add_argument('--distill_model', type=str2bool, default=False)
parser.add_argument('--hardness_task', type=float, default=1.0)
parser.add_argument('--hardness_kldiv', type=float, default=1.0)
parser.add_argument('--hardness_squarehead', type=float, default=1.0)
parser.add_argument('--eval_interval', type=int, default=200)
parser.add_argument('--skip_wiki_ppl', type=str2bool, default=True)  # Skip WikiText PPL by default (stability)
# Backward-compat: older launchers used --skip_eval to mean "skip wiki_ppl".
parser.add_argument('--skip_eval', type=str2bool, default=None)  # DEPRECATED alias of --skip_wiki_ppl

# lm_eval harness 评估（finalization 阶段）
parser.add_argument('--finalize_lm_eval', type=str2bool, default=False, help='在 finalization（post-finalize finetune）阶段的每次 eval 时运行 lm_eval harness，并保存 lm_eval mean accuracy 最佳的 checkpoint')
parser.add_argument('--lm_eval_tasks', type=str, default='boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa', help='lm_eval 评估的任务列表，逗号分隔')
parser.add_argument('--lm_eval_batch_size', type=int, default=4, help='lm_eval 评估的 batch size')

parser.add_argument('--save_interval', type=int, default=None)
parser.add_argument('--always_save_checkpoint', type=str2bool, default=True, help='Always save checkpoint at save_interval, regardless of PPL improvement')
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--eval_iters', type=int, default=20)
parser.add_argument('--output_flip_every', type=int, default=10)
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
parser.add_argument('--mode', choices=['sparse_forward', 'dense_forward'], default='sparse_forward')
parser.add_argument('--mask_type', choices=['structured', 'unstructured'], default='structured')
parser.add_argument('--hard_mask_type', choices=['match','unstructured','structured','block16','block_sparse16','block_sparse32','nm_2_4'], default='match')
parser.add_argument('--mask_metric', choices=['wanda','movement','hessian_obd','hessian_ratio','magnitude'], default='magnitude')
parser.add_argument('--change_mask', type=str2bool, default=False)
parser.add_argument('--beta', type=float, default=0.99)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--mask_warmup_steps', type=int, default=0)
parser.add_argument('--mask_transition_steps', type=int, default=0)
parser.add_argument('--hybrid_alpha', type=float, default=1.0)
parser.add_argument('--sparsity_ratio', type=float, default=0.5)
parser.add_argument('--SLoRB_k', type=int, default=64)
parser.add_argument('--SLoRB', type=str2bool, default=False)
parser.add_argument('--SLoRB_init_type', choices=['mean','sum','xavier'], default='mean')
parser.add_argument('--trainable_projection', type=str2bool, default=False)
parser.add_argument('--gradient_checkpointing', type=str2bool, default=False)
parser.add_argument('--wandb_logging', type=str2bool, default=False)
parser.add_argument('--lambda_mid_max', type=float, default=0.01)
parser.add_argument('--enable_hutchinson', type=str2bool, default=False)
parser.add_argument('--hardening_period', type=int, default=2000)
parser.add_argument('--hardening_fraction', type=float, default=0.2)
parser.add_argument('--hardening_band_low', type=float, default=0.4)
parser.add_argument('--hardening_band_high', type=float, default=0.6)
parser.add_argument('--wandb_project', type=str, default='LLaMA-Sparse')
parser.add_argument('--wandb_run_name', type=str, default='llama-7b')
parser.add_argument('--dtype', type=str, default='auto', choices=['auto','float16','bfloat16','float32'])
parser.add_argument('--eager_attention', type=str2bool, default=False, help='Use eager (manual) attention instead of Flash Attention. Required for Hutchinson Hessian estimation.')
parser.add_argument('--use_fsdp', type=str2bool, default=True)  # Default to True for multi-GPU to avoid OOM
parser.add_argument('--fsdp_mode', type=str, default='fully_sharded', choices=['fully_sharded','shard_grad_op','no_shard','hybrid_sharded'])
parser.add_argument('--fsdp_cpu_offload', type=str2bool, default=False)
parser.add_argument('--fsdp_mixed_precision', type=str2bool, default=True)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--mask_update_period', type=int, default=10)
parser.add_argument('--mask_update_switch_step', type=int, default=0)
parser.add_argument('--mask_update_period_before', type=int, default=None)
parser.add_argument('--mask_update_period_after', type=int, default=None)
parser.add_argument('--mask_lr', type=float, default=0.1)
parser.add_argument('--mask_penalty_lr', type=float, default=None)
parser.add_argument('--mask_penalty_lr_min', type=float, default=None, help='If set, enables penalty lr schedule: start at min, grow to mask_penalty_lr')
parser.add_argument('--mask_penalty_lr_schedule', type=str, choices=['constant', 'linear', 'cosine'], default='constant', help='Schedule for mask_penalty_lr: constant (no change), linear (from min to max), cosine (slow start fast end)')
parser.add_argument('--mask_penalty_mode', choices=['mid','structured_topn','block16','block_sparse16','block_sparse32','nm_2_4'], default='mid')
parser.add_argument('--score_ema_beta', type=float, default=0.99)
parser.add_argument('--temp_init', type=float, default=1.0)
parser.add_argument('--temp_min', type=float, default=0.05)
parser.add_argument('--temp_decay', type=float, default=0.97)
parser.add_argument('--sparsity_warmup_steps', type=int, default=0)
parser.add_argument('--structured_n', type=int, default=2)
parser.add_argument('--structured_m', type=int, default=4)
parser.add_argument('--tau_sample_size', type=int, default=262144)
parser.add_argument('--sparsity_alpha', type=float, default=0.0)
parser.add_argument('--mask_hardening_start', type=int, default=0)
parser.add_argument('--mask_hardening_duration', type=int, default=10000)
parser.add_argument('--beta_structural_start', type=int, default=0, help='step where β structural mixing starts rising from 0')
parser.add_argument('--beta_structural_end', type=int, default=0, help='step where β structural mixing reaches 1.0 (0=disabled)')
parser.add_argument('--structured_exact', type=str2bool, default=False)
parser.add_argument('--glu_joint_mask', type=str2bool, default=False, help='For GLU architectures (LLaMA/Qwen/Mistral), use joint gate/up mask to ensure aligned pruning')
parser.add_argument('--weight_scaling', type=str2bool, default=False, help='(CAST) Enable weight scaling to compensate energy loss from sparsification. Scale = M/N for structured, 1/(1-sr) for unstructured.')
parser.add_argument('--adaptive_l1_decay', type=float, default=0.0, help='(CAST AdamS) Adaptive L1 decay strength. Applied as sign(w)*lambda/denom to drive small weights toward zero. Recommended: 1e-4 ~ 1e-3.')
parser.add_argument('--final_finetune_iters', type=int, default=1000)
parser.add_argument('--freeze_low', type=float, default=0.0)
parser.add_argument('--freeze_high', type=float, default=1.0)
parser.add_argument('--use_deepspeed', action='store_true')
parser.add_argument('--deepspeed_config', type=str, default='configs/deepspeed_xl.json')
parser.add_argument('--out_dir', type=str, default='out_llama')
parser.add_argument('--dataset', type=str, default='c4_dataset')

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

# Resume training 参数
parser.add_argument('--resume', type=str2bool, default=False, help='是否从 checkpoint 恢复训练')
parser.add_argument('--resume_dir', type=str, default=None, help='checkpoint 目录路径（包含 model.pt）。如果为 None，将尝试从 out_dir/last 恢复')
parser.add_argument('--resume_optimizer', type=str2bool, default=True, help='是否恢复 optimizer 状态')

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
parser.add_argument('--pca_normalize_mode', type=str, default='sigmoid', choices=['sigmoid','softmax','linear'], help='PCA 特征值归一化方式')
parser.add_argument('--pca_ema_beta', type=float, default=0.99, help='PCA 协方差矩阵 EMA 系数')
parser.add_argument('--pca_use_activation_norm', type=str2bool, default=True, help='PCA 评分是否结合中间激活幅值')
parser.add_argument('--pca_update_period', type=int, default=100, help='PCA 特征分解更新周期（步）')
parser.add_argument('--pca_regression_on_hardening', type=str2bool, default=False, help='在 channel mask hardening 时执行线性回归恢复 min||Y_orig - (diag(A)*Y_pruned + B)||')

# LoRA Bypass 补偿 Channel Pruning
parser.add_argument('--enable_lora_bypass', type=str2bool, default=False, help='在 FFN 输出端加 LoRA bypass 补偿 channel pruning 的信息损失')
parser.add_argument('--lora_bypass_rank', type=int, default=64, help='LoRA bypass 低秩维度 r')
parser.add_argument('--lora_bypass_alpha', type=float, default=1.0, help='LoRA bypass 缩放因子 alpha')
parser.add_argument('--lora_bypass_dropout', type=float, default=0.0, help='LoRA bypass dropout 比率')

# PyTorch Profiler 参数 (用于分析 compute-bound vs communication-bound)
parser.add_argument('--enable_profiler', type=str2bool, default=False, help='启用 PyTorch Profiler 分析性能瓶颈')
parser.add_argument('--profiler_start_step', type=int, default=50, help='Profiler 开始记录的步数')
parser.add_argument('--profiler_warmup_steps', type=int, default=3, help='Profiler 预热步数')
parser.add_argument('--profiler_active_steps', type=int, default=5, help='Profiler 活跃记录步数')
parser.add_argument('--profiler_repeat', type=int, default=1, help='Profiler 重复次数')
parser.add_argument('--profiler_log_interval', type=int, default=500, help='Profiler 统计 log 间隔')

args = parser.parse_args()

# Backward-compat mapping for old flag name.
if getattr(args, 'skip_eval', None) is not None:
    args.skip_wiki_ppl = bool(args.skip_eval)

# configs
# Set block_size based on model type to respect max_position_embeddings:
# - OPT models: max_position_embeddings = 2048
# - LLaMA models: max_position_embeddings = 4096
# - GPT-2 models: max_position_embeddings = 1024
# - Qwen models: max_position_embeddings = 32768+ (use 4096 for training)
# - Mistral models: max_position_embeddings = 32768 (use 4096 for training)
# - DeepSeek MoE: max_position_embeddings = 4096
if args.model_type == 'opt':
    block_size = 2048
elif args.model_type == 'llama':
    block_size = 4096
elif args.model_type == 'gpt2':
    block_size = 1024
elif args.model_type in ['qwen', 'mistral', 'deepseek_moe', 'hunyuan']:
    block_size = 4096  # These models support longer contexts, but 4096 is practical for training
else:
    block_size = 2048  # conservative default
print(f"[CONFIG] block_size set to {block_size} for model_type={args.model_type}")
out_dir = args.out_dir
batch_size = args.batch_size
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
min_lr = args.min_lr

# config for logging/checkpointing
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# ddp setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    ddp_rank = int(os.environ['RANK']); ddp_local_rank = int(os.environ['LOCAL_RANK']); ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', ddp_local_rank)
    torch.cuda.set_device(ddp_local_rank)
    
    # Set NCCL timeout - critical for HYBRID_SHARD mode with multi-node setups
    # Default 60 min, can be overridden by NCCL_TIMEOUT env var (in seconds)
    import datetime
    nccl_timeout_sec = int(os.environ.get('NCCL_TIMEOUT', 3600))  # Default 60 min
    timeout = datetime.timedelta(seconds=nccl_timeout_sec)
    init_process_group(backend='nccl', timeout=timeout)
    
    # Print initialization success for ALL ranks (helps diagnose which nodes are lagging)
    node_id = ddp_rank // max(1, int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count())))
    print(f"[NODE {node_id} | RANK {ddp_rank}] DDP initialized: world_size={ddp_world_size}")
    sys.stdout.flush()
    
    # Initialize debug log file for this rank (writes to shared filesystem)
    _init_debug_log()
    _debug_log(f"DDP init complete: backend=nccl, world_size={ddp_world_size}")
    
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

# DEBUG: Force float32 to rule out mixed-precision issues
if os.environ.get('DEBUG_FLOAT32', '0') == '1':
    dtype = 'float32'
    if master_process:
        print("[DEBUG] DEBUG_FLOAT32=1: Forcing float32 dtype")
elif args.dtype == 'auto':
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = 'bfloat16'
    else:
        dtype = 'float16'
else:
    dtype = args.dtype
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32' or args.use_deepspeed) else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def _ddp_assert_consistent_config():
    """Fail fast if ranks disagree on critical config (common cause of FSDP collective mismatch)."""
    if not ddp:
        return
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return
    
    # 允许通过环境变量跳过此检查（对于大规模多节点环境，此检查可能因 Gloo 不稳定而超时）
    if os.environ.get("AST_SKIP_CONFIG_CHECK", "0") == "1":
        if master_process:
            print("[CONFIG CHECK] Skipped via AST_SKIP_CONFIG_CHECK=1")
        return

    # Collect a minimal but high-signal set of knobs that change control flow / FSDP behavior.
    local = {
        "rank": int(ddp_rank),
        "world_size": int(ddp_world_size),
        "student_model": str(args.student_model),
        "teacher_model": str(args.teacher_model),
        "distill_model": bool(args.distill_model),
        "use_fsdp": bool(args.use_fsdp),
        "fsdp_mode": str(args.fsdp_mode),
        "fsdp_mixed_precision": bool(args.fsdp_mixed_precision),
        "fsdp_cpu_offload": bool(args.fsdp_cpu_offload),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "eval_interval": int(args.eval_interval),
        "skip_wiki_ppl": bool(getattr(args, "skip_wiki_ppl", True)),
        "global_batch_size": int(args.global_batch_size),
        "batch_size": int(args.batch_size),
        "dtype_arg": str(args.dtype),
        "dtype_resolved": str(dtype),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "cuda_name": str(torch.cuda.get_device_name(ddp_local_rank)) if torch.cuda.is_available() else "cpu",
        "cuda_cap": tuple(torch.cuda.get_device_capability(ddp_local_rank)) if torch.cuda.is_available() else (0, 0),
    }

    # 使用 try-except 包裹 all_gather_object，在多节点大规模环境下可能因 Gloo 超时而失败
    gathered = [None for _ in range(dist.get_world_size())]
    try:
        dist.all_gather_object(gathered, local)
    except Exception as e:
        if master_process:
            print(f"[CONFIG CHECK] Warning: all_gather_object failed ({e}), skipping config consistency check.")
            print("[CONFIG CHECK] This is expected in large multi-node setups. Set AST_SKIP_CONFIG_CHECK=1 to suppress.")
        return

    # Compare against rank0 reference.
    ref = dict(gathered[0])
    ref.pop("rank", None)
    mismatches = []
    for obj in gathered:
        cur = dict(obj)
        r = cur.pop("rank", None)
        if cur != ref:
            mismatches.append((r, cur))

    if mismatches:
        if master_process:
            print("[CONFIG MISMATCH] Detected inconsistent settings across ranks. This will break FSDP collectives.")
            print(f"[CONFIG MISMATCH] Reference (rank0): {ref}")
            for r, cur in mismatches:
                print(f"[CONFIG MISMATCH] Rank {r}: {cur}")
            print("[CONFIG MISMATCH] Fix: ensure identical args/env on all ranks; if mixed GPUs, set --dtype float16 explicitly.")
        # Make sure everyone hits the same failure point.
        _safe_barrier()
        raise RuntimeError("Inconsistent distributed configuration across ranks")

    # Extra guard: if dtype was auto-resolved and some ranks can't do bf16, require explicit dtype.
    bf16_flags = [obj.get("bf16_supported", False) for obj in gathered]
    if (args.dtype == 'auto') and (min(bf16_flags) != max(bf16_flags)):
        if master_process:
            print("[CONFIG WARNING] Mixed bf16 support across ranks with --dtype auto.")
            print("[CONFIG WARNING] Please rerun with --dtype float16 (or use uniform bf16-capable GPUs).")
        _safe_barrier()
        raise RuntimeError("Mixed bf16 support across ranks with dtype=auto")


_ddp_assert_consistent_config()

# data loader (reuse wikitext memmap)
data_dir = os.path.join('data', args.dataset)

# 自动检测 dtype：检查 dtype.txt 文件或根据模型类型推断
# - Qwen/DeepSeek 等大词表模型使用 uint32
# - LLaMA/OPT/GPT-2 等传统模型使用 uint16
dtype_file = os.path.join(data_dir, 'dtype.txt')
if os.path.exists(dtype_file):
    with open(dtype_file, 'r') as f:
        dtype_name = f.read().strip()
    data_dtype = np.uint32 if dtype_name == 'uint32' else np.uint16
    if master_process:
        print(f"[DATA] Using dtype={dtype_name} from {dtype_file}")
else:
    # 根据模型类型推断 dtype
    if args.model_type in ['qwen', 'deepseek_moe']:
        data_dtype = np.uint32  # 大词表模型
    else:
        data_dtype = np.uint16  # 传统模型
    if master_process:
        print(f"[DATA] Auto-detected dtype={data_dtype.__name__} for model_type={args.model_type}")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_dtype, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_dtype, mode='r')

# Will be filled after loading the student model; used to catch invalid token ids early.
VOCAB_SIZE_CHECK = None

# ============================================================================
# Async Data Prefetcher for better GPU utilization
# ============================================================================
import threading
import queue

class AsyncDataPrefetcher:
    """
    异步数据预取器：在 GPU 计算时，后台线程预先准备下一批数据。
    这样可以减少 GPU 等待数据加载的时间，提高利用率。
    """
    def __init__(self, data, block_size, batch_size, device, prefetch_count=2):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None
        
    def _prefetch_worker(self):
        """后台线程：持续预取数据"""
        while not self.stop_event.is_set():
            try:
                # 生成随机索引
                ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
                # 从 memmap 读取数据（CPU 操作）
                x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
                # Pin memory 以加速 GPU 传输
                x = x.pin_memory()
                y = y.pin_memory()
                # 放入队列（如果满了会阻塞）
                self.queue.put((x, y), timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Prefetcher] Error: {e}")
                break
                
    def start(self):
        """启动预取线程"""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.thread.start()
            
    def stop(self):
        """停止预取线程"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
    def get_batch(self):
        """获取预取的数据批次"""
        try:
            x, y = self.queue.get(timeout=10.0)
            # 异步传输到 GPU
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            return x, y
        except queue.Empty:
            # 队列为空，回退到同步加载
            return None

# 全局预取器实例（训练循环开始后初始化）
train_prefetcher = None
PREFETCH_ENABLED = os.environ.get('DISABLE_PREFETCH', '0') != '1'

def get_batch(split):
    global train_prefetcher
    
    # 训练时使用异步预取器（如果已启用）
    if split == 'train' and PREFETCH_ENABLED and train_prefetcher is not None:
        result = train_prefetcher.get_batch()
        if result is not None:
            x, y = result
            # 检查 vocab size
            if VOCAB_SIZE_CHECK is not None:
                max_id = int(x.max().item())
                if max_id >= int(VOCAB_SIZE_CHECK):
                    raise ValueError(
                        f"Dataset token id out of range: max_id={max_id} >= vocab_size={int(VOCAB_SIZE_CHECK)}. "
                    )
            return x, y
        # 如果队列为空，回退到同步加载
    
    # 同步加载（用于验证集或预取器未启用时）
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    # Guardrail: CUDA device-side asserts often come from input_ids >= vocab_size.
    # This happens if the dataset was tokenized with a different tokenizer (e.g. GPT-2 vocab 50k)
    # but the model expects LLaMA vocab 32k.
    if VOCAB_SIZE_CHECK is not None:
        max_id = int(x.max().item())
        if max_id >= int(VOCAB_SIZE_CHECK):
            raise ValueError(
                f"Dataset token id out of range: max_id={max_id} >= vocab_size={int(VOCAB_SIZE_CHECK)}. "
                f"Your dataset '{args.dataset}' is likely tokenized for a different tokenizer. "
                f"Re-tokenize with the LLaMA tokenizer or switch to a LLaMA-tokenized dataset. "
                f"If you want to generate a compatible memmap from data/c4_dataset shards, run: "
                f"python data/c4_llama/prepare.py --tokenizer {args.student_model}"
            )
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model config
sparselinear_config = SparseLinearConfig(
    change_mask=args.change_mask,
    mask_metric=args.mask_metric,
    mask_type=args.mask_type,
    mode=args.mode,
    sparsity_ratio=args.sparsity_ratio,
    N=args.structured_n,
    M=args.structured_m,
    beta=args.beta,
    temperature=args.temperature,
    score_ema_beta=args.score_ema_beta,
    mask_update_period=args.mask_update_period,
    mask_update_switch_step=args.mask_update_switch_step,
    mask_update_period_before=args.mask_update_period_before,
    mask_update_period_after=args.mask_update_period_after,
    mask_lr=args.mask_lr,
    temp_init=args.temp_init,
    temp_min=args.temp_min,
    temp_decay=args.temp_decay,
    sparsity_warmup_steps=args.sparsity_warmup_steps,
    tau_sample_size=args.tau_sample_size,
    mask_penalty_lr=args.mask_penalty_lr,
    mask_penalty_mode=args.mask_penalty_mode,
    mask_hardening_start=args.mask_hardening_start,
    mask_hardening_duration=args.mask_hardening_duration,
    structured_exact=args.structured_exact,
    SLoRB=args.SLoRB,
    SLoRB_k=args.SLoRB_k,
    SLoRB_init_type=args.SLoRB_init_type,
    trainable_projection=args.trainable_projection,
    gradient_checkpointing=args.gradient_checkpointing,
    freeze_low=args.freeze_low,
    freeze_high=args.freeze_high,
    srste_decay=args.srste_decay,
    hard_mask_type=args.hard_mask_type,
    beta_structural_start=args.beta_structural_start,
    beta_structural_end=args.beta_structural_end,
    glu_joint_mask=args.glu_joint_mask,
    weight_scaling=args.weight_scaling,
)

override_args = dict(dropout=0.0, output_hidden_state=False, gradient_checkpointing=args.gradient_checkpointing, eager_attention=args.eager_attention)

# Hutchinson Hessian estimation requires second-order derivatives,
# but Flash Attention (SDPA) backward doesn't support them.
# However, Hutchinson already has graceful error handling (try/catch + all_reduce skip),
# so we keep Flash Attention enabled for better performance and FSDP compatibility.
# Hutchinson steps that fail double-backward will be silently skipped.
if args.enable_hutchinson and args.eager_attention and master_process:
    print("[Hutchinson] Enabled with eager attention. Full Hessian estimation supported.")
elif args.enable_hutchinson and not args.eager_attention and master_process:
    print("[Hutchinson] Enabled with Flash Attention (SDPA). Steps requiring 2nd-order grad will be skipped gracefully.")

# 检测并显示模型类型
detected_model_type = args.model_type or detect_model_type(args.student_model)
if master_process:
    print(f"[INFO] Model type: {detected_model_type} (auto-detected)" if args.model_type is None 
          else f"[INFO] Model type: {detected_model_type} (specified)")

# ============================================================================
# STAGE 1: Load student to CPU + initialize buffers
# ============================================================================
if master_process:
    print("=" * 80)
    print("MEMORY ALLOCATION SEQUENCE:")
    print("=" * 80)
    print("STAGE 1: Load student to CPU + initialize buffers...")
    log_memory("[S1-BEFORE] student load")

# ==========================================================================
# Broadcast model loading for multi-node training
# Strategy: Rank 0 loads from disk once, then broadcasts state_dict to all
# other ranks. This avoids repeated cephfs I/O (N nodes × disk read).
# Each non-rank-0 process builds model skeleton + loads broadcasted weights.
# ==========================================================================
if ddp and ddp_world_size > 1:
    from model_factory import broadcast_state_dict, create_model_skeleton
    import time as _time
    
    if master_process:
        print(f"[LOAD] Broadcast loading: rank 0 loads, then broadcasts to {ddp_world_size - 1} ranks")
        _t0 = _time.time()
    
    # Step 1: Rank 0 loads model from disk (only 1 disk read across entire cluster)
    if ddp_rank == 0:
        print(f"[LOAD] Rank 0 loading model from disk...")
        student_model = get_sparse_model(
            args.student_model,
            model_type=args.model_type,
            override_args=override_args,
            sparselinear_config=sparselinear_config,
            is_teacher=False,
        )
        print(f"[LOAD] Rank 0 finished loading, broadcasting state_dict...")
        sd = student_model.state_dict()
    else:
        sd = None
    
    _safe_barrier()
    
    # Step 2: Broadcast state_dict from rank 0 to all ranks
    if master_process:
        print(f"[LOAD] Broadcasting state_dict to all ranks...")
    sd = broadcast_state_dict(sd, src=0)
    
    # Step 3: Non-rank-0 processes create model skeleton (no disk I/O) and load broadcasted weights
    if ddp_rank != 0:
        student_model = create_model_skeleton(
            args.student_model,
            model_type=args.model_type,
            override_args=override_args,
            sparselinear_config=sparselinear_config,
            is_teacher=False,
        )
        student_model.load_state_dict(sd, strict=False)
    
    # 释放临时 state_dict
    del sd
    import gc as _gc
    _gc.collect()
    
    _safe_barrier()
    if master_process:
        _elapsed = _time.time() - _t0
        print(f"[LOAD] All ranks finished loading student model ({_elapsed:.1f}s)")
else:
    # Single process: load normally
    student_model = get_sparse_model(
        args.student_model,
        model_type=args.model_type,
        override_args=override_args,
        sparselinear_config=sparselinear_config,
        is_teacher=False,
    )

# Fill vocab size check after we know which model/tokenizer we're using.
# This is crucial for catching tokenizer/model mismatch errors early (before CUDA device-side asserts).
VOCAB_SIZE_CHECK = None
try:
    # Try different access patterns for various model wrappers
    # 1. OPTSparse/LLaMAWrapper: student_model.model.config.vocab_size
    # 2. Direct HF model: student_model.config.vocab_size
    # 3. Some wrappers have .config directly
    if hasattr(student_model, 'config') and hasattr(student_model.config, 'vocab_size'):
        VOCAB_SIZE_CHECK = int(student_model.config.vocab_size)
    elif hasattr(student_model, 'model') and hasattr(student_model.model, 'config') and hasattr(student_model.model.config, 'vocab_size'):
        VOCAB_SIZE_CHECK = int(student_model.model.config.vocab_size)
    elif hasattr(student_model, 'model') and hasattr(student_model.model, 'model') and hasattr(student_model.model.model, 'config'):
        # Nested model wrappers (e.g., FSDP wrapped)
        VOCAB_SIZE_CHECK = int(student_model.model.model.config.vocab_size)
except Exception as e:
    if master_process:
        print(f"   ⚠ Warning: Could not determine vocab_size from model config: {e}")
    VOCAB_SIZE_CHECK = None

if VOCAB_SIZE_CHECK is not None:
    if master_process:
        print(f"   ✓ VOCAB_SIZE_CHECK set to {VOCAB_SIZE_CHECK}")
    
    # ==========================================================================
    # CRITICAL: Pre-validate dataset token IDs before training starts
    # This catches tokenizer/model mismatch early (before CUDA device-side asserts)
    # ==========================================================================
    if master_process:
        print("   → Validating dataset token IDs against model vocab_size...")
        
        # Sample validation: check first/last chunks + random samples
        # For large datasets, full validation is too slow, so we sample
        sample_positions = [0]  # Always check beginning
        val_len = len(val_data)
        train_len = len(train_data)
        
        # Validate entire val set (usually small enough)
        val_max = int(val_data[:min(val_len, 10_000_000)].max())
        if val_max >= VOCAB_SIZE_CHECK:
            raise ValueError(
                f"❌ FATAL: Validation dataset has token ID {val_max} >= vocab_size {VOCAB_SIZE_CHECK}!\n"
                f"   Your dataset '{args.dataset}' was tokenized with a DIFFERENT tokenizer.\n"
                f"   Model expects vocab_size={VOCAB_SIZE_CHECK}, but dataset contains tokens up to {val_max}.\n"
                f"   → Re-tokenize the dataset with the correct tokenizer for {args.student_model}."
            )
        
        # Sample check on train set (check beginning, middle, end)
        check_size = min(1_000_000, train_len // 10)
        train_samples = [
            train_data[:check_size],  # Beginning
            train_data[train_len//2 : train_len//2 + check_size],  # Middle
            train_data[-check_size:],  # End
        ]
        for i, sample in enumerate(train_samples):
            sample_max = int(sample.max())
            if sample_max >= VOCAB_SIZE_CHECK:
                raise ValueError(
                    f"❌ FATAL: Training dataset has token ID {sample_max} >= vocab_size {VOCAB_SIZE_CHECK}!\n"
                    f"   Found in sample region {i} (0=start, 1=middle, 2=end).\n"
                    f"   Your dataset '{args.dataset}' was tokenized with a DIFFERENT tokenizer.\n"
                    f"   → Re-tokenize the dataset with the correct tokenizer for {args.student_model}."
                )
        print(f"   ✓ Dataset validation passed (val_max={val_max}, train_samples_max={max(int(s.max()) for s in train_samples)})")
else:
    if master_process:
        print(f"   ⚠ WARNING: VOCAB_SIZE_CHECK is None! Token ID validation disabled.")
        print(f"      This may cause CUDA device-side asserts if dataset tokenizer doesn't match model.")

if master_process:
    log_memory("[S1-AFTER-LOAD] student loaded to CPU")
    print("   ✓ Student loaded to CPU (14GB system RAM)")

# Enable gradient checkpointing to reduce activation memory
# IMPORTANT: 必须使用 use_reentrant=False！
# 新版 transformers (4.50+) 中 GradientCheckpointingLayer 默认 use_reentrant=True，
# 在 FSDP 下会导致 backward recompute 时各 rank 的 all_gather 顺序不一致，
# 引发 NCCL 集合操作不匹配 → 600s 超时死锁。
if args.gradient_checkpointing:
    gc_kwargs = {"use_reentrant": False}
    if hasattr(student_model, 'model') and hasattr(student_model.model, 'gradient_checkpointing_enable'):
        student_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        if master_process:
            print("   ✓ Gradient checkpointing enabled for student model (use_reentrant=False)")
    elif hasattr(student_model, 'gradient_checkpointing_enable'):
        student_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        if master_process:
            print("   ✓ Gradient checkpointing enabled for student model (use_reentrant=False)")

student_model.eval()
initialize_model(student_model)
student_model.train()

# CRITICAL DEBUG: Check for NaN/Inf in model parameters and buffers after initialization
if master_process:
    nan_params = []
    inf_params = []
    for name, param in student_model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    for name, buf in student_model.named_buffers():
        if torch.isnan(buf).any():
            nan_params.append(f"buffer:{name}")
        if torch.isinf(buf).any():
            inf_params.append(f"buffer:{name}")
    if nan_params:
        print(f"[CRITICAL] NaN detected in params/buffers after initialize_model: {nan_params}")
    if inf_params:
        print(f"[CRITICAL] Inf detected in params/buffers after initialize_model: {inf_params}")
    if not nan_params and not inf_params:
        print("   ✓ No NaN/Inf in model parameters after initialization")

if master_process:
    log_memory("[S1-FINAL] student buffers initialized on CPU")
    print("   ✓ Buffers created on CPU (grad_ema, hessian_diag, mask...)")
    print("-" * 80)

# ============================================================================
# STAGE 1.5 (RESUME): Load checkpoint weights BEFORE FSDP wrap
# This is CRITICAL for memory efficiency! Loading into FSDP-wrapped model
# requires unshard which causes OOM. Loading before FSDP avoids this.
# ============================================================================
_resume_ckpt_loaded = False  # Flag to track if we loaded checkpoint in stage 1.5
_resume_iter_num = 0
_resume_best_wiki_ppl = 1e9
_resume_optimizer_state = None
_resume_scaler_state = None

if args.resume:
    if master_process:
        print("=" * 80)
        print("STAGE 1.5 (RESUME): Load checkpoint weights BEFORE FSDP wrap...")
        print("=" * 80)
    
    resume_ckpt_dir = args.resume_dir
    if resume_ckpt_dir is None:
        # 尝试从 out_dir/last 符号链接恢复
        last_link = os.path.join(out_dir, "last")
        last_dir_file = os.path.join(out_dir, "last_dir.txt")
        if os.path.islink(last_link) or os.path.isdir(last_link):
            resume_ckpt_dir = os.path.realpath(last_link)
        elif os.path.exists(last_dir_file):
            with open(last_dir_file, "r") as f:
                resume_ckpt_dir = f.read().strip()
    
    if resume_ckpt_dir and os.path.isdir(resume_ckpt_dir):
        model_pt_path = os.path.join(resume_ckpt_dir, "model.pt")
        if os.path.exists(model_pt_path):
            if master_process:
                print(f"[RESUME-S1.5] Loading checkpoint from {model_pt_path}")
            
            # 加载 checkpoint 到 CPU
            resume_ckpt = torch.load(model_pt_path, map_location="cpu", weights_only=False)
            
            # 恢复模型权重到 student_model（FSDP wrap 之前！）
            if "model_state_dict" in resume_ckpt:
                ckpt_sd = resume_ckpt["model_state_dict"]
                
                if master_process:
                    ckpt_keys = list(ckpt_sd.keys())
                    target_keys = list(student_model.state_dict().keys())
                    print(f"[RESUME-S1.5] Checkpoint keys sample: {ckpt_keys[:3]}")
                    print(f"[RESUME-S1.5] Target (student_model) keys sample: {target_keys[:3]}")
                
                # 直接加载到 student_model（未被 FSDP wrap）
                try:
                    missing, unexpected = student_model.load_state_dict(ckpt_sd, strict=False)
                    _resume_ckpt_loaded = True
                    if master_process:
                        if missing:
                            print(f"[RESUME-S1.5] Missing keys ({len(missing)}): {missing[:5]}..." if len(missing) > 5 else f"[RESUME-S1.5] Missing keys: {missing}")
                        if unexpected:
                            print(f"[RESUME-S1.5] Unexpected keys ({len(unexpected)}): {unexpected[:5]}..." if len(unexpected) > 5 else f"[RESUME-S1.5] Unexpected keys: {unexpected}")
                        print(f"[RESUME-S1.5] ✓ Model weights loaded to student_model (before FSDP wrap)")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-S1.5] Warning: load_state_dict failed: {e}")
                
                # 释放 checkpoint state_dict 内存
                del ckpt_sd
            
            # 保存 optimizer 和 scaler state 供后续使用（在 optimizer 创建后加载）
            if args.resume_optimizer and "optimizer_state_dict" in resume_ckpt:
                _resume_optimizer_state = resume_ckpt["optimizer_state_dict"]
                if master_process:
                    print(f"[RESUME-S1.5] Optimizer state saved for later restoration")
            
            if "scaler_state_dict" in resume_ckpt:
                _resume_scaler_state = resume_ckpt["scaler_state_dict"]
                if master_process:
                    print(f"[RESUME-S1.5] Scaler state saved for later restoration")
            
            # 恢复 iter_num
            if "iter_num" in resume_ckpt:
                _resume_iter_num = int(resume_ckpt["iter_num"])
                if master_process:
                    print(f"[RESUME-S1.5] Will resume from iter_num={_resume_iter_num}")
            
            # 尝试恢复 best_wiki_ppl
            eval_json_path = os.path.join(resume_ckpt_dir, "eval.json")
            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, "r") as f:
                        eval_data = json.load(f)
                    if "wiki_ppl" in eval_data and eval_data["wiki_ppl"] is not None:
                        _resume_best_wiki_ppl = float(eval_data["wiki_ppl"])
                        if master_process:
                            print(f"[RESUME-S1.5] Restored best_wiki_ppl={_resume_best_wiki_ppl}")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-S1.5] Warning: failed to load eval.json: {e}")
            
            # 清理 checkpoint 内存（保留 optimizer/scaler state）
            del resume_ckpt
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            if master_process:
                print(f"[RESUME-S1.5] ✓ Checkpoint loaded, memory cleaned")
                print("-" * 80)
        else:
            if master_process:
                print(f"[RESUME-S1.5] Warning: model.pt not found in {resume_ckpt_dir}")
    else:
        if master_process:
            print(f"[RESUME-S1.5] Warning: resume_dir not found or invalid: {resume_ckpt_dir}")

# ============================================================================
# STAGE 2: FSDP-wrap student from CPU (no .to(cuda) before wrap)
# ============================================================================
if master_process:
    print("STAGE 2: FSDP-wrap student from CPU (no pre-move to GPU)...")
    # Sanity check: ensure parameters are on CPU before FSDP
    try:
        any_on_cuda = any(p.is_cuda for p in student_model.parameters())
        if any_on_cuda:
            print("[WARN] Student has CUDA params before FSDP wrap; this increases peak memory.")
    except Exception:
        pass
    log_memory("[S2-BEFORE] FSDP wrap (student on CPU)")

fsdp_requested = bool(args.use_fsdp)

# Prepare FSDP config up-front so distill mode can wrap in Stage 5.
sharding = None
cpu_offload = None
mp = None
auto_wrap_policy_student = None

if fsdp_requested:
    # Setup FSDP config
    if FSDP is None:
        raise RuntimeError("FSDP not available")
    mode = args.fsdp_mode
    # Map fsdp_mode to ShardingStrategy
    # - 'fully_sharded': FULL_SHARD (shard across ALL ranks globally - most memory efficient but slow cross-node)
    # - 'hybrid_shard': HYBRID_SHARD (shard within node, replicate across nodes - RECOMMENDED for multi-node)
    #   Uses NVLink (400GB/s) for intra-node AllGather, only AllReduce gradients across nodes
    # - 'shard_grad_op': SHARD_GRAD_OP (simpler collectives)
    # - 'no_shard': NO_SHARD (DDP-like behavior)
    if mode == 'fully_sharded':
        sharding = ShardingStrategy.FULL_SHARD
    elif mode == 'hybrid_sharded':
        # HYBRID_SHARD: shard within each node (8 GPUs), replicate across nodes
        # This significantly reduces cross-node communication:
        # - Forward: AllGather within node only (NVLink, ~400GB/s)
        # - Backward: ReduceScatter within node + AllReduce across nodes
        # Performance gain: 20-50% faster than FULL_SHARD for multi-node training
        sharding = ShardingStrategy.HYBRID_SHARD
        # HYBRID_SHARD requires device_mesh to define intra-node and inter-node topology
        fsdp_device_mesh = None
        if init_device_mesh is not None and ddp_world_size > 1:
            # Determine nodes and GPUs per node
            gpus_per_node = int(os.environ.get('NPROC_PER_NODE', 8))
            num_nodes = ddp_world_size // gpus_per_node
            if num_nodes < 1:
                num_nodes = 1
                gpus_per_node = ddp_world_size
            if master_process:
                print(f"   Creating device_mesh for HYBRID_SHARD: {num_nodes} nodes × {gpus_per_node} GPUs")
            # Create 2D mesh: (num_nodes, gpus_per_node) -> ("replicate", "shard")
            # NOTE: init_device_mesh may fail on older PyTorch versions that don't support split_group
            try:
                fsdp_device_mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node), mesh_dim_names=("replicate", "shard"))
            except RuntimeError as e:
                if master_process:
                    print(f"   ⚠ init_device_mesh failed: {e}")
                    print(f"   ⚠ Falling back to FULL_SHARD (your PyTorch version may not support HYBRID_SHARD)")
                fsdp_device_mesh = None
                sharding = ShardingStrategy.FULL_SHARD
        else:
            if master_process:
                print(f"   ⚠ init_device_mesh not available or single GPU, falling back to FULL_SHARD")
            sharding = ShardingStrategy.FULL_SHARD
            fsdp_device_mesh = None
    elif mode == 'shard_grad_op':
        sharding = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding = ShardingStrategy.NO_SHARD
    # Ensure fsdp_device_mesh is defined for non-hybrid modes
    if 'fsdp_device_mesh' not in dir():
        fsdp_device_mesh = None
    if master_process:
        print(f"   FSDP sharding strategy: {sharding} (from fsdp_mode='{mode}')")
        if fsdp_device_mesh is not None:
            print(f"   device_mesh: {fsdp_device_mesh}")
    cpu_offload = CPUOffload(offload_params=bool(args.fsdp_cpu_offload))
    mp = None
    # FSDP MixedPrecision: enable for memory efficiency if requested
    if bool(args.fsdp_mixed_precision) and dtype != 'float32' and MixedPrecision is not None:
        mp = MixedPrecision(param_dtype=ptdtype, reduce_dtype=ptdtype, buffer_dtype=ptdtype)
        if master_process:
            print(f"   FSDP MixedPrecision enabled: param_dtype={ptdtype}, reduce_dtype={ptdtype}")
    
    # Try to get auto_wrap_policy for student
    # CRITICAL: auto_wrap_policy is REQUIRED for cpu_offload to work properly.
    # Without auto_wrap_policy, FSDP wraps the entire model as one unit,
    # preventing cpu_offload from selectively offloading layers.
    if transformer_auto_wrap_policy is not None:
        try:
            block_cls = None
            # Try LLaMA structure: model.model.layers
            if hasattr(student_model, 'model') and hasattr(student_model.model, 'model') and hasattr(student_model.model.model, 'layers'):
                block_cls = student_model.model.model.layers[0].__class__
                if master_process:
                    print(f"   Detected LLaMA-style layers at model.model.model.layers")
            # Try OPT structure: model.model.decoder.layers
            elif hasattr(student_model, 'model') and hasattr(student_model.model, 'model') and hasattr(student_model.model.model, 'decoder') and hasattr(student_model.model.model.decoder, 'layers'):
                block_cls = student_model.model.model.decoder.layers[0].__class__
                if master_process:
                    print(f"   Detected OPT-style layers at model.model.model.decoder.layers")
            # Try GPT-2 structure: transformer.h
            elif hasattr(student_model, 'transformer') and hasattr(student_model.transformer, 'h'):
                block_cls = student_model.transformer.h[0].__class__
                if master_process:
                    print(f"   Detected GPT2-style layers at transformer.h")
            
            if block_cls is not None:
                # transformer_auto_wrap_policy is a predicate; use functools.partial to bind layer classes.
                auto_wrap_policy_student = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={block_cls},
                )
                if master_process:
                    print(f"   ✓ Using transformer_auto_wrap_policy with layer class: {block_cls.__name__}")
        except Exception as e:
            if master_process:
                print(f"   ⚠ transformer_auto_wrap_policy setup failed: {e}, trying fallback...")
    
    # Fallback: use size_based_auto_wrap_policy if transformer_auto_wrap_policy failed or is None
    # This ensures cpu_offload works correctly by wrapping each layer individually
    if auto_wrap_policy_student is None:
        try:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            # Wrap each transformer layer separately to enable effective cpu_offload
            # Size limit should be around the size of one transformer layer (~100-200M parameters for LLaMA-7B)
            auto_wrap_policy_student = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=100_000_000,  # ~100M params per layer for LLaMA-7B
            )
            if master_process:
                print(f"   ✓ Using fallback size_based_auto_wrap_policy (min_num_params=100M)")
        except Exception as e:
            if master_process:
                print(f"   ⚠ Fallback auto_wrap_policy failed: {e}")
                print(f"   ⚠ WARNING: cpu_offload may not work correctly without auto_wrap_policy!")
    
    if master_process:
        print(f"   FSDP config: sharding={sharding}, cpu_offload={bool(args.fsdp_cpu_offload)}, mixed_precision={mp is not None}")
    
    torch.cuda.set_device(ddp_local_rank)
    device_id = ddp_local_rank
    
    # CRITICAL: Initialize CUDA context explicitly before barrier
    # This prevents "Cuda failure 802 'system not yet initialized'" errors in NCCL
    # In multi-node environments, some ranks may not have initialized CUDA context yet
    try:
        # Create a small tensor to force CUDA initialization
        _ = torch.zeros(1, device=f'cuda:{device_id}')
        torch.cuda.synchronize(device_id)
    except Exception as e:
        if master_process:
            print(f"   ⚠ Warning: CUDA initialization failed: {e}")
    
    # CRITICAL: Clean up memory before FSDP wrapping to prevent OOM
    # This is especially important when using cpu_offload as FSDP initialization needs memory
    if master_process:
        print(f"   → Cleaning up memory before FSDP wrapping...")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Log memory usage before FSDP
    if master_process:
        try:
            mem_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            print(f"   GPU memory before FSDP: allocated={mem_allocated:.2f}GB reserved={mem_reserved:.2f}GB")
        except Exception:
            pass
    
    # CRITICAL: Barrier BEFORE FSDP wrap to ensure all ranks are ready.
    # FSDP.__init__ with sync_module_states=True triggers a GATHER collective;
    # if any rank is lagging behind, the GATHER will hang or mismatch.
    if ddp:
        print(f"[RANK {ddp_rank}] Ready to start FSDP wrap, waiting for barrier...")
        import sys
        sys.stdout.flush()
        _safe_barrier()
        print(f"[RANK {ddp_rank}] Barrier passed, starting FSDP wrap...")
        sys.stdout.flush()
    
    student_wrap_error = None
    try:
        # NOTE: sync_module_states=False because ALL ranks load from the same pretrained
        # checkpoint (NousResearch/Llama-2-7b-hf). This avoids the internal GATHER/BROADCAST
        # that sync_module_states=True triggers, which can hang in multi-node environments
        # due to network delays or memory pressure.
        # NOTE: limit_all_gathers=False for debugging - setting to True can cause
        # collective ordering issues in some multi-node setups.
        # backward_prefetch options:
        # - BACKWARD_PRE (default): aggressive prefetching, best performance
        # - BACKWARD_POST: more conservative, triggers all_gather after reduce_scatter completes
        # Use BACKWARD_POST for multi-node stability
        bwd_prefetch = BackwardPrefetch.BACKWARD_POST if BackwardPrefetch is not None else None
        
        # Build FSDP kwargs
        fsdp_kwargs = dict(
            sharding_strategy=sharding,
            auto_wrap_policy=auto_wrap_policy_student,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            device_id=device_id,        # FSDP will materialize only this rank's shard to this GPU
            use_orig_params=True,
            sync_module_states=False,   # All ranks have identical weights from pretrained checkpoint
            limit_all_gathers=True,     # Limit outstanding all_gathers to reduce memory spikes
            forward_prefetch=True,      # Enable forward prefetch for performance
            backward_prefetch=bwd_prefetch,  # BACKWARD_POST for multi-node stability
        )
        # Add device_mesh for HYBRID_SHARD
        if fsdp_device_mesh is not None and sharding == ShardingStrategy.HYBRID_SHARD:
            fsdp_kwargs['device_mesh'] = fsdp_device_mesh
        student_model = FSDP(
            student_model,              # still on CPU here
            **fsdp_kwargs
        )
        # FSDP wrap completed - debug output disabled for clean logs
        if master_process:
            log_memory("[S2-AFTER-FSDP] student FSDP wrapped")
            print(f"   ✓ Student sharded (14GB → {14/ddp_world_size:.2f}GB per GPU)")
        using_fsdp = True
    except Exception as e:
        student_wrap_error = str(e)
        print(f"[RANK {ddp_rank}] FSDP wrapping failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()

    # CRITICAL: Synchronize all ranks after FSDP wrap to ensure consistency.
    # If ANY rank failed, we must abort cleanly to avoid collective mismatch.
    if ddp:
        local_ok = 1 if (student_wrap_error is None) else 0
        ok_tensor = torch.tensor([local_ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        all_ok = int(ok_tensor.item()) == 1
        if not all_ok:
            print(f"[RANK {ddp_rank}] Student FSDP wrap FAILED: {student_wrap_error}")
            _safe_barrier()
            raise RuntimeError(
                "Student FSDP wrap failed on at least one rank. "
                "Aborting to avoid FSDP collective mismatch. "
                "(Tip: this is often OOM or inconsistent module structure.)"
            )
        # Explicit barrier after successful FSDP wrap to ensure all ranks are synchronized
        _safe_barrier()
        if master_process:
            print("   ✓ All ranks synchronized after student FSDP wrap")
    elif student_wrap_error is not None:
        raise RuntimeError(f"Student FSDP wrap failed: {student_wrap_error}")

    model = student_model
else:
    torch.cuda.empty_cache()
    student_model = student_model.to(device)
    if master_process:
        log_memory("[S2-AFTER-GPU] student on GPU (non-FSDP)")
        print("   ✓ Student moved to GPU")
    using_fsdp = False
    model = student_model

if master_process:
    print("-" * 80)

# ============================================================================
# STAGE 3 (Optional): Load teacher + FSDP-wrap to GPU
# ============================================================================
teacher_model = None
if args.distill_model:
    if master_process:
        print("STAGE 3: Load teacher to CPU + FSDP-wrap to GPU...")
        log_memory("[S3-BEFORE] teacher load")
    
    # Broadcast teacher loading (same strategy as student)
    if ddp and ddp_world_size > 1:
        from model_factory import broadcast_state_dict, create_model_skeleton
        import time as _time
        
        if master_process:
            print(f"[LOAD] Broadcast teacher loading: rank 0 loads, then broadcasts")
            _t0 = _time.time()
        
        # Step 1: Rank 0 loads teacher from disk
        if ddp_rank == 0:
            print(f"[LOAD] Rank 0 loading teacher model from disk...")
            teacher_model = get_sparse_model(
                args.teacher_model,
                model_type=args.model_type,
                override_args=override_args,
                sparselinear_config=None,
                is_teacher=True,
            )
            print(f"[LOAD] Rank 0 finished loading teacher, broadcasting state_dict...")
            sd = teacher_model.state_dict()
        else:
            sd = None
        
        _safe_barrier()
        
        # Step 2: Broadcast state_dict
        if master_process:
            print(f"[LOAD] Broadcasting teacher state_dict to all ranks...")
        sd = broadcast_state_dict(sd, src=0)
        
        # Step 3: Non-rank-0 processes create teacher skeleton and load broadcasted weights
        if ddp_rank != 0:
            teacher_model = create_model_skeleton(
                args.teacher_model,
                model_type=args.model_type,
                override_args=override_args,
                sparselinear_config=None,
                is_teacher=True,
            )
            teacher_model.load_state_dict(sd, strict=False)
        
        del sd
        import gc as _gc
        _gc.collect()
        
        _safe_barrier()
        if master_process:
            _elapsed = _time.time() - _t0
            print(f"[LOAD] All ranks finished loading teacher model ({_elapsed:.1f}s)")
    else:
        teacher_model = get_sparse_model(
            args.teacher_model,
            model_type=args.model_type,
            override_args=override_args,
            sparselinear_config=None,
            is_teacher=True,
        )
    
    if master_process:
        log_memory("[S3-AFTER-LOAD] teacher loaded to CPU")
        print("   ✓ Teacher loaded to CPU (14GB system RAM)")
    
    # IMPORTANT: Teacher does NOT need gradient_checkpointing!
    # Teacher runs in torch.no_grad() mode during forward, so it never needs to
    # recompute activations for backward. Enabling gradient_checkpointing on teacher
    # under FSDP can cause NCCL collective mismatches because:
    # 1. Checkpoint hooks may trigger unexpected all_gather operations
    # 2. use_reentrant=True (default in older transformers) interacts badly with
    #    FSDP's parameter sharding under torch.no_grad()
    # This is the root cause of LLaMA training hanging at the first iteration
    # (eval uses student-only and succeeds, but training uses student+teacher).
    if args.gradient_checkpointing:
        if master_process:
            print("   ⓘ Skipping gradient_checkpointing for teacher (no backward needed)")
    
    teacher_model.eval()

    # Freeze teacher parameters (no optimizer / no grads) — applies to BOTH FSDP and non-FSDP.
    # CRITICAL: This must happen BEFORE FSDP wrapping. Under FSDP, if teacher params have
    # requires_grad=True, FSDP may schedule reduce_scatter for teacher gradients during
    # backward, causing collective mismatch with student's FSDP process group.
    try:
        teacher_model.requires_grad_(False)
    except Exception:
        for p in teacher_model.parameters():
            p.requires_grad_(False)
    if master_process:
        print("   ✓ Teacher parameters frozen (requires_grad=False)")

    if not fsdp_requested:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    
    # Wrap teacher with FSDP too (frozen, no optimizer needed)
    # IMPORTANT: All ranks must apply the same wrapping strategy to keep collectives in sync!
    teacher_wrap_error = None
    if fsdp_requested:
        if master_process:
            print("   → FSDP-wrapping teacher (frozen parameters)...")
        
        # Get auto_wrap_policy for teacher (same as student)
        auto_wrap_policy_teacher = None
        if transformer_auto_wrap_policy is not None:
            try:
                block_cls = None
                # Try LLaMA structure
                if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'layers'):
                    block_cls = teacher_model.model.model.layers[0].__class__
                # Try OPT structure
                elif hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'decoder') and hasattr(teacher_model.model.model.decoder, 'layers'):
                    block_cls = teacher_model.model.model.decoder.layers[0].__class__
                # Try GPT-2 structure
                elif hasattr(teacher_model, 'transformer') and hasattr(teacher_model.transformer, 'h'):
                    block_cls = teacher_model.transformer.h[0].__class__
                
                if block_cls is not None:
                    auto_wrap_policy_teacher = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={block_cls},
                    )
            except Exception as e:
                if master_process:
                    print(f"   [WARN] Failed to get auto_wrap_policy: {e}")
        
        try:
            # NOTE: sync_module_states=False because ALL ranks load from the same pretrained
            # checkpoint. This avoids the internal GATHER/BROADCAST that can hang in
            # multi-node environments.
            # Build FSDP kwargs for teacher
            teacher_fsdp_kwargs = dict(
                sharding_strategy=sharding,
                auto_wrap_policy=auto_wrap_policy_teacher,
                mixed_precision=mp,
                cpu_offload=cpu_offload,
                device_id=device_id,
                use_orig_params=True,
                sync_module_states=False,   # All ranks have identical weights from pretrained checkpoint
                limit_all_gathers=True
            )
            # Add device_mesh for HYBRID_SHARD
            if fsdp_device_mesh is not None and sharding == ShardingStrategy.HYBRID_SHARD:
                teacher_fsdp_kwargs['device_mesh'] = fsdp_device_mesh
            teacher_model = FSDP(
                teacher_model,
                **teacher_fsdp_kwargs
            )
            if master_process:
                log_memory("[S3-AFTER-FSDP] teacher FSDP wrapped")
                print(f"   ✓ Teacher sharded to GPU (14GB → {14/ddp_world_size:.2f}GB per GPU)")
        except Exception as e:
            if master_process:
                print(f"[ERROR] Teacher FSDP wrap failed on rank {ddp_rank}: {e}")
                import traceback
                traceback.print_exc()
            teacher_wrap_error = str(e)

    # Make teacher wrapping decision CONSISTENT across all ranks.
    # If any rank fails, we must not proceed with a mixed teacher state (some FSDP, some not),
    # otherwise FSDP collectives will mismatch during forward/backward.
    if ddp:
        import torch.distributed as dist
        local_ok = 1 if (teacher_wrap_error is None) else 0
        ok_tensor = torch.tensor([local_ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        all_ok = int(ok_tensor.item()) == 1
        if not all_ok:
            # Print the local error on each rank to aid debugging.
            print(f"[RANK {ddp_rank}] Teacher FSDP wrap FAILED: {teacher_wrap_error}")
            # Ensure everyone reaches the same failure point before raising.
            _safe_barrier()
            raise RuntimeError(
                "Teacher FSDP wrap failed on at least one rank. "
                "Aborting to avoid FSDP collective mismatch. "
                "(Tip: this is often OOM or inconsistent CUDA env on some node.)"
            )
        using_fsdp_teacher = bool(fsdp_requested)
        _safe_barrier()
    else:
        # Single-process case
        if teacher_wrap_error is not None:
            raise RuntimeError(f"Teacher FSDP wrap failed: {teacher_wrap_error}")
        using_fsdp_teacher = bool(fsdp_requested)
    
    if master_process:
        log_memory("[S3-FINAL] teacher ready")
        print("   ✓ Teacher ready")
        print("-" * 80)
else:
    using_fsdp_teacher = False

# ============================================================================
# STAGE 3.5: DDP wrap student BEFORE creating Distill_Model (non-FSDP only)
# ============================================================================
# IMPORTANT: distill + non-FSDP under torchrun MUST use DDP for student gradient sync.
# We wrap the Distill_Model with DDP after it is created (Stage 4) to match main.py.
if ddp and (not using_fsdp) and teacher_model is not None:
    if master_process:
        print("STAGE 3.5: Will DDP-wrap Distill_Model (distill + non-FSDP)...")
        print("   ✓ DDP will synchronize student gradients across ranks")
        print("-" * 80)

# ============================================================================
# STAGE 4: Create Distill_Model wrapper (if distillation enabled)
# ============================================================================
if args.distill_model:
    # ========================================================================
    # Create Distill_Model wrapper with (GPU sharded student + GPU sharded teacher)
    # ========================================================================
    if master_process:
        print("STAGE 4: Create Distill_Model wrapper...")
        log_memory("[S4-BEFORE] Distill_Model wrap")
    
    # IMPORTANT: do not unwrap FSDP/DDP via `.module` for forward; keep the wrapper
    # to avoid sharded/flattened parameter views breaking e.g. embeddings.
    student_for_distill = model
    model = Distill_Model(student_for_distill, teacher_model, output_hidden_state=False)

    # Distill + non-FSDP + DDP: wrap Distill_Model with DDP so student gradients are synchronized.
    if ddp and (not using_fsdp):
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        if master_process:
            log_memory("[S4-AFTER-DDP] Distill_Model DDP wrapped")
            print("   ✓ DDP wrapped (distill + non-FSDP): gradients synchronized")
            print("-" * 80)
    
    if master_process:
        log_memory("[S4-AFTER-WRAP] Distill_Model wrapped")
        if using_fsdp:
            print("   ✓ Distill_Model created (student FSDP on GPU, teacher FSDP on GPU)")
        else:
            print("   ✓ Distill_Model created (student on GPU, teacher on GPU)")
        print("-" * 80)
    
    # ========================================================================
    # STAGE 5: SKIP additional DDP/FSDP wrap for Distill_Model
    #
    # Do NOT wrap Distill_Model itself with DDP/FSDP. The student and teacher
    # are already wrapped (FSDP from Stage 2 and Stage 3). Distill_Model is
    # just a container for forward coordination.
    # ========================================================================
    if master_process:
        print("STAGE 5: Skip additional wrap for Distill_Model (student & teacher already sharded)...")
        if using_fsdp:
            print("   ✓ Distill_Model ready (student FSDP from Stage 2, teacher FSDP from Stage 3)")
        else:
            print("   ✓ Distill_Model ready (student & teacher on GPU)")
        print("-" * 80)
else:
    # Non-distill mode: will wrap with DDP later at line 484
    using_fsdp_final = using_fsdp

# ============================================================================
# NOTE: Large state tensors are now FROZEN PARAMETERS (not buffers)
# ============================================================================
# REASON: FSDP shards parameters but NOT buffers. By registering as
# nn.Parameter(requires_grad=False), they get sharded instead of replicated.
# 
# Changes in sparse_modeling.py:
#  - mask: nn.Parameter(..., requires_grad=False, dtype=fp16)
#  - grad_ema: frozen param if movement metric, else placeholder param
#  - hessian_diag: frozen param if hessian metric, else placeholder param  
#  - frozen_mask_flags: frozen param (same dtype as weight) if freeze region exists
#  - init_mask: DELETED to save 14GB/rank
#
# Peak memory: 95GB → 15-20GB per GPU during FSDP wrap ✅


# DDP wrap (non-distill mode; distill+non-FSDP is wrapped above)
if ddp and (not using_fsdp) and (not args.distill_model):
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    if master_process:
        log_memory("[AFTER DDP wrapping]")
        print("   ✓ DDP wrapped (non-distill mode)")

# NOTE: In distill+non-FSDP mode we DO wrap with DDP (above), otherwise gradients diverge.

# NOTE: Model wrapping depends on mode:
# - FSDP mode: model = FSDP(student) or Distill_Model(FSDP(student), teacher)
# - DDP non-distill mode: model = DDP(student)
# - DDP distill mode: model = Distill_Model(student, teacher) [no DDP wrap to avoid mixed devices]
# - Non-DDP: model is raw student or Distill_Model
raw_model = model.module if hasattr(model, "module") else model

# In distill mode, the student is the trainable module (often FSDP-wrapped).
train_student = raw_model.student if args.distill_model else raw_model

def _configure_optimizers(m):
    # FSDP wrapper does not always expose the underlying module's methods.
    al1 = float(getattr(args, 'adaptive_l1_decay', 0.0))
    if hasattr(m, "configure_optimizers"):
        return m.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, adaptive_l1_decay=al1)
    inner = getattr(m, "_fsdp_wrapped_module", None)
    if inner is not None and hasattr(inner, "configure_optimizers"):
        return inner.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, adaptive_l1_decay=al1)
    raise AttributeError("No configure_optimizers found on model/FSDP-wrapped module")

if master_process:
    log_memory("[AFTER raw_model extraction]")

optimizer = _configure_optimizers(train_student)

def _align_fsdp_optimizer_state(saved_optim_sd, current_optimizer, verbose=False):
    """对齐 FSDP resume 时 optimizer state 的 flat parameter padding 差异。
    
    FSDP wrap 时会将参数 flatten + concat + pad（使其能被 world_size 整除）。
    不同运行之间（如 buffer 增减、参数微调等），flat parameter 的 padding
    可能略有差异，导致 optimizer state (exp_avg/exp_avg_sq) 与当前参数大小不匹配。
    
    本函数遍历 saved state，对每个 tensor 进行 pad/truncate 以匹配当前参数大小。
    """
    current_sd = current_optimizer.state_dict()
    current_params = {i: p for i, p in enumerate(current_optimizer.param_groups[0]['params'])}
    
    # 获取当前 optimizer 中每个 param 的实际 numel
    param_sizes = {}
    for group_idx, group in enumerate(current_optimizer.param_groups):
        for p in group['params']:
            param_id = id(p)
            param_sizes[param_id] = p.numel()
    
    # 构建 param index -> current numel 映射
    # optimizer state_dict 中 state 的 key 是 param 在所有 groups 中的全局索引
    idx = 0
    idx_to_numel = {}
    for group in current_optimizer.param_groups:
        for p in group['params']:
            idx_to_numel[idx] = p.numel()
            idx += 1
    
    n_aligned = 0
    n_skipped = 0
    
    if 'state' not in saved_optim_sd:
        return saved_optim_sd
    
    for param_idx, state_dict_val in saved_optim_sd['state'].items():
        pidx = int(param_idx)
        if pidx not in idx_to_numel:
            continue
        target_numel = idx_to_numel[pidx]
        
        for key in ['exp_avg', 'exp_avg_sq']:
            if key not in state_dict_val:
                continue
            t = state_dict_val[key]
            if not isinstance(t, torch.Tensor):
                continue
            if t.numel() == target_numel:
                n_skipped += 1
                continue
            
            # 大小不匹配 -> pad 或 truncate
            old_numel = t.numel()
            if verbose:
                print(f"[RESUME] Aligning optimizer state param={pidx} {key}: "
                      f"{old_numel} -> {target_numel} (diff={target_numel - old_numel})")
            
            t_flat = t.flatten()
            if old_numel < target_numel:
                # pad with zeros
                pad = torch.zeros(target_numel - old_numel, dtype=t.dtype, device=t.device)
                t_flat = torch.cat([t_flat, pad])
            else:
                # truncate
                t_flat = t_flat[:target_numel]
            
            state_dict_val[key] = t_flat
            n_aligned += 1
    
    if verbose or n_aligned > 0:
        print(f"[RESUME] FSDP optimizer state alignment: {n_aligned} tensors aligned, "
              f"{n_skipped} already matched")
    
    return saved_optim_sd

# GradScaler API differs across PyTorch versions:
# - Newer: torch.amp.GradScaler(device_type='cuda', ...)
# - Older: torch.cuda.amp.GradScaler(...)
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler('cuda', enabled=((dtype == 'float16') and (not args.use_deepspeed)))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=((dtype == 'float16') and (not args.use_deepspeed)))

# lr schedule
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    decay_ratio = min(1.0, max(0.0, float(decay_ratio)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def get_decay(it):
    inc = int(args.increase_step)
    dmax = float(args.srste_decay)
    if dmax <= 0.0:
        return 0.0
    if it < inc:
        return dmax / max(1, inc) * it
    return dmax

# eval helper
@torch.no_grad()
def estimate_loss():
    out = {}
    # IMPORTANT: under FSDP, calling the unwrapped module (`.module`) will see sharded/flattened
    # parameter views (e.g. embedding weights become 1-D), which breaks forward. Always run the
    # wrapped `model` for eval when using FSDP/DDP.
    # For distill training, evaluate student-only (no teacher forward needed).
    if args.distill_model:
        container = model.module if hasattr(model, "module") else model
        eval_model = container.student
        
        # DEBUG: Print model structure info (only once on master)
        if master_process and not hasattr(estimate_loss, '_debug_printed'):
            estimate_loss._debug_printed = True
            print(f"[DEBUG estimate_loss] container type: {type(container)}")
            print(f"[DEBUG estimate_loss] eval_model type: {type(eval_model)}")
            # Check if eval_model is FSDP wrapped
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP_Check
            print(f"[DEBUG estimate_loss] eval_model is FSDP: {isinstance(eval_model, FSDP_Check)}")
            # Try to get lm_head info
            try:
                if hasattr(eval_model, 'module'):
                    inner = eval_model.module
                else:
                    inner = eval_model
                if hasattr(inner, 'model') and hasattr(inner.model, 'lm_head'):
                    lm_head = inner.model.lm_head
                    print(f"[DEBUG estimate_loss] lm_head.weight.shape: {lm_head.weight.shape}")
                    print(f"[DEBUG estimate_loss] lm_head type: {type(lm_head)}")
            except Exception as e:
                print(f"[DEBUG estimate_loss] Could not get lm_head info: {e}")
    else:
        eval_model = model
    eval_model.eval()
    for split in ['train','val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            # DEBUG: Print first batch shape
            if k == 0 and master_process and not hasattr(estimate_loss, '_batch_debug_printed'):
                estimate_loss._batch_debug_printed = True
                print(f"[DEBUG estimate_loss] X.shape: {X.shape}, Y.shape: {Y.shape}")
                print(f"[DEBUG estimate_loss] block_size: {block_size}")
            with ctx:
                logits, loss, _ = eval_model(X, Y)
            losses[k] = loss.item()
            # Explicitly delete large tensors to free memory during eval loop
            del logits, loss, X, Y
        out[split] = losses.mean().item()
        # Clear cache between train/val splits
        torch.cuda.empty_cache()
    eval_model.train()
    # CRITICAL: Synchronize CUDA before returning to ensure all GPU operations are complete.
    # This is especially important for HYBRID_SHARD mode where mesh_shard process group
    # requires all ranks to complete their operations before starting new collectives.
    torch.cuda.synchronize()
    return out

# Training loop: FULLY ALIGNED with main.py
iter_num = 0
best_val_loss = None
best_wiki_ppl = 1e9
best_lm_eval_mean = 0.0  # 跟踪 finalization 阶段 lm_eval 的最佳 mean accuracy
eval_count = 0
always_save_checkpoint = args.always_save_checkpoint  # 默认 True，每次都保存
save_interval = int(args.save_interval) if args.save_interval is not None else int(args.eval_interval)

# ===== RESUME: Restore optimizer/scaler state (model weights already loaded in STAGE 1.5) =====
if args.resume and _resume_ckpt_loaded:
    if master_process:
        print(f"[RESUME] Model weights were loaded in STAGE 1.5 (before FSDP wrap)")
    
    # 恢复 iter_num
    iter_num = _resume_iter_num
    best_wiki_ppl = _resume_best_wiki_ppl
    if master_process:
        print(f"[RESUME] Resuming from iter_num={iter_num}, best_wiki_ppl={best_wiki_ppl}")
    
    # 恢复 optimizer 状态
    if _resume_optimizer_state is not None:
        try:
            # FSDP 下 flat parameter 的 padding 可能在不同运行间变化，
            # 先对齐 optimizer state 中各 tensor 的大小
            if using_fsdp:
                _resume_optimizer_state = _align_fsdp_optimizer_state(
                    _resume_optimizer_state, optimizer, verbose=master_process
                )
            optimizer.load_state_dict(_resume_optimizer_state)
            if master_process:
                print(f"[RESUME] Optimizer state restored")
        except Exception as e:
            if master_process:
                print(f"[RESUME] Warning: optimizer restore failed: {e}")
                print(f"[RESUME] Training will continue without optimizer state (momentum will rebuild)")
        # 清理内存
        del _resume_optimizer_state
    
    # 恢复 scaler 状态
    if _resume_scaler_state is not None and scaler is not None:
        try:
            scaler.load_state_dict(_resume_scaler_state)
            if master_process:
                print(f"[RESUME] Scaler state restored")
        except Exception as e:
            if master_process:
                print(f"[RESUME] Warning: scaler restore failed: {e}")
        # 清理内存
        del _resume_scaler_state
    
    # 清理内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if master_process:
        print(f"[RESUME] ✓ Resume complete, starting from iter {iter_num}")
    
    # 同步所有 rank 的 iter_num
    if ddp:
        iter_num_tensor = torch.tensor([iter_num], dtype=torch.long, device=device)
        torch.distributed.broadcast(iter_num_tensor, src=0)
        iter_num = int(iter_num_tensor.item())
        if not master_process:
            print(f"[RESUME] Rank {ddp_rank}: synced iter_num={iter_num}")

elif args.resume and not _resume_ckpt_loaded:
    # Fallback: 如果 STAGE 1.5 没有成功加载，尝试旧的方式（可能 OOM）
    if master_process:
        print(f"[RESUME] Warning: STAGE 1.5 did not load checkpoint, attempting post-FSDP load (may OOM)...")
    
    resume_ckpt_dir = args.resume_dir
    if resume_ckpt_dir is None:
        # 尝试从 out_dir/last 符号链接恢复
        last_link = os.path.join(out_dir, "last")
        last_dir_file = os.path.join(out_dir, "last_dir.txt")
        if os.path.islink(last_link) or os.path.isdir(last_link):
            resume_ckpt_dir = os.path.realpath(last_link)
        elif os.path.exists(last_dir_file):
            with open(last_dir_file, "r") as f:
                resume_ckpt_dir = f.read().strip()
    
    if resume_ckpt_dir and os.path.isdir(resume_ckpt_dir):
        model_pt_path = os.path.join(resume_ckpt_dir, "model.pt")
        if os.path.exists(model_pt_path):
            if master_process:
                print(f"[RESUME-FALLBACK] Loading checkpoint from {model_pt_path}")
            
            # 加载 checkpoint
            resume_ckpt = torch.load(model_pt_path, map_location="cpu", weights_only=False)
            
            # 恢复模型权重（需要 FSDP summon_full_params）
            if "model_state_dict" in resume_ckpt:
                ckpt_sd = resume_ckpt["model_state_dict"]
                
                # 确定加载目标
                _raw = model.module if hasattr(model, 'module') else model
                if args.distill_model and hasattr(_raw, 'student'):
                    load_target = _raw.student
                    load_target_name = "model.student"
                else:
                    load_target = model
                    load_target_name = "model"
                
                if master_process:
                    print(f"[RESUME-FALLBACK] Loading checkpoint to {load_target_name}")
                
                # 尝试使用 FSDP.summon_full_params 加载
                try:
                    if using_fsdp and FSDP is not None:
                        with FSDP.summon_full_params(load_target, writeback=True, recurse=True):
                            missing, unexpected = load_target.load_state_dict(ckpt_sd, strict=False)
                    else:
                        missing, unexpected = load_target.load_state_dict(ckpt_sd, strict=False)
                    
                    if master_process:
                        if missing:
                            print(f"[RESUME-FALLBACK] Missing keys ({len(missing)})")
                        if unexpected:
                            print(f"[RESUME-FALLBACK] Unexpected keys ({len(unexpected)})")
                        print(f"[RESUME-FALLBACK] Model weights loaded to {load_target_name}")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-FALLBACK] Warning: load_state_dict failed: {e}")
            
            # 恢复 optimizer 状态
            if args.resume_optimizer and "optimizer_state_dict" in resume_ckpt:
                try:
                    _fb_optim_sd = resume_ckpt["optimizer_state_dict"]
                    if using_fsdp:
                        _fb_optim_sd = _align_fsdp_optimizer_state(
                            _fb_optim_sd, optimizer, verbose=master_process
                        )
                    optimizer.load_state_dict(_fb_optim_sd)
                    del _fb_optim_sd
                    if master_process:
                        print(f"[RESUME-FALLBACK] Optimizer state restored")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-FALLBACK] Warning: optimizer restore failed: {e}")
                        print(f"[RESUME-FALLBACK] Training will continue without optimizer state")
            
            # 恢复 scaler 状态
            if "scaler_state_dict" in resume_ckpt and scaler is not None:
                try:
                    scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
                    if master_process:
                        print(f"[RESUME-FALLBACK] Scaler state restored")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-FALLBACK] Warning: scaler restore failed: {e}")
            
            # 恢复 iter_num
            if "iter_num" in resume_ckpt:
                iter_num = int(resume_ckpt["iter_num"])
                if master_process:
                    print(f"[RESUME-FALLBACK] Resuming from iter_num={iter_num}")
            
            # 尝试恢复 best_wiki_ppl
            eval_json_path = os.path.join(resume_ckpt_dir, "eval.json")
            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, "r") as f:
                        eval_data = json.load(f)
                    if "wiki_ppl" in eval_data and eval_data["wiki_ppl"] is not None:
                        best_wiki_ppl = float(eval_data["wiki_ppl"])
                        if master_process:
                            print(f"[RESUME-FALLBACK] Restored best_wiki_ppl={best_wiki_ppl}")
                except Exception as e:
                    if master_process:
                        print(f"[RESUME-FALLBACK] Warning: failed to load eval.json: {e}")
            
            # 清理内存
            del resume_ckpt
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            if master_process:
                print(f"[RESUME-FALLBACK] Checkpoint loaded successfully, starting from iter {iter_num}")
        else:
            if master_process:
                print(f"[RESUME-FALLBACK] Warning: model.pt not found in {resume_ckpt_dir}")
    else:
        if master_process:
            print(f"[RESUME-FALLBACK] Warning: resume_dir not found or invalid: {resume_ckpt_dir}")
    
    # 同步所有 rank 的 iter_num
    if ddp:
        iter_num_tensor = torch.tensor([iter_num], dtype=torch.long, device=device)
        torch.distributed.broadcast(iter_num_tensor, src=0)
        iter_num = int(iter_num_tensor.item())
        if not master_process:
            print(f"[RESUME-FALLBACK] Rank {ddp_rank}: synced iter_num={iter_num}")

# Match main.py semantics for gradient accumulation:
# 1) global_batch_size must be divisible by micro_batch_size
# 2) under DDP/FSDP, we split accumulation steps across ranks
assert int(args.global_batch_size) % int(batch_size) == 0, (
    f"global_batch_size ({args.global_batch_size}) must be divisible by batch_size ({batch_size})"
)
gradient_accumulation_steps = int(args.global_batch_size) // int(batch_size)
if ddp:
    assert gradient_accumulation_steps % int(ddp_world_size) == 0, (
        f"gradient_accumulation_steps ({gradient_accumulation_steps}) must be divisible by world_size ({ddp_world_size})"
    )
    gradient_accumulation_steps //= int(ddp_world_size)

finalization_done = False
finalization_iter = 0
local_iter_num = 0
training_start = time.time()

# wandb setup
wandb = None
if args.wandb_logging:
    try:
        import wandb as wandb_lib
        if master_process:
            wandb = wandb_lib.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    except Exception as e:
        if master_process:
            print(f"[wandb] init failed: {e}")

wandb_log = args.wandb_logging

# Create a single target directory for this run (all eval saves will overwrite files here).
# We include a timestamp once at startup so concurrent runs don't collide.
safe_student = str(args.student_model).replace('/', '_')
safe_mask = str(args.mask_type).replace('/', '_')
safe_sparsity = str(args.sparsity_ratio)
safe_metric = str(args.mask_metric).replace('/', '_')
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
subdir_name = f"{safe_student}_mask-{safe_mask}_s{safe_sparsity}_m-{safe_metric}_{run_timestamp}"
target_dir = os.path.join(out_dir, subdir_name)
if master_process:
    os.makedirs(target_dir, exist_ok=True)

hardness_task = args.hardness_task
hardness_kldiv = args.hardness_kldiv
hardness_squarehead = args.hardness_squarehead

# ========== Channel Pruning 初始化 ==========
channel_pruning_enabled = bool(args.enable_channel_pruning)
channel_mask_state = None
channel_score_computer = None
channel_original_forwards = None
channel_regression_recovery = None

if channel_pruning_enabled:
    if master_process:
        print("=" * 60)
        print("[Channel Pruning] 初始化结构化通道剪枝...")
        print("=" * 60)
    
    # 自动计算 hardening_start_step
    _ch_hardening_start = args.channel_hardening_start_step
    if _ch_hardening_start == 0:
        _ch_hardening_start = int(0.6 * max_iters)
        if master_process:
            print(f"[Channel Pruning] hardening_start_step 自动设为 {_ch_hardening_start} (60% of max_iters)")
    
    # 构建 ChannelPruningConfig
    channel_pruning_config = ChannelPruningConfig(
        model_name=args.student_model,
        model_type=detected_model_type,
        ffn_keep_ratio=args.ffn_keep_ratio,
        importance_metric=args.channel_importance_metric,
        score_alpha=args.channel_score_alpha,
        score_beta=args.channel_score_beta,
        score_gamma=args.channel_score_gamma,
        normalize_scores=args.channel_normalize_scores,
        score_ema_beta=args.channel_score_ema_beta,
        mask_update_period=args.channel_mask_update_period,
        mask_lr=args.channel_mask_lr,
        temp_init=args.channel_temp_init,
        temp_min=args.channel_temp_min,
        temp_decay=args.channel_temp_decay,
        sparsity_warmup_steps=args.channel_sparsity_warmup_steps,
        hardening_start_step=_ch_hardening_start,
        hardening_duration=args.channel_hardening_duration,
        # SlimLLM PCA-aware scoring
        enable_pca_scoring=args.enable_pca_scoring,
        pca_lambda=args.pca_lambda,
        pca_top_k=args.pca_top_k,
        pca_normalize_mode=args.pca_normalize_mode,
        pca_ema_beta=args.pca_ema_beta,
        pca_use_activation_norm=args.pca_use_activation_norm,
        pca_update_period=args.pca_update_period,
        pca_regression_on_hardening=args.pca_regression_on_hardening,
        # LoRA Bypass 补偿
        enable_lora_bypass=args.enable_lora_bypass,
        lora_bypass_rank=args.lora_bypass_rank,
        lora_bypass_alpha=args.lora_bypass_alpha,
        lora_bypass_dropout=args.lora_bypass_dropout,
    )
    
    # 初始化 mask state 和 score computer
    channel_mask_state = ChannelMaskState(
        train_student,
        channel_pruning_config,
        device=device
    )
    
    channel_score_computer = ChannelScoreComputer(
        train_student,
        channel_pruning_config,
        device=device
    )
    
    # 初始化线性回归恢复（在 mask hardening 时执行）
    # 需要在 patch_model_mlp_forward 之前创建，以便传入 forward hook
    if args.pca_regression_on_hardening:
        channel_regression_recovery = ChannelRegressionRecovery(
            train_student,
            channel_mask_state,
            channel_pruning_config,
            device=device,
        )
        if master_process:
            print(f"[Channel Pruning] Regression recovery 已启用: 将在 hardening 跨越 0.5 阈值时执行")
    
    # Patch MLP forward 以在前向传播中应用 channel mask（+ 可选 LoRA bypass）
    channel_original_forwards, lora_bypass_modules = patch_model_mlp_forward(
        train_student,
        channel_mask_state,
        channel_pruning_config,
        regression_recovery=channel_regression_recovery,
    )
    
    # 如果启用了 LoRA bypass，将其参数追加到 optimizer
    # （optimizer 在 channel pruning 初始化之前创建，所以需要手动追加）
    if lora_bypass_modules:
        lora_params_decay = []
        lora_params_nodecay = []
        total_lora_params = 0
        for lora_mod in lora_bypass_modules:
            for p in lora_mod.parameters():
                if p.requires_grad:
                    total_lora_params += p.numel()
                    if p.dim() >= 2:
                        lora_params_decay.append(p)
                    else:
                        lora_params_nodecay.append(p)
        if lora_params_decay:
            optimizer.add_param_group({"params": lora_params_decay, "weight_decay": weight_decay})
        if lora_params_nodecay:
            optimizer.add_param_group({"params": lora_params_nodecay, "weight_decay": 0.0})
        if master_process:
            print(f"[Channel Pruning] LoRA bypass 已启用:")
            print(f"  rank={args.lora_bypass_rank}, alpha={args.lora_bypass_alpha}, dropout={args.lora_bypass_dropout}")
            print(f"  num_layers={len(lora_bypass_modules)}, total_lora_params={total_lora_params:,}")
            model_params = sum(p.numel() for p in train_student.parameters())
            print(f"  LoRA overhead: {total_lora_params/model_params*100:.2f}% of model params")
    
    # 如果启用 PCA scoring，安装数据收集 hooks
    if args.enable_pca_scoring and channel_score_computer.pca_manager is not None:
        channel_score_computer.install_pca_hooks(train_student)
        if master_process:
            print(f"[Channel Pruning] PCA scoring 已启用:")
            print(f"  pca_lambda={args.pca_lambda}, pca_normalize_mode={args.pca_normalize_mode}")
            print(f"  pca_top_k={args.pca_top_k}, pca_update_period={args.pca_update_period}")
            print(f"  pca_use_activation_norm={args.pca_use_activation_norm}")
    
    if master_process:
        intermediate_size = channel_mask_state.intermediate_size
        keep_k = channel_mask_state.get_keep_k(0)
        print(f"[Channel Pruning] intermediate_size: {intermediate_size}")
        print(f"[Channel Pruning] num_layers: {channel_mask_state.num_layers}")
        print(f"[Channel Pruning] target keep channels: {keep_k} ({args.ffn_keep_ratio*100:.1f}%)")
        print(f"[Channel Pruning] importance_metric: {args.channel_importance_metric}")
        print(f"[Channel Pruning] mask_update_period: {args.channel_mask_update_period}")
        print(f"[Channel Pruning] hardening: start={_ch_hardening_start}, duration={args.channel_hardening_duration}")
        print("=" * 60)

# CRITICAL: Final barrier before training loop to ensure all ranks are synchronized
# after all FSDP wrapping, optimizer creation, and initialization steps.
if ddp:
    # First, verify ALL ranks are present by doing an all_reduce count
    rank_count = torch.tensor([1], dtype=torch.int64, device=device)
    dist.all_reduce(rank_count, op=dist.ReduceOp.SUM)
    actual_world_size = int(rank_count.item())
    if master_process:
        print(f"[SYNC CHECK] Expected world_size={ddp_world_size}, actual participants={actual_world_size}")
    if actual_world_size != ddp_world_size:
        raise RuntimeError(f"World size mismatch! Expected {ddp_world_size} but only {actual_world_size} ranks responded.")
    
    _safe_barrier()
    
    if master_process:
        print("=" * 80)
        print("All ranks synchronized. Starting training loop...")
        print("=" * 80)

# ========== 初始化异步数据预取器 ==========
if PREFETCH_ENABLED:
    train_prefetcher = AsyncDataPrefetcher(
        data=train_data,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        prefetch_count=3  # 预取 3 个 batch
    )
    train_prefetcher.start()
    if master_process:
        print("[Prefetcher] Async data prefetching enabled (prefetch_count=3)")
else:
    if master_process:
        print("[Prefetcher] Disabled (set DISABLE_PREFETCH=0 to enable)")

X, Y = get_batch('train')
t0 = time.time()

# ========== PyTorch Profiler 初始化 ==========
# 用于分析训练是 compute-bound 还是 communication-bound
# 注意: 在 FSDP 模式下，PyTorch Profiler 可能导致 rank 间不同步，建议禁用
profiler_ctx = None
profiler_trace_dir = None  # Chrome trace 文件导出目录
profiler_stats = {
    'cuda_time_total': 0.0,
    'cpu_time_total': 0.0,
    'comm_time_total': 0.0,  # NCCL 通信时间
    'compute_time_total': 0.0,  # 计算时间 (CUDA kernels 除去通信)
    'forward_time': 0.0,
    'backward_time': 0.0,
    'optimizer_time': 0.0,
    'num_profiles': 0,
}
profiler_last_step_stats = {}  # 存储最近一次 step 的统计

# 轻量级时间统计 (FSDP-safe，所有 rank 都可以使用)
step_timing = {
    'forward_ms': 0.0,
    'backward_ms': 0.0,
    'optimizer_ms': 0.0,
    'total_ms': 0.0,
    'num_steps': 0,
}

def profiler_trace_handler(prof):
    """
    每当 profiler 完成一个活跃周期时调用。
    解析 profiler 统计数据，区分计算时间和通信时间。
    通信操作通常包含: nccl, all_reduce, all_gather, broadcast, reduce_scatter
    
    同时导出 Chrome trace 文件到 profiler_trace_dir
    """
    global profiler_stats, profiler_last_step_stats, profiler_trace_dir
    
    # 获取按 CUDA 时间排序的事件
    events = prof.key_averages()
    
    cuda_total = 0.0
    cpu_total = 0.0
    comm_total = 0.0
    forward_total = 0.0
    backward_total = 0.0
    
    comm_keywords = ['nccl', 'all_reduce', 'all_gather', 'broadcast', 'reduce_scatter', 
                     'AllReduce', 'AllGather', 'Broadcast', 'ReduceScatter', 'c10d', 'ncclKernel']
    
    for evt in events:
        cuda_time = evt.cuda_time_total / 1000.0  # 转换为 ms
        cpu_time = evt.cpu_time_total / 1000.0
        cuda_total += cuda_time
        cpu_total += cpu_time
        
        name_lower = evt.key.lower()
        # 检查是否为通信操作
        is_comm = any(kw.lower() in name_lower for kw in comm_keywords)
        if is_comm:
            comm_total += cuda_time
        
        # 检查 forward/backward (通过我们的 record_function 标记)
        if 'forward_pass' in name_lower:
            forward_total += cuda_time
        if 'backward_pass' in name_lower:
            backward_total += cuda_time
    
    compute_total = max(0.0, cuda_total - comm_total)
    
    # 累积统计
    profiler_stats['cuda_time_total'] += cuda_total
    profiler_stats['cpu_time_total'] += cpu_total
    profiler_stats['comm_time_total'] += comm_total
    profiler_stats['compute_time_total'] += compute_total
    profiler_stats['forward_time'] += forward_total
    profiler_stats['backward_time'] += backward_total
    profiler_stats['num_profiles'] += 1
    
    # 保存最近一次的统计
    profiler_last_step_stats = {
        'cuda_time_ms': cuda_total,
        'cpu_time_ms': cpu_total,
        'comm_time_ms': comm_total,
        'compute_time_ms': compute_total,
        'forward_time_ms': forward_total,
        'backward_time_ms': backward_total,
        'comm_ratio': comm_total / max(cuda_total, 1e-6),
        'compute_ratio': compute_total / max(cuda_total, 1e-6),
    }
    
    # 导出 Chrome trace 文件
    if profiler_trace_dir:
        try:
            trace_file = os.path.join(profiler_trace_dir, f"trace_step_{prof.step_num}.json")
            prof.export_chrome_trace(trace_file)
            print(f"[Profiler] Chrome trace exported to: {trace_file}")
        except Exception as e:
            print(f"[Profiler] WARNING: Failed to export Chrome trace: {e}")
    
    # 打印详细的 profiler 表格（每个活跃周期结束时）
    print(f"\n[Profiler] Step {prof.step_num} trace ready:")
    print(f"  CUDA total: {cuda_total:.2f}ms, Compute: {compute_total:.2f}ms, Comm: {comm_total:.2f}ms")
    print(f"  Compute ratio: {compute_total/max(cuda_total,1e-6)*100:.1f}%, Comm ratio: {comm_total/max(cuda_total,1e-6)*100:.1f}%")
    
    # 打印 top 10 最耗时的操作
    print(f"  Top 10 CUDA operations:")
    sorted_events = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:10]
    for i, evt in enumerate(sorted_events):
        cuda_ms = evt.cuda_time_total / 1000.0
        is_comm = any(kw.lower() in evt.key.lower() for kw in comm_keywords)
        marker = "[COMM]" if is_comm else ""
        print(f"    {i+1}. {evt.key[:50]:50s} {cuda_ms:8.2f}ms {marker}")

def get_profiler_summary():
    """获取 profiler 统计摘要"""
    n = max(profiler_stats['num_profiles'], 1)
    cuda_avg = profiler_stats['cuda_time_total'] / n
    comm_avg = profiler_stats['comm_time_total'] / n
    compute_avg = profiler_stats['compute_time_total'] / n
    
    comm_ratio = comm_avg / max(cuda_avg, 1e-6)
    compute_ratio = compute_avg / max(cuda_avg, 1e-6)
    
    # 判断是 compute-bound 还是 communication-bound
    if comm_ratio > 0.4:
        bottleneck = "COMMUNICATION-BOUND"
    elif compute_ratio > 0.6:
        bottleneck = "COMPUTE-BOUND"
    else:
        bottleneck = "BALANCED"
    
    return {
        'avg_cuda_time_ms': cuda_avg,
        'avg_comm_time_ms': comm_avg,
        'avg_compute_time_ms': compute_avg,
        'comm_ratio': comm_ratio,
        'compute_ratio': compute_ratio,
        'bottleneck': bottleneck,
        'num_profiles': n,
    }

# 检查是否在 FSDP 模式下启用 profiler
profiler_enabled = args.enable_profiler
# FORCE ENABLE: 强制启用 profiler 以生成 Chrome trace（用户请求）
# 注意：在 FSDP 模式下可能会有 rank 不同步的风险，但为了获取 profiling 数据暂时启用
if profiler_enabled and using_fsdp:
    if master_process:
        print(f"[Profiler] WARNING: PyTorch Profiler enabled in FSDP mode (forced by user).")
        print(f"[Profiler] This may cause rank desync. Monitor for collective mismatches.")
# if profiler_enabled and using_fsdp:
#     if master_process:
#         print(f"[Profiler] WARNING: PyTorch Profiler may cause rank desync in FSDP mode.")
#         print(f"[Profiler] Using lightweight timing instead of full profiler.")
#         print(f"[Profiler] Set --enable_profiler False to disable all profiling.")
#     # 在 FSDP 模式下使用轻量级时间统计替代
#     profiler_enabled = False

if profiler_enabled and master_process:
    print(f"[Profiler] Enabled. Will start profiling at step {args.profiler_start_step}")
    print(f"[Profiler] Warmup={args.profiler_warmup_steps}, Active={args.profiler_active_steps}, Repeat={args.profiler_repeat}")

# 轻量级时间统计始终启用 (FSDP-safe)
if args.enable_profiler and master_process:
    print(f"[Timing] Lightweight timing enabled for all steps (FSDP-safe).")
    print(f"[Timing] Will log timing stats every {args.profiler_log_interval} steps.")

if master_process:
    pbar = tqdm(total=max_iters, initial=iter_num, desc="Training", dynamic_ncols=True)

while True:
    # ========== PyTorch Profiler 控制 ==========
    # 在指定的步数范围内启用 profiler (仅非 FSDP 模式)
    profiler_active_this_step = False
    if profiler_enabled:
        profiler_end_step = args.profiler_start_step + (args.profiler_warmup_steps + args.profiler_active_steps) * args.profiler_repeat
        if args.profiler_start_step <= iter_num < profiler_end_step:
            profiler_active_this_step = True
            if iter_num == args.profiler_start_step and master_process:
                # 创建 Chrome trace 导出目录
                profiler_trace_dir = os.path.join(args.out_dir, "profiler_traces")
                os.makedirs(profiler_trace_dir, exist_ok=True)
                print(f"[Profiler] Chrome traces will be saved to: {profiler_trace_dir}")
                
                # 创建 profiler，使用自定义的 trace_handler 来收集统计
                profiler_schedule = schedule(
                    wait=0,
                    warmup=args.profiler_warmup_steps,
                    active=args.profiler_active_steps,
                    repeat=args.profiler_repeat
                )
                profiler_ctx = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=profiler_schedule,
                    on_trace_ready=profiler_trace_handler,  # 使用自定义 handler 收集统计
                    record_shapes=True,
                    with_stack=False,
                    profile_memory=False,
                )
                profiler_ctx.__enter__()
                print(f"[Profiler] Started profiling at step {iter_num}")
                print(f"[Profiler] Will run for {args.profiler_warmup_steps} warmup + {args.profiler_active_steps} active steps, repeat {args.profiler_repeat}x")
        
        # FSDP + Profiler 退出同步: 必须在 profiler exit 之前让所有 rank 同步等待
        # 关键：这个 barrier 必须在 master_process 条件块之外，让所有 rank 都参与
        if iter_num == profiler_end_step and ddp and using_fsdp:
            try:
                # 先同步，确保所有 rank 都到达这个点
                dist.barrier()
                if master_process:
                    print(f"[Profiler] All ranks synchronized before profiler exit")
            except Exception as e:
                if master_process:
                    print(f"[Profiler] Pre-exit barrier failed: {e}")
        
        # Profiler 结束时打印摘要 (只在 rank 0 执行)
        if iter_num == profiler_end_step and master_process and profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)
            summary = get_profiler_summary()
            print(f"\n{'='*70}")
            print(f"[Profiler] PERFORMANCE ANALYSIS SUMMARY")
            print(f"{'='*70}")
            print(f"  Total profiled cycles: {summary['num_profiles']}")
            print(f"  Avg CUDA time per step: {summary['avg_cuda_time_ms']:.2f} ms")
            print(f"  Avg Compute time: {summary['avg_compute_time_ms']:.2f} ms ({summary['compute_ratio']*100:.1f}%)")
            print(f"  Avg Communication time: {summary['avg_comm_time_ms']:.2f} ms ({summary['comm_ratio']*100:.1f}%)")
            print(f"")
            print(f"  >>> BOTTLENECK ANALYSIS: {summary['bottleneck']} <<<")
            if summary['bottleneck'] == "COMMUNICATION-BOUND":
                print(f"  Suggestion: Consider reducing world_size, increasing batch_size, or using gradient accumulation")
            elif summary['bottleneck'] == "COMPUTE-BOUND":
                print(f"  Suggestion: Training is efficiently using compute. Consider larger model or more GPUs.")
            print(f"{'='*70}\n")
            
            if wandb_log and wandb:
                wandb.log({
                    'profiler/avg_cuda_time_ms': summary['avg_cuda_time_ms'],
                    'profiler/avg_compute_time_ms': summary['avg_compute_time_ms'],
                    'profiler/avg_comm_time_ms': summary['avg_comm_time_ms'],
                    'profiler/compute_ratio': summary['compute_ratio'],
                    'profiler/comm_ratio': summary['comm_ratio'],
                }, step=iter_num)
            profiler_ctx = None
        
        # FSDP + Profiler 退出同步: 等待 rank 0 完成 profiler 清理后再继续
        if iter_num == profiler_end_step and ddp and using_fsdp:
            try:
                dist.barrier()
                if master_process:
                    print(f"[Profiler] All ranks synchronized after profiler exit")
            except Exception as e:
                if master_process:
                    print(f"[Profiler] Barrier after exit failed: {e}")
    
    # ========== 轻量级时间统计 (FSDP-safe) ==========
    # 所有 rank 都可以安全使用，只在 rank 0 打印
    step_start_time = time.time()
    forward_start_time = None
    backward_start_time = None
    optimizer_start_time = None

    # ========== PERIODIC SYNC CHECK (every 100 steps) ==========
    # This helps detect rank divergence early before it causes NCCL hangs.
    if ddp and (iter_num % 100 == 0) and (iter_num > 0):
        try:
            sync_check = torch.tensor([iter_num], dtype=torch.int64, device=device)
            dist.all_reduce(sync_check, op=dist.ReduceOp.MAX)
            max_iter = int(sync_check.item())
            if max_iter != iter_num:
                print(f"[RANK {ddp_rank}] SYNC WARNING: local iter={iter_num}, max across ranks={max_iter}")
        except Exception as e:
            if master_process:
                print(f"[SYNC CHECK] Exception during periodic check at iter {iter_num}: {e}")

    # ========== EARLY STEP DEBUG (first 20 steps): all ranks report their status ==========
    # NOTE: all_gather can cause collective mismatch if ranks are desynchronized.
    # Use a simple barrier + local print instead for safer debugging.
    if ddp and iter_num < 20:
        _debug_log(f"iter={iter_num} START_OF_ITER")
        
        # Use barrier to detect hangs - if any rank hangs here, others will timeout
        try:
            _safe_barrier()
            _debug_log(f"iter={iter_num} PASSED_START_BARRIER")
        except Exception as e:
            _debug_log(f"iter={iter_num} BARRIER_FAILED: {e}")

    # ========== EVALUATION CHECKPOINT ==========
    if iter_num % args.eval_interval == 0:
        # DEBUG: Print before evaluation (expanded range)
        if iter_num < 20 or (iter_num % 100 == 0):
            _debug_log(f"iter={iter_num} ENTERING_EVAL_SECTION")
        
        # Eval rank participation strategy (match GPT script):
        # - ddp + FSDP: ALL ranks must run eval forward (FSDP triggers collectives)
        # - otherwise: rank0 only (efficiency); other ranks will proceed and naturally block
        #   at the next DDP collectives during backward.
        all_rank_eval = bool(ddp and using_fsdp)
        
        if iter_num < 20 or (iter_num % 100 == 0):
            _debug_log(f"iter={iter_num} all_rank_eval={all_rank_eval} ddp={ddp} using_fsdp={using_fsdp}")

        if all_rank_eval:
            _debug_log(f"iter={iter_num} ENTERING_EVAL_BARRIER")
            _safe_barrier()
            _debug_log(f"iter={iter_num} PASSED_EVAL_BARRIER")

        if args.distill_model:
            container = model.module if hasattr(model, "module") else model
            eval_model = container.student
        else:
            eval_model = model

        if all_rank_eval:
            losses = estimate_loss()
        else:
            if master_process:
                losses = estimate_loss()
            else:
                # placeholders to keep downstream logic type-stable
                losses = {'train': float('inf'), 'val': float('inf')}

        # WikiText PPL (optional): IMPORTANT
        # - If enabled under DDP/FSDP, ALL ranks must participate in the same collectives.
        # - Therefore we use eval_ppl_distributed() (all-rank) instead of rank0-only eval_ppl().
        _skip_wiki = getattr(args, 'skip_wiki_ppl', True)
        if master_process:
            print(f"[DEBUG wiki_ppl] skip_wiki_ppl={_skip_wiki}, type={type(_skip_wiki)}, bool(skip_wiki_ppl)={bool(_skip_wiki)}")
        if bool(_skip_wiki):
            wiki_ppl = float('inf')
        else:
            try:
                if all_rank_eval:
                    wiki_ppl = eval_ppl_distributed(
                        eval_model,
                        bs=2,
                        device=device,
                        block_size=block_size,
                        model_name_or_path=args.student_model,
                        ptdtype=ptdtype,
                    )
                else:
                    if master_process:
                        wiki_ppl = eval_ppl(
                            eval_model,
                            bs=2,
                            device=device,
                            block_size=block_size,
                            model_name_or_path=args.student_model,
                        )
                    else:
                        wiki_ppl = float('inf')
            except Exception as e:
                if master_process:
                    print(f"[WARN] WikiText PPL eval failed: {e}. Using inf.")
                wiki_ppl = float('inf')

        # Under DDP+FSDP, average metrics across ranks for stable logging/checkpoint decisions.
        # NOTE: If eval_ppl_distributed() was used, wiki_ppl is already averaged across ranks.
        # Only average it here if we used rank0-only eval_ppl() (all_rank_eval=False).
        if ddp and using_fsdp:
            try:
                import torch.distributed as _dist

                def _ddp_mean_float(x: float) -> float:
                    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
                    _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
                    t /= float(_dist.get_world_size())
                    return float(t.item())

                losses = {
                    'train': _ddp_mean_float(losses['train']),
                    'val': _ddp_mean_float(losses['val']),
                }
                # Only average wiki_ppl if we used rank0-only eval (all_rank_eval=False)
                # eval_ppl_distributed() already does cross-rank averaging internally
                if not all_rank_eval and math.isfinite(float(wiki_ppl)):
                    wiki_ppl = _ddp_mean_float(float(wiki_ppl))
            except Exception:
                # Best-effort: keep local metrics if reduction fails.
                losses = {'train': float(losses['train']), 'val': float(losses['val'])}
                wiki_ppl = float(wiki_ppl)

        # Synchronize after eval so all ranks resume training together (FSDP all-rank eval only)
        if all_rank_eval:
            _safe_barrier()

        # CRITICAL: Release GPU memory after eval.
        # Under FSDP (especially SHARD_GRAD_OP), eval forward all-gathers full parameters
        # which can consume significant memory. Force release before training resumes.
        torch.cuda.empty_cache()

        if master_process:
            now = time.time()
            if iter_num > 0:
                avg_iter = (now - training_start) / float(max(1, iter_num))
            else:
                avg_iter = 0.0
            eta_seconds = max(0.0, (max_iters - iter_num) * avg_iter)
            eta_time = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            eval_msg = (
                f"evaluating: iter_num {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, wiki_ppl {wiki_ppl:.4f}, eta_time {eta_time}"
            )
            # For iter 0, print with extra emphasis to ensure visibility
            if iter_num == 0:
                print("=" * 80, flush=True)
                print("[INITIAL EVAL] Before training starts:", flush=True)
            pbar.write(eval_msg)
            # Also print to stdout with flush to ensure visibility in SSH/multi-node setups
            print(eval_msg, flush=True)
            if iter_num == 0:
                print("=" * 80, flush=True)
            if wandb_log and wandb:
                wandb.log({
                    "iter": iter_num,
                    "val/loss": losses['val'],
                    "lr": get_lr(iter_num),
                    "wiki_ppl": wiki_ppl,
                    "eta_time": eta_time,
                    "eta_seconds": eta_seconds,
                }, step=iter_num)
            # write eval json
            try:
                eval_payload = {
                    "iter": int(iter_num),
                    "train_loss": float(losses['train']),
                    "val_loss": float(losses['val']),
                    "wiki_ppl": float(wiki_ppl),
                    "eta_time": eta_time,
                    "eta_seconds": float(eta_seconds),
                }
                eval_path = os.path.join(target_dir, "eval.json")
                tmp_path = eval_path + ".tmp"
                with open(tmp_path, "w") as ef:
                    json.dump(eval_payload, ef, indent=2)
                os.replace(tmp_path, eval_path)
            except Exception:
                pass

            eval_count += 1

        # ===== LM_EVAL IN FINALIZATION PHASE =====
        # 在 finalization 阶段（post-finalize finetune），若启用 --finalize_lm_eval，
        # 每次 eval 时额外运行 lm_eval harness 并保存 mean accuracy 最佳的 checkpoint。
        if finalization_done and args.finalize_lm_eval:
            lm_eval_results = {}
            _cur_mean = 0.0  # 所有 rank 初始化，避免非 master 进程变量未定义

            # 获取 eval 用的模型（所有 rank 都需要执行，为 FSDP summon 做准备）
            if args.distill_model:
                _container = model.module if hasattr(model, 'module') else model
                _eval_target = _container.student
            else:
                _eval_target = model

            # ---- 阶段 1: 在 summon_full_params 内收集完整参数（所有 rank 参与集合操作）----
            # lm_eval 推理只能在单 rank 上跑，但 summon_full_params 是集合操作，
            # 如果在 summon 内做推理，只有 rank 0 调用 forward，其他 rank 不调用，
            # 会导致 NCCL 集合操作不匹配 → 死锁/超时。
            # 解决方案：在 summon 内只做参数复制（快速），退出 summon 后 rank 0 单独推理。
            _lm_eval_state_dict = None  # rank 0 用来存完整参数
            _need_fsdp_gather = using_fsdp and FSDP is not None and isinstance(_eval_target, FSDP)

            if _need_fsdp_gather:
                # 使用 FULL_STATE_DICT 收集完整参数（所有 rank 参与，rank0_only=True 节省内存）
                if master_process:
                    print(f"[Finalize lm_eval] Gathering full params for lm_eval at iter {iter_num}...")
                try:
                    if StateDictType is not None and FullStateDictConfig is not None:
                        _fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(_eval_target, StateDictType.FULL_STATE_DICT, _fsdp_cfg):
                            _lm_eval_state_dict = _eval_target.state_dict()
                    else:
                        # fallback: summon_full_params + 手动复制
                        with FSDP.summon_full_params(_eval_target, writeback=False, recurse=True, offload_to_cpu=True):
                            if master_process:
                                _lm_raw = _eval_target.module if hasattr(_eval_target, 'module') else _eval_target
                                _lm_eval_state_dict = {k: v.clone().cpu() for k, v in _lm_raw.state_dict().items()}
                except Exception as _gather_e:
                    if master_process:
                        print(f"[Finalize lm_eval] WARNING: failed to gather params: {_gather_e}")
                        import traceback
                        traceback.print_exc()

            # 所有 rank 同步（确保集合操作全部完成后再继续）
            if ddp:
                try:
                    dist.barrier()
                except Exception:
                    pass

            # ---- 阶段 2: 仅 rank 0 用收集到的参数做 lm_eval 推理（无集合操作）----
            if master_process:
                print(f"[Finalize lm_eval] Running lm_eval at iter {iter_num}...")
                try:
                    from eval_wiki_ppl import run_lm_eval_benchmarks
                    from model_factory import create_model_skeleton

                    if _need_fsdp_gather and _lm_eval_state_dict is not None:
                        # 从收集到的 state_dict 创建独立的模型副本（非 FSDP，不触发集合操作）
                        # 注意：不能用 copy.deepcopy，因为 FSDP 包装的模型含有不可 pickle 的 C 模块
                        # 方案：用 create_model_skeleton 从 config 创建空壳模型，再加载 state_dict
                        _lm_eval_copy = create_model_skeleton(
                            args.student_model,
                            model_type=args.model_type,
                            override_args=dict(dropout=0.0, output_hidden_state=False, gradient_checkpointing=False),
                            sparselinear_config=sparselinear_config,
                            is_teacher=False,
                        )
                        # 加载完整参数（strict=False 容忍缺失的训练专用 key）
                        _load_info = _lm_eval_copy.load_state_dict(_lm_eval_state_dict, strict=False)
                        if _load_info.missing_keys:
                            print(f"[Finalize lm_eval] load_state_dict missing_keys ({len(_load_info.missing_keys)}): {_load_info.missing_keys[:5]}...")
                        _lm_eval_copy = _lm_eval_copy.to(device)
                        _lm_eval_copy.eval()
                        del _lm_eval_state_dict  # 释放 CPU 内存
                        _lm_eval_model = _lm_eval_copy
                        print(f"[Finalize lm_eval] Using standalone model copy (no FSDP collective ops)")
                    else:
                        # 非 FSDP 模式，直接用原始模型
                        _lm_eval_model = _eval_target.module if hasattr(_eval_target, 'module') else _eval_target

                    _lm_tasks = [t.strip() for t in args.lm_eval_tasks.split(',') if t.strip()]
                    lm_eval_results = run_lm_eval_benchmarks(
                        model=_lm_eval_model,
                        model_path=args.student_model,
                        device=str(device),
                        tasks=_lm_tasks,
                        batch_size=args.lm_eval_batch_size,
                        num_fewshot=0,
                    )

                    # 清理模型副本，释放 GPU 内存
                    if _need_fsdp_gather:
                        try:
                            del _lm_eval_model
                            del _lm_eval_copy
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                    if lm_eval_results:
                        _cur_mean = lm_eval_results.get('mean', 0.0)
                        print(f"[Finalize lm_eval] iter {iter_num}: mean_acc={_cur_mean:.2f}%")
                        for _tk, _tv in lm_eval_results.items():
                            if _tk != 'mean':
                                print(f"  - {_tk}: {_tv:.2f}%")

                        # wandb 日志
                        if wandb_log and wandb:
                            _lm_log = {f"lm_eval/{k}": v for k, v in lm_eval_results.items()}
                            _lm_log["iter"] = iter_num
                            wandb.log(_lm_log, step=iter_num)
                    else:
                        _cur_mean = 0.0
                        print(f"[Finalize lm_eval] iter {iter_num}: no results returned")

                except Exception as e:
                    print(f"[Finalize lm_eval] WARNING: lm_eval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    _cur_mean = 0.0
            else:
                # 非 master rank：清理可能残留的 state_dict
                if _lm_eval_state_dict is not None:
                    del _lm_eval_state_dict

            # 同步所有 rank（rank 0 完成 lm_eval 后）
            if ddp:
                try:
                    dist.barrier()
                except Exception:
                    pass

            # 广播 _cur_mean 到所有 rank，以便一致地判断是否保存
            if ddp:
                try:
                    _mean_t = torch.tensor([_cur_mean if master_process else 0.0], device=device, dtype=torch.float32)
                    dist.broadcast(_mean_t, src=0)
                    _cur_mean = float(_mean_t.item())
                except Exception:
                    pass

            # 判断是否为 best lm_eval，并保存 checkpoint
            if _cur_mean > best_lm_eval_mean:
                best_lm_eval_mean = _cur_mean
                if master_process:
                    print(f"[Finalize lm_eval] New best lm_eval mean: {best_lm_eval_mean:.2f}% at iter {iter_num}")

                # 收集 state_dict（FSDP 是集合操作，所有 rank 必须参与）
                _lm_model_sd = None
                try:
                    if using_fsdp:
                        if StateDictType is not None and FullStateDictConfig is not None and FSDP is not None:
                            _fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                            _save_tgt = raw_model.student if args.distill_model else model
                            with FSDP.state_dict_type(_save_tgt, StateDictType.FULL_STATE_DICT, _fsdp_cfg):
                                _lm_model_sd = _save_tgt.state_dict()
                    else:
                        _save_mdl = raw_model.student if args.distill_model else raw_model
                        _lm_model_sd = _save_mdl.state_dict()
                except Exception as _sd_e:
                    if master_process:
                        print(f"[Finalize lm_eval] WARNING: failed to gather state_dict: {_sd_e}")

                if master_process and _lm_model_sd is not None:
                    try:
                        _lm_ckpt = {
                            'model_state_dict': _lm_model_sd,
                            'iter_num': iter_num,
                            'args': vars(args),
                            'best_lm_eval_mean': best_lm_eval_mean,
                            'lm_eval_results': lm_eval_results,
                            'finalization_done': True,
                        }
                        # 保存 channel pruning mask 状态（如果启用）
                        if channel_pruning_enabled and channel_mask_state is not None:
                            _lm_ckpt['channel_masks'] = {
                                i: {
                                    "mask": channel_mask_state.masks[i].mask.cpu(),
                                    "temperature": channel_mask_state.masks[i].temperature,
                                    "hardening_x": channel_mask_state.masks[i].hardening_x,
                                }
                                for i in range(channel_mask_state.num_layers)
                            }
                        _lm_save_path = os.path.join(target_dir, "model_best_lm_eval.pt")
                        torch.save(_lm_ckpt, _lm_save_path)
                        print(f"[Finalize lm_eval] Saved best lm_eval checkpoint to {_lm_save_path}")

                        # 同时保存 lm_eval 结果 json
                        _lm_eval_json = {
                            'iter': iter_num,
                            'best_lm_eval_mean': best_lm_eval_mean,
                            'lm_eval_results': lm_eval_results,
                            'wiki_ppl': float(wiki_ppl) if math.isfinite(float(wiki_ppl)) else None,
                            'val_loss': float(losses.get('val', float('inf'))),
                        }
                        with open(os.path.join(target_dir, 'best_lm_eval.json'), 'w') as _jf:
                            json.dump(_lm_eval_json, _jf, indent=2)

                        del _lm_ckpt
                    except Exception as _save_e:
                        print(f"[Finalize lm_eval] WARNING: failed to save checkpoint: {_save_e}")

                if _lm_model_sd is not None:
                    del _lm_model_sd
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                if ddp:
                    try:
                        dist.barrier()
                    except Exception:
                        pass

        # ===== CHECKPOINT SAVE (FSDP-aware) =====
        need_save_local = (
            (iter_num % save_interval == 0)
            and (eval_count >= 2)
            and (wiki_ppl < best_wiki_ppl or always_save_checkpoint)
            and (iter_num > 0)
        )
        if ddp and using_fsdp:
            try:
                import torch.distributed as _dist
                flag = 1 if (master_process and need_save_local) else 0
                save_flag = torch.tensor([flag], device=device, dtype=torch.int32)
                _dist.broadcast(save_flag, src=0)
                need_save = int(save_flag.item()) == 1
            except Exception:
                need_save = False
        else:
            need_save = bool(master_process and need_save_local)

        if need_save:
            if master_process:
                best_val_loss = losses['val']
                best_wiki_ppl = wiki_ppl

            model_state_dict = None
            if using_fsdp:
                if StateDictType is None or FullStateDictConfig is None or FSDP is None:
                    raise RuntimeError("FSDP state_dict utilities unavailable")
                fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                # In distill mode, `model` is a container (Distill_Model) and the student
                # is the actual FSDP module we want to checkpoint.
                save_target = raw_model.student if args.distill_model else model
                with FSDP.state_dict_type(save_target, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                    model_state_dict = save_target.state_dict()
            else:
                save_model = raw_model.student if args.distill_model else raw_model
                model_state_dict = save_model.state_dict()

            if master_process:
                checkpoint = {
                    'model': model_state_dict,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'args': vars(args),
                }
                try:
                    if 'optimizer' in locals() and optimizer is not None:
                        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                except Exception:
                    pass
                try:
                    if 'scaler' in locals() and scaler is not None and hasattr(scaler, 'state_dict'):
                        checkpoint['scaler_state_dict'] = scaler.state_dict()
                except Exception:
                    pass

                # Model save path
                model_save_path = os.path.join(target_dir, "model.pt")
                model_checkpoint = {
                    'model_state_dict': model_state_dict,
                    'iter_num': iter_num,
                    'args': vars(args)
                }
                # 保存 channel pruning mask 状态（如果启用）
                if channel_pruning_enabled and channel_mask_state is not None:
                    model_checkpoint['channel_masks'] = {
                        i: {
                            "mask": channel_mask_state.masks[i].mask.cpu(),
                            "temperature": channel_mask_state.masks[i].temperature,
                            "hardening_x": channel_mask_state.masks[i].hardening_x,
                        }
                        for i in range(channel_mask_state.num_layers)
                    }
                try:
                    if 'optimizer' in locals() and optimizer is not None:
                        model_checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                except Exception:
                    pass
                try:
                    if 'scaler' in locals() and scaler is not None and hasattr(scaler, 'state_dict'):
                        model_checkpoint['scaler_state_dict'] = scaler.state_dict()
                except Exception:
                    pass

                # also extract SLoRB / projection related params explicitly for clarity
                try:
                    slorb_keys = [k for k in model_checkpoint['model_state_dict'].keys() if ('SLoRB' in k or 'x_proj' in k or '.proj' in k or 'proj' in k)]
                    slorb_state = {k: model_checkpoint['model_state_dict'][k] for k in slorb_keys}
                    model_checkpoint['slorb_state_dict'] = slorb_state
                    if master_process:
                        print(f"[save] included slorb_state_dict with {len(slorb_state)} entries")
                except Exception:
                    pass

                torch.save(model_checkpoint, model_save_path)
                print(f"[save] iter={iter_num}, saved to {model_save_path}, wiki_ppl={wiki_ppl:.4f}")

                # save legacy-style checkpoint into the same folder for compatibility
                try:
                    torch.save(checkpoint, os.path.join(target_dir, "legacy_ckpt.pt"))
                except Exception:
                    pass

                # update latest link
                try:
                    last_link = os.path.join(out_dir, "last")
                    if os.path.islink(last_link) or os.path.exists(last_link):
                        os.remove(last_link)
                    os.symlink(target_dir, last_link)
                except Exception:
                    try:
                        with open(os.path.join(out_dir, "last_dir.txt"), "w") as f:
                            f.write(target_dir + "\n")
                    except Exception:
                        pass

                # save a json copy of config for quick inspection
                try:
                    with open(os.path.join(target_dir, 'config.json'), 'w') as f:
                        json.dump(checkpoint['config'], f, indent=2)
                except Exception:
                    # fallback: skip json if not serializable
                    pass

                # Save args json
                try:
                    args_dict = vars(args)
                    safe_args = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in args_dict.items()}
                    with open(os.path.join(target_dir, 'args.json'), 'w') as f:
                        json.dump(safe_args, f, indent=2)
                except Exception:
                    pass

            if ddp and using_fsdp:
                _safe_barrier()

                if master_process:
                    save_msg = f"saving checkpoint to {target_dir}"
                    pbar.write(save_msg)
                    # Also print to stdout with flush to ensure visibility
                    print(save_msg, flush=True)
            
            # CRITICAL: Release memory used by checkpoint saving
            # model_state_dict can be very large (full model weights offloaded to CPU)
            del model_state_dict
            if master_process:
                del checkpoint, model_checkpoint
                try:
                    del slorb_state
                except Exception:
                    pass
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # CRITICAL: Explicit barrier after eval section to ensure all ranks are synchronized
    # before starting the forward pass. This is especially important for HYBRID_SHARD mode
    # where the mesh_shard process group requires tight synchronization.
    if ddp and using_fsdp and (iter_num % args.eval_interval == 0):
        _debug_log(f"iter={iter_num} POST_EVAL_BARRIER_START")
        _safe_barrier()
        _debug_log(f"iter={iter_num} POST_EVAL_BARRIER_DONE")
        # Extra CUDA sync to ensure all GPU operations from eval are complete
        torch.cuda.synchronize()

    # ========== FORWARD/BACKWARD TRAINING STEP ==========
    skip_step = False
    
    # Initialize variables that will be set inside the micro-step loop but used outside for logging
    mid = None
    cur_sparsity = None
    sparsity_pen = torch.tensor(0.0, device=device)  # Initialize as tensor for safe .item() call
    train_student = raw_model.student if args.distill_model else raw_model

    # Hutchinson in multi-rank FSDP requires careful handling due to collective operations.
    # By default, we allow it if args.enable_hutchinson is True. User can set DISABLE_HUTCHINSON_FSDP=1 to disable.
    hutchinson_active = bool(args.enable_hutchinson)
    if hutchinson_active and ddp and using_fsdp:
        # Check if user wants to force-disable Hutchinson under FSDP
        if os.environ.get("DISABLE_HUTCHINSON_FSDP", "0") == "1":
            hutchinson_active = False
            if master_process and (iter_num == 0):
                print("[Hutchinson] Disabled under multi-rank FSDP (DISABLE_HUTCHINSON_FSDP=1).")
        else:
            # Hutchinson is enabled under FSDP - warn about potential issues
            if master_process and (iter_num == 0):
                print("[Hutchinson] Enabled under multi-rank FSDP. If you see collective mismatch errors, set DISABLE_HUTCHINSON_FSDP=1.")

    for micro_step in range(gradient_accumulation_steps):
        is_last_micro = (micro_step == gradient_accumulation_steps - 1)

        # Gradient sync control
        no_sync_ctx = nullcontext()
        fsdp_sync_model = raw_model.student if (args.distill_model and hasattr(raw_model, 'student')) else model
        if using_fsdp and hasattr(fsdp_sync_model, 'no_sync') and (not is_last_micro):
            no_sync_ctx = fsdp_sync_model.no_sync()
        elif ddp and (not using_fsdp) and hasattr(model, 'require_backward_grad_sync'):
            model.require_backward_grad_sync = is_last_micro

        with no_sync_ctx:
            # DEBUG: Print before forward (expanded range to catch iter=13 issue)
            if iter_num < 20 or (iter_num % 100 == 0):
                _debug_log(f"iter={iter_num} micro={micro_step} BEFORE_FORWARD X.shape={X.shape}")
            
            with ctx:
                # Forward pass (轻量级时间测量)
                if micro_step == 0:
                    forward_start_time = time.time()
                
                # record_function 仅在非 FSDP 模式下使用 (FSDP 模式下可能导致 rank 不同步)
                forward_ctx = record_function("forward_pass") if (profiler_enabled and profiler_active_this_step) else nullcontext()
                with forward_ctx:
                    if args.distill_model:
                        logits, task_loss, layerwise_loss, kl_loss = model(X, Y)
                        if task_loss is None:
                            task_loss = 0.0
                        if layerwise_loss is None:
                            loss = hardness_task * task_loss + hardness_kldiv * kl_loss
                        else:
                            loss = hardness_task * task_loss + hardness_squarehead * layerwise_loss + hardness_kldiv * kl_loss
                        
                        # DEBUG: Check each loss component for NaN (first 100 iters or every 100 iters)
                        if iter_num < 100 or (iter_num % 100 == 0):
                            task_val = task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
                            kl_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                            layer_val = layerwise_loss.item() if isinstance(layerwise_loss, torch.Tensor) else (layerwise_loss if layerwise_loss is not None else 0.0)
                            logits_has_nan = torch.isnan(logits).any().item() if logits is not None else False
                            logits_has_inf = torch.isinf(logits).any().item() if logits is not None else False
                            _debug_log(f"iter={iter_num} LOSS_COMPONENTS: task={task_val:.4f} kl={kl_val:.4f} layer={layer_val:.4f} logits_nan={logits_has_nan} logits_inf={logits_has_inf}")
                    else:
                        logits, loss, hidden_states = model(X, Y)

                # DEBUG: Print after forward (expanded range)
                if iter_num < 20 or (iter_num % 100 == 0):
                    _debug_log(f"iter={iter_num} micro={micro_step} AFTER_FORWARD loss={loss.item():.4f}")

                # Mid penalty schedule: fast warmup to lambda_mid_max
                # Original slow schedule (over entire training): lambda_mid_max * (iter_num / max_iters)
                # New fast schedule: reach max within 500 steps, then hold constant
                lambda_mid_warmup_steps = 500  # 快速预热步数
                if iter_num < lambda_mid_warmup_steps:
                    lambda_mid = args.lambda_mid_max * (iter_num / lambda_mid_warmup_steps)
                else:
                    lambda_mid = args.lambda_mid_max

                mid = mid_penalty(train_student, lambda_mid)
                loss = loss + mid

                # Sparsity penalty: encourage global hard sparsity to match args.sparsity_ratio
                sparsity_pen = torch.tensor(0.0, device=next(model.parameters()).device)
                cur_sparsity = None
                if args.sparsity_alpha is not None and float(args.sparsity_alpha) != 0.0:
                    sparsity_pen, cur_sparsity = sparsity_penalty(train_student, target_sparsity=args.sparsity_ratio, alpha=args.sparsity_alpha)
                    loss = loss + sparsity_pen

                # Scale for grad accumulation
                loss = loss / gradient_accumulation_steps

            # Match main.py data-advance behavior: prefetch next batch every micro-step.
            # IMPORTANT: keep this outside autocast context and before any potential early-exit.
            X, Y = get_batch('train')

            # ========== CRITICAL: All-ranks loss finite check ==========
            # If any rank has non-finite loss, all ranks must skip backward/step consistently
            # to avoid FSDP collective mismatch (some ranks in backward reduce_scatter, others in eval/save gather).
            loss_detached = loss.detach()
            is_finite_local = torch.isfinite(loss_detached).to(torch.int32)
            if ddp:
                try:
                    import torch.distributed as _dist
                    if _dist.is_initialized():
                        _dist.all_reduce(is_finite_local, op=_dist.ReduceOp.MIN)
                except Exception:
                    pass
            all_finite = bool(is_finite_local.item())
            if not all_finite:
                # DEBUG: Print detailed info about which component caused the NaN
                local_is_nan = not torch.isfinite(loss_detached).item()
                if local_is_nan:
                    # This rank has NaN - print detailed breakdown
                    debug_info = f"[step {iter_num}][rank{ddp_rank}] LOCAL NaN detected!"
                    if args.distill_model:
                        task_val = task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss
                        kl_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                        mid_val = mid.item() if mid is not None and isinstance(mid, torch.Tensor) else 0.0
                        sp_val = sparsity_pen.item() if isinstance(sparsity_pen, torch.Tensor) else 0.0
                        debug_info += f" task_loss={task_val} kl_loss={kl_val} mid={mid_val} sparsity_pen={sp_val}"
                        # Check logits for NaN/Inf
                        if logits is not None:
                            debug_info += f" logits_nan={torch.isnan(logits).any().item()} logits_inf={torch.isinf(logits).any().item()}"
                            debug_info += f" logits_max={logits.max().item():.2f} logits_min={logits.min().item():.2f}"
                    print(debug_info)
                if master_process:
                    print(f"[step {iter_num}] Non-finite loss detected on at least one rank; skipping backward on ALL ranks to maintain FSDP consistency.")
                skip_step = True
                break

            # Hutchinson Hessian update (before backward)
            # CRITICAL: DDP-safe version - all ranks must agree on whether to run/skip
            if hutchinson_active and is_last_micro:
                hutch_ok = torch.tensor(1, dtype=torch.int32, device=loss.device)
                try:
                    update_hessian_hutchinson(train_student, loss)
                except RuntimeError as e:
                    hutch_ok.fill_(0)
                    if master_process:
                        print(f"[Hutchinson] local error: {e}")
                # All-reduce to check if any rank failed
                if ddp:
                    try:
                        import torch.distributed as _dist
                        if _dist.is_initialized():
                            _dist.all_reduce(hutch_ok, op=_dist.ReduceOp.MIN)
                    except Exception:
                        pass
                if not bool(hutch_ok.item()) and master_process:
                    print(f"[Hutchinson] skipped globally (at least one rank failed)")

            # Backward pass
            # DEBUG: Add barrier before backward to ensure all ranks are at the same point.
            # This is expensive but helps diagnose FSDP collective mismatch issues.
            # Remove after debugging is complete.
            if ddp and (iter_num < 50):  # Only for first 50 iters to catch early issues
                _debug_log(f"iter={iter_num} micro={micro_step} ENTERING_BWD_BARRIER")
                try:
                    _safe_barrier()
                    _debug_log(f"iter={iter_num} micro={micro_step} PASSED_BWD_BARRIER")
                except Exception as e:
                    _debug_log(f"iter={iter_num} micro={micro_step} BWD_BARRIER_FAILED: {e}")
            
            # DEBUG: Print before backward (expanded range to catch iter=13 issue)
            if iter_num < 20 or (iter_num % 100 == 0):
                _debug_log(f"iter={iter_num} micro={micro_step} BEFORE_BACKWARD loss={loss.item():.4f}")
            
            # 轻量级时间测量: backward
            if micro_step == 0:
                # 同步 GPU 以准确测量 forward 时间
                if args.enable_profiler and torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_start_time = time.time()
            
            try:
                # record_function 仅在非 FSDP 模式下使用
                backward_ctx = record_function("backward_pass") if (profiler_enabled and profiler_active_this_step) else nullcontext()
                with backward_ctx:
                    scaler.scale(loss).backward()
            except Exception as e:
                _debug_log(f"!!!BACKWARD FAILED!!! iter={iter_num} micro={micro_step} error={e}")
                import traceback
                traceback.print_exc()
                raise
            
            # DEBUG: Print after backward
            if iter_num < 20 or (iter_num % 100 == 0):
                _debug_log(f"iter={iter_num} micro={micro_step} AFTER_BACKWARD")

    # If any rank had non-finite loss, all ranks must skip optimizer step/mask/logging.
    if skip_step:
        optimizer.zero_grad(set_to_none=True)
        if master_process:
            pbar.update(1)
        iter_num += 1
        local_iter_num += 1
        continue

    # ========== GRADIENT CLIPPING & OPTIMIZER STEP ==========
    # 轻量级时间测量: optimizer
    if args.enable_profiler and torch.cuda.is_available():
        torch.cuda.synchronize()
    optimizer_start_time = time.time()
    
    # record_function 仅在非 FSDP 模式下使用
    optimizer_ctx = record_function("optimizer_step") if (profiler_enabled and profiler_active_this_step) else nullcontext()
    with optimizer_ctx:
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_student.parameters(), grad_clip)

        # Grad EMA update (for movement/hessian metrics)
        # This MUST be called BEFORE optimizer.zero_grad() to capture gradients
        if args.mask_metric in ["movement", "hessian_obd", "hessian_ratio", "hessian"]:
            update_model_grad_ema(train_student, update_hessian_with_grad2=not hutchinson_active)

        # Channel Pruning: 在 optimizer.zero_grad() 之前更新 Hessian EMA
        if channel_pruning_enabled and channel_score_computer is not None:
            if args.channel_importance_metric in ["hessian_obd", "taylor"]:
                channel_score_computer.update_all_hessian_ema()

        # Optimizer step
        lr = get_lr(iter_num)
        decay = get_decay(iter_num)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        scaler.step(optimizer, decay=decay)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # ========== MASK UPDATE & LOGGING ==========
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # Update mask_penalty_lr based on schedule (if enabled)
    current_penalty_lr = None
    if args.mask_penalty_lr_schedule != 'constant' and args.mask_penalty_lr_min is not None:
        penalty_lr_max = args.mask_penalty_lr if args.mask_penalty_lr is not None else args.mask_lr
        current_penalty_lr = update_mask_penalty_lr(
            train_student, iter_num, max_iters,
            penalty_lr_min=args.mask_penalty_lr_min,
            penalty_lr_max=penalty_lr_max,
            schedule=args.mask_penalty_lr_schedule
        )
    
    # Update masks: SparseLinear.update_mask() internally checks:
    # 1. self.change_mask (skip if False)
    # 2. step % mask_update_period (skip if not time to update)
    # Skip mask update on the LAST iteration (iter_num == max_iters) to ensure
    # the finalization uses the same mask as the final eval (which happens at
    # iter_num == max_iters when max_iters % eval_interval == 0).
    if iter_num < max_iters:
        calculate_model_mask(train_student, iter_num, lambda_mid=lambda_mid)
    
    # Channel Pruning: 更新 Hessian EMA + PCA 特征分解 + channel mask
    if channel_pruning_enabled and channel_mask_state is not None and channel_score_computer is not None:
        # 每步更新 Hessian EMA（利用当前梯度信息）
        if iter_num >= args.channel_sparsity_warmup_steps:
            channel_score_computer.update_all_hessian_ema()
        
        # PCA: 定期更新特征分解（开销较大，按 pca_update_period 周期执行）
        if args.enable_pca_scoring and channel_score_computer.pca_manager is not None:
            channel_score_computer.update_pca(iter_num)
        
        # 更新 channel mask（按 channel_mask_update_period 周期）
        if iter_num < max_iters and iter_num % args.channel_mask_update_period == 0:
            if iter_num >= args.channel_sparsity_warmup_steps:
                all_ch_scores = channel_score_computer.compute_all_scores()
                ch_score_tensors = [ls.scores for ls in all_ch_scores]
                channel_mask_state.update_all_masks(ch_score_tensors, iter_num)
                channel_score_computer.update_score_ema(all_ch_scores)
        
        # 线性回归恢复: 在 mask hardening 跨过 0.5 阈值时执行
        if channel_regression_recovery is not None:
            # 控制校准数据收集：在 hardening_x 接近 0.5 时开始收集
            # 收集窗口: hardening_x 在 [0.4, 0.7] 之间
            any_needs_collection = False
            for li in range(channel_mask_state.num_layers):
                hx = channel_mask_state.masks[li].hardening_x
                if 0.3 < hx <= 0.7:
                    any_needs_collection = True
                    break
            channel_regression_recovery._collecting = any_needs_collection
            
            # 检查并执行回归
            channel_regression_recovery.maybe_run_regression(iter_num, master_process=master_process)
    
    # log_mask_stats: MUST be called on ALL ranks (uses dist.all_reduce internally)
    # Only master_process will emit to wandb, but all ranks participate in collective
    if iter_num % args.output_flip_every == 0:
        log_mask_stats(train_student, iter_num, wandb_run=wandb if (wandb_log and master_process) else None)

    # nm_2_4_tile_stats: MUST be called on ALL ranks (uses dist.all_reduce internally)
    # 提前在所有 rank 上计算，避免放在 master_process 分支内导致 all_reduce 死锁
    tile_stats = None
    if iter_num % args.output_flip_every == 0:
        if getattr(args, 'hard_mask_type', '') == 'nm_2_4' or args.mask_type == 'structured':
            tile_stats = nm_2_4_tile_stats(train_student, debug=(master_process and iter_num < 20))

    if iter_num % args.output_flip_every == 0 and master_process:
        # Compute flip rate (matches main.py)
        flipped, flipped_ratio, init_flipped, init_flipped_ratio = calculate_flip_rate(train_student)
        loss_scale_for_log = gradient_accumulation_steps
        lossf = loss.item() * loss_scale_for_log
        if args.distill_model:
            task_lossf = task_loss.item()
            layerwise_lossf = layerwise_loss.item() if layerwise_loss is not None else 0
            kl_lossf = kl_loss.item()
            pbar.set_postfix(loss=lossf, flip_ratio=flipped_ratio, **({'ch_sp': f"{channel_mask_state.get_sparsity_stats()['hard_sparsity']:.3f}"} if channel_pruning_enabled and channel_mask_state is not None else {}))
            if wandb_log and wandb:
                # mid is computed per micro-step; use last micro-step value (same as main.py)
                midf = float(mid.item() * loss_scale_for_log) if mid is not None else 0.0
                log_data = {
                    "iter": iter_num,
                    "flip_num": flipped,
                    "init_flip_num": init_flipped,
                    "flip_ratio": flipped_ratio,
                    "init_flip_ratio": init_flipped_ratio,
                    "train/loss": lossf,
                    "train/task_loss": task_lossf,
                    "train/layerwise_loss": layerwise_lossf,
                    "train/kl_loss": kl_lossf,
                    "train/mid_penalty": midf,
                    "lr": lr,
                    "time": dt*1000,
                    "srste_decay": decay,
                }
                if cur_sparsity is not None:
                    log_data["train/current_sparsity"] = float(cur_sparsity)
                    log_data["train/sparsity_penalty"] = float(sparsity_pen.item())
                # 记录 mask_penalty_lr schedule
                if current_penalty_lr is not None:
                    log_data["train/mask_penalty_lr"] = current_penalty_lr
                # 2:4 Tile 统计 (已在所有 rank 上提前计算)
                if tile_stats is not None:
                    log_data["nm24/top2_avg"] = tile_stats["top2_avg"]
                    log_data["nm24/bot2_avg"] = tile_stats["bot2_avg"]
                    log_data["nm24/top2_gap"] = tile_stats["top2_gap"]
                    log_data["nm24/bot2_avg"] = tile_stats["bot2_avg"]
                    log_data["nm24/total_gap"] = tile_stats["total_gap"]
                # Channel Pruning 统计
                if channel_pruning_enabled and channel_mask_state is not None:
                    ch_stats = channel_mask_state.get_sparsity_stats()
                    log_data["channel/hard_sparsity"] = ch_stats["hard_sparsity"]
                    log_data["channel/avg_mask"] = ch_stats["avg_mask"]
                wandb.log(log_data, step=iter_num)
        else:
            pbar.set_postfix(loss=lossf, flip_ratio=flipped_ratio, **({'ch_sp': f"{channel_mask_state.get_sparsity_stats()['hard_sparsity']:.3f}"} if channel_pruning_enabled and channel_mask_state is not None else {}))
            if wandb_log and wandb:
                midf = float(mid.item() * loss_scale_for_log) if mid is not None else 0.0
                log_data = {
                    "iter": iter_num,
                    "flip_num": flipped,
                    "init_flip_num": init_flipped,
                    "flip_ratio": flipped_ratio,
                    "init_flip_ratio": init_flipped_ratio,
                    "train/loss": lossf,
                    "train/mid_penalty": midf,
                    "lr": lr,
                    "time": dt*1000,
                    "srste_decay": decay,
                }
                if cur_sparsity is not None:
                    log_data["train/current_sparsity"] = float(cur_sparsity)
                    log_data["train/sparsity_penalty"] = float(sparsity_pen.item())
                # 记录 mask_penalty_lr schedule
                if current_penalty_lr is not None:
                    log_data["train/mask_penalty_lr"] = current_penalty_lr
                # 2:4 Tile 统计 (已在所有 rank 上提前计算)
                if tile_stats is not None:
                    log_data["nm24/top2_avg"] = tile_stats["top2_avg"]
                    log_data["nm24/bot2_avg"] = tile_stats["bot2_avg"]
                    log_data["nm24/top2_gap"] = tile_stats["top2_gap"]
                    log_data["nm24/bot2_gap"] = tile_stats["bot2_gap"]
                    log_data["nm24/total_gap"] = tile_stats["total_gap"]
                # Channel Pruning 统计
                if channel_pruning_enabled and channel_mask_state is not None:
                    ch_stats = channel_mask_state.get_sparsity_stats()
                    log_data["channel/hard_sparsity"] = ch_stats["hard_sparsity"]
                    log_data["channel/avg_mask"] = ch_stats["avg_mask"]
                wandb.log(log_data, step=iter_num)

    if master_process:
        pbar.update(1)

    # ========== 轻量级时间统计 (FSDP-safe) ==========
    # 这部分代码在所有 rank 上执行相同操作，不会导致 collective 不同步
    if args.enable_profiler:
        step_end_time = time.time()
        step_total_ms = (step_end_time - step_start_time) * 1000.0
        forward_ms = (backward_start_time - forward_start_time) * 1000.0 if (forward_start_time and backward_start_time) else 0.0
        backward_ms = (optimizer_start_time - backward_start_time) * 1000.0 if (backward_start_time and optimizer_start_time) else 0.0
        optimizer_ms = (step_end_time - optimizer_start_time) * 1000.0 if optimizer_start_time else 0.0
        
        # 累积统计
        step_timing['forward_ms'] += forward_ms
        step_timing['backward_ms'] += backward_ms
        step_timing['optimizer_ms'] += optimizer_ms
        step_timing['total_ms'] += step_total_ms
        step_timing['num_steps'] += 1
        
        # 定期输出统计 (仅 rank 0)
        # 每隔 profiler_log_interval 步输出一次，从 profiler_start_step 开始
        should_log_timing = (iter_num >= args.profiler_start_step) and (
            (iter_num - args.profiler_start_step) % args.profiler_log_interval == 0
        )
        if master_process and should_log_timing:
            n = step_timing['num_steps']
            avg_forward = step_timing['forward_ms'] / max(n, 1)
            avg_backward = step_timing['backward_ms'] / max(n, 1)
            avg_optimizer = step_timing['optimizer_ms'] / max(n, 1)
            avg_total = step_timing['total_ms'] / max(n, 1)
            
            # 估算 compute vs communication 比例
            # 注意: 这是粗略估计，因为 forward/backward 中包含了 FSDP 的 all_gather/reduce_scatter
            compute_ms = avg_forward + avg_backward + avg_optimizer
            other_ms = max(0, avg_total - compute_ms)
            
            print(f"\n[Timing step {iter_num}] Avg over {n} steps:")
            print(f"  Forward: {avg_forward:.1f}ms, Backward: {avg_backward:.1f}ms, Optimizer: {avg_optimizer:.1f}ms")
            print(f"  Total step: {avg_total:.1f}ms (compute: {compute_ms:.1f}ms, other/overhead: {other_ms:.1f}ms)")
            
            # Estimate bottleneck based on forward+backward vs optimizer ratio
            if avg_total > 0:
                fwd_bwd_ratio = (avg_forward + avg_backward) / avg_total
                if fwd_bwd_ratio > 0.8:
                    print(f"  >>> LIKELY COMPUTE-BOUND (forward+backward = {fwd_bwd_ratio*100:.1f}% of step)")
                elif fwd_bwd_ratio < 0.5:
                    print(f"  >>> LIKELY COMMUNICATION/IO-BOUND (forward+backward = {fwd_bwd_ratio*100:.1f}% of step)")
                else:
                    print(f"  >>> BALANCED (forward+backward = {fwd_bwd_ratio*100:.1f}% of step)")
            
            # Log to wandb
            if wandb_log and wandb:
                wandb.log({
                    'timing/avg_forward_ms': avg_forward,
                    'timing/avg_backward_ms': avg_backward,
                    'timing/avg_optimizer_ms': avg_optimizer,
                    'timing/avg_total_ms': avg_total,
                    'timing/compute_ms': compute_ms,
                }, step=iter_num)
            
            # Reset stats for next interval
            step_timing['forward_ms'] = 0.0
            step_timing['backward_ms'] = 0.0
            step_timing['optimizer_ms'] = 0.0
            step_timing['total_ms'] = 0.0
            step_timing['num_steps'] = 0

    # ========== Full Profiler step (非 FSDP 模式) ==========
    # 统计数据在 profiler_trace_handler 中自动收集，这里只需要调用 step()
    # 重要：Profiler 的 on_trace_ready 回调会在 rank 0 上执行 I/O 操作，
    # 这可能导致 rank 0 延迟，而其他 rank 已经进入下一步的 forward，
    # 触发 FSDP 的 all_gather 集合通信，导致 rank 不同步和超时。
    # 解决方案：在 profiler 活跃期间，每个 step 结束后添加 barrier 同步。
    profiler_needs_sync = False
    if profiler_enabled and master_process and profiler_ctx is not None:
        profiler_end_step = args.profiler_start_step + (args.profiler_warmup_steps + args.profiler_active_steps) * args.profiler_repeat
        if args.profiler_start_step <= iter_num < profiler_end_step:
            profiler_needs_sync = True
            try:
                # 调用 profiler step (会触发 trace_handler 如果活跃周期结束)
                profiler_ctx.step()
                
                # 定期 log 最近的统计到 wandb
                if wandb_log and wandb and profiler_last_step_stats and (iter_num % args.profiler_log_interval == 0):
                    wandb.log({
                        'profiler/cuda_time_ms': profiler_last_step_stats.get('cuda_time_ms', 0),
                        'profiler/compute_time_ms': profiler_last_step_stats.get('compute_time_ms', 0),
                        'profiler/comm_time_ms': profiler_last_step_stats.get('comm_time_ms', 0),
                        'profiler/compute_ratio': profiler_last_step_stats.get('compute_ratio', 0),
                        'profiler/comm_ratio': profiler_last_step_stats.get('comm_ratio', 0),
                    }, step=iter_num)
            except Exception as e:
                if iter_num % 100 == 0:
                    print(f"[Profiler] Warning at step {iter_num}: {e}")
    
    # FSDP + Profiler 同步: 等待 rank 0 完成 profiler I/O 后再继续
    # 所有 rank 都需要参与这个 barrier，不管它们是否运行 profiler
    if profiler_enabled and ddp and using_fsdp:
        profiler_end_step = args.profiler_start_step + (args.profiler_warmup_steps + args.profiler_active_steps) * args.profiler_repeat
        if args.profiler_start_step <= iter_num < profiler_end_step:
            try:
                dist.barrier()
            except Exception as e:
                if master_process:
                    print(f"[Profiler] Barrier failed at step {iter_num}: {e}")

    iter_num += 1
    local_iter_num += 1

    # Hardening schedule
    hardening_start = int(0.6 * max_iters)
    hardening_enabled = (args.hardening_fraction > 0) and (args.hardening_period is not None) and (args.hardening_period > 0)
    if hardening_enabled and iter_num >= hardening_start:
        if (iter_num - hardening_start) % args.hardening_period == 0:
            hardened = harden_fraction(model, fraction=args.hardening_fraction)
            if master_process:
                print(f"[Hardening] iter {iter_num}: hardened {hardened} entries")

    # ========== TRAINING TERMINATION & FINALIZATION ==========
    if iter_num > max_iters:
        if not finalization_done:
            if master_process:
                try:
                    pbar.set_description("Finalizing masks")
                    pbar.refresh()
                except Exception:
                    pass

            # ========== RELEASE UNNECESSARY MEMORY BEFORE FINALIZATION ==========
            # This is critical to avoid OOM during mask hardening with FSDP
            if master_process:
                print("[Finalization] Releasing unnecessary memory before mask hardening...")
            
            # 1. Release optimizer state (can be very large with Adam)
            # Only release if NOT doing post-finalize finetune
            if args.final_finetune_iters <= 0:
                if 'optimizer' in dir() and optimizer is not None:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        # Clear optimizer state dict to free memory
                        for state in optimizer.state.values():
                            for k, v in list(state.items()):
                                if isinstance(v, torch.Tensor):
                                    state[k] = None
                        optimizer.state.clear()
                        if master_process:
                            print("[Finalization] Optimizer state cleared")
                    except Exception as e:
                        if master_process:
                            print(f"[Finalization] Warning: failed to clear optimizer: {e}")
            else:
                if master_process:
                    print("[Finalization] Keeping optimizer state for post-finalize finetune")
            
            # 2. Release GradScaler if exists (only if NOT doing post-finalize finetune)
            if args.final_finetune_iters <= 0:
                if 'scaler' in dir() and scaler is not None:
                    try:
                        scaler = None
                        if master_process:
                            print("[Finalization] GradScaler released")
                    except Exception:
                        pass
            else:
                if master_process:
                    print("[Finalization] Keeping GradScaler for post-finalize finetune")
            
            # 3. Release teacher model if using distillation
            _raw = model.module if hasattr(model, 'module') else model
            if args.distill_model and hasattr(_raw, 'teacher') and _raw.teacher is not None:
                try:
                    # Move teacher to CPU first to free GPU memory
                    _raw.teacher.cpu()
                    # Delete teacher model reference
                    del _raw.teacher
                    _raw.teacher = None
                    if master_process:
                        print("[Finalization] Teacher model released")
                except Exception as e:
                    if master_process:
                        print(f"[Finalization] Warning: failed to release teacher: {e}")
            
            # 4. Clear CUDA cache and run garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 5. Synchronize all ranks before proceeding
            if ddp:
                try:
                    dist.barrier()
                except Exception:
                    pass
            
            if master_process:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[Finalization] GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            try:
                import torch.distributed as _dist
                dist_inited = (_dist.is_available() and _dist.is_initialized())
            except Exception:
                _dist = None
                dist_inited = False

            if master_process:
                print("[Training End] finalizing masks...")

            with torch.no_grad():
                raw_for_finalize = get_raw_model(model)
                if master_process:
                    print(f"[Finalization] raw_for_finalize type: {type(raw_for_finalize).__name__}")

                def _broadcast_like(t: torch.Tensor, src: int = 0) -> torch.Tensor:
                    if not dist_inited:
                        return t
                    device = t.device
                    dtype = t.dtype
                    if master_process:
                        ndim = torch.tensor([t.dim()], device=device, dtype=torch.int64)
                    else:
                        ndim = torch.zeros(1, device=device, dtype=torch.int64)
                    _dist.broadcast(ndim, src=src)
                    ndim_val = int(ndim.item())
                    if master_process:
                        shape_t = torch.tensor(list(t.shape), device=device, dtype=torch.int64)
                    else:
                        shape_t = torch.empty((ndim_val,), device=device, dtype=torch.int64)
                    _dist.broadcast(shape_t, src=src)
                    target_shape = tuple(int(x) for x in shape_t.tolist())
                    if (not master_process) and tuple(t.shape) != target_shape:
                        t = torch.empty(target_shape, device=device, dtype=dtype)
                    _dist.broadcast(t, src=src)
                    return t

                fsdp_full_ctx = nullcontext()
                if using_fsdp:
                    if FSDP is None:
                        raise RuntimeError("FSDP unavailable")
                    
                    # 清理 GPU 内存以避免 OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if master_process:
                            print("[Finalization] Cleared GPU cache before summon_full_params")
                    
                    # In distill mode, model is Distill_Model and .student is the FSDP-wrapped module
                    # We need to summon params on the FSDP-wrapped module, not the container
                    _raw = model.module if hasattr(model, 'module') else model
                    fsdp_target = _raw.student if args.distill_model else model
                    if master_process:
                        print(f"[Finalization] FSDP summon target type: {type(fsdp_target).__name__}")
                    
                    try:
                        import warnings
                        warnings.filterwarnings('ignore', category=UserWarning)
                        fsdp_full_ctx = FSDP.summon_full_params(
                            fsdp_target, 
                            writeback=True,  # 需要写回以更新权重
                            recurse=True,
                            offload_to_cpu=False  # 不要 offload，否则会触发更多通信
                        )
                        if master_process:
                            print("[Finalization] FSDP.summon_full_params context created")
                    except Exception as e:
                        if master_process:
                            print(f"[Finalization] WARNING: summon_full_params failed: {e}")
                            print("[Finalization] Trying alternative approach...")
                        # 回退方案：不使用 summon_full_params，直接在 FSDP 状态下操作
                        fsdp_full_ctx = nullcontext()

                with fsdp_full_ctx:
                    # IMPORTANT: _hard_mask_from_soft MUST be called inside summon_full_params context
                    # because in FSDP, module.mask is sharded (1D) outside the context.
                    # Only inside summon_full_params do we have the full 2D mask shape.
                    if master_process:
                        print("[Finalization] Computing hard masks inside summon_full_params context...")
                    
                    masks = []
                    frozen_flags = []
                    for name, module in raw_for_finalize.named_modules():
                        if isinstance(module, SparseLinear):
                            try:
                                hard = module._hard_mask_from_soft(module.mask).detach()
                                masks.append(hard)
                                frozen_flags.append(hard.to(dtype=torch.uint8))
                            except Exception as e:
                                if master_process:
                                    print(f"[Finalization] WARNING: _hard_mask_from_soft failed for {name}: {e}")
                                    print(f"[Finalization] mask shape: {module.mask.shape}, dim: {module.mask.dim()}")
                                # Fallback: use thresholding for sharded masks
                                hard = (module.mask > 0.5).to(dtype=module.mask.dtype).detach()
                                masks.append(hard)
                                frozen_flags.append(hard.to(dtype=torch.uint8))

                    if master_process:
                        print(f"[Finalization] Computed {len(masks)} hard masks")

                    # Broadcast masks from rank 0 to all ranks (ensure consistency)
                    if dist_inited:
                        src = 0
                        for i, (m, ff) in enumerate(zip(masks, frozen_flags)):
                            m_contig = _broadcast_like(m.contiguous(), src=src)
                            ff_contig = _broadcast_like(ff.contiguous(), src=src)
                            masks[i] = m_contig
                            frozen_flags[i] = ff_contig

                    # Apply masks to modules
                    raw_apply = raw_for_finalize
                    if master_process:
                        print(f"[Finalization] raw_apply type: {type(raw_apply).__name__}")
                        apply_sparse_count = sum(1 for _, m in raw_apply.named_modules() if isinstance(m, SparseLinear))
                        print(f"[Finalization] masks collected: {len(masks)}, SparseLinear in raw_apply: {apply_sparse_count}")
                        if len(masks) != apply_sparse_count:
                            print(f"[Finalization] WARNING: mask count mismatch! This may cause incorrect finalization.")

                    idx = 0
                    for name, module in raw_apply.named_modules():
                        if isinstance(module, SparseLinear):
                            try:
                                hard = masks[idx]
                                ff = frozen_flags[idx].to(dtype=module.frozen_mask_flags.dtype)
                                module.mask.copy_(hard)
                                module.frozen_mask_flags.data = torch.maximum(module.frozen_mask_flags.data, ff)
                                module.weight.mul_(module.mask)
                                module.change_mask = False
                                module._hardening_finalized = True
                                idx += 1
                            except Exception as e:
                                if master_process:
                                    print(f"[Finalization] WARNING: failed to finalize module {name}: {e}")
                                continue
                    
                    # 定期清理内存
                    if idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if master_process:
                print(f"[Training End] finalized {idx} SparseLinear modules")

            finalization_done = True
            finalization_iter = iter_num
            
            # ========== VERIFY MASK AFTER FINALIZATION ==========
            # Check that all masks are binary (0/1) and have correct sparsity
            if master_process:
                print("[Finalization] Verifying masks after hardening...")
                # Use raw_for_finalize for verification to ensure consistency
                verify_model = raw_for_finalize
                total_modules = 0
                all_binary = True
                sparsity_list = []
                non_binary_modules = []
                
                for name, module in verify_model.named_modules():
                    if hasattr(module, 'mask') and module.mask is not None:
                        mask = module.mask.detach().float()
                        total_modules += 1
                        
                        # Check binary: all values should be 0 or 1
                        unique_vals = torch.unique(mask)
                        is_binary = len(unique_vals) <= 2 and all(v in [0.0, 1.0] for v in unique_vals.tolist())
                        
                        if not is_binary:
                            all_binary = False
                            non_binary_modules.append((name, unique_vals[:10].tolist()))  # First 10 unique values
                        
                        # Calculate sparsity (fraction of zeros)
                        sparsity = 1.0 - mask.mean().item()
                        sparsity_list.append(sparsity)
                
                if total_modules > 0:
                    avg_sparsity = sum(sparsity_list) / len(sparsity_list)
                    min_sparsity = min(sparsity_list)
                    max_sparsity = max(sparsity_list)
                    
                    print(f"[Finalization] Mask verification results:")
                    print(f"  - Total modules with mask: {total_modules}")
                    print(f"  - All masks binary (0/1): {all_binary}")
                    print(f"  - Sparsity: avg={avg_sparsity:.4f}, min={min_sparsity:.4f}, max={max_sparsity:.4f}")
                    
                    if not all_binary:
                        print(f"  - WARNING: {len(non_binary_modules)} modules have non-binary masks!")
                        for mod_name, vals in non_binary_modules[:5]:  # Show first 5
                            print(f"    - {mod_name}: unique values = {vals}")
                    
                    # Verify target sparsity
                    target_sparsity = getattr(args, 'target_sparsity', 0.5)
                    if abs(avg_sparsity - target_sparsity) > 0.05:
                        print(f"  - WARNING: avg sparsity {avg_sparsity:.4f} differs from target {target_sparsity:.4f}")
                    else:
                        print(f"  - Sparsity matches target ({target_sparsity:.4f}) ✓")
                else:
                    print("[Finalization] WARNING: No modules with mask found!")
            
            # ========== EVAL AFTER FINALIZATION ==========
            # Re-run eval to get accurate metrics for the finalized model
            # This is important because the last eval was done BEFORE the final training step
            # and finalization, so the PPL reported then doesn't match the finalized weights.
            if master_process:
                print("[Finalization] Running eval on finalized model...")
            
            try:
                # All ranks must participate in eval under FSDP
                all_rank_eval = bool(ddp and using_fsdp)
                if all_rank_eval:
                    _safe_barrier()
                
                # Define eval_model for this scope (in case it wasn't defined in training loop)
                if args.distill_model:
                    container = model.module if hasattr(model, "module") else model
                    eval_model = container.student
                else:
                    eval_model = model
                
                if all_rank_eval:
                    losses = estimate_loss()
                else:
                    if master_process:
                        losses = estimate_loss()
                    else:
                        losses = {'train': float('inf'), 'val': float('inf')}
                
                # WikiText PPL
                if bool(getattr(args, 'skip_wiki_ppl', True)):
                    wiki_ppl = float('inf')
                else:
                    try:
                        if all_rank_eval:
                            wiki_ppl = eval_ppl_distributed(
                                eval_model,
                                bs=2,
                                device=device,
                                block_size=block_size,
                                model_name_or_path=args.student_model,
                                ptdtype=ptdtype,
                            )
                        else:
                            if master_process:
                                wiki_ppl = eval_ppl(
                                    eval_model,
                                    bs=2,
                                    device=device,
                                    block_size=block_size,
                                    model_name_or_path=args.student_model,
                                )
                            else:
                                wiki_ppl = float('inf')
                    except Exception as e:
                        if master_process:
                            print(f"[Finalization] Wiki PPL eval failed: {e}")
                        wiki_ppl = float('inf')
                
                if master_process:
                    print(f"[Finalization] Finalized model metrics: train_loss={losses.get('train', 0):.4f} val_loss={losses.get('val', 0):.4f} wiki_ppl={wiki_ppl:.4f}")
            except Exception as e:
                if master_process:
                    print(f"[Finalization] Warning: eval after finalization failed: {e}")
                # Keep the old values as fallback
            
            # ========== SAVE FINALIZED CHECKPOINT FOR RETRAIN (best-only) ==========
            # Only save if finalized model achieves lower wiki_ppl than previous best.
            # Fallback: if wiki_ppl is unavailable (skip_wiki_ppl=True), use val_loss.
            _finalize_wiki_ppl = float(wiki_ppl) if math.isfinite(float(wiki_ppl)) else None
            _finalize_val_loss = float(losses.get('val', float('inf')))

            if _finalize_wiki_ppl is not None:
                # 优先用 wiki_ppl 判断
                _should_save_finalized = (_finalize_wiki_ppl < best_wiki_ppl)
                _save_reason = f"wiki_ppl {_finalize_wiki_ppl:.4f} < best {best_wiki_ppl:.4f}"
                _skip_reason = f"wiki_ppl {_finalize_wiki_ppl:.4f} >= best {best_wiki_ppl:.4f}"
            else:
                # fallback: val_loss
                _should_save_finalized = (_finalize_val_loss < best_val_loss)
                _save_reason = f"val_loss {_finalize_val_loss:.4f} < best {best_val_loss:.4f} (wiki_ppl unavailable)"
                _skip_reason = f"val_loss {_finalize_val_loss:.4f} >= best {best_val_loss:.4f} (wiki_ppl unavailable)"

            if master_process:
                if _should_save_finalized:
                    print(f"[Finalization] Save condition MET: {_save_reason}")
                else:
                    print(f"[Finalization] Save condition NOT met: {_skip_reason}, skipping save.")

            # ========== FSDP state_dict() 是集合操作，所有 rank 必须执行 ==========
            # 无论是否保存，都要执行 state_dict() 以避免 NCCL 超时
            model_state_dict = None
            try:
                if using_fsdp:
                    if StateDictType is None or FullStateDictConfig is None or FSDP is None:
                        raise RuntimeError("FSDP state_dict utilities unavailable")
                    fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    # In distill mode, `model` is Distill_Model and .student is the FSDP-wrapped student
                    _raw = model.module if hasattr(model, 'module') else model
                    save_target = _raw.student if args.distill_model else model
                    if master_process:
                        print(f"[Finalization] Gathering state_dict (save_target type: {type(save_target).__name__})...")
                    with FSDP.state_dict_type(save_target, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                        model_state_dict = save_target.state_dict()
                else:
                    # Non-FSDP mode: use raw_model or raw_model.student
                    save_model = raw_model.student if args.distill_model else raw_model
                    if master_process:
                        print(f"[Finalization] save_model type: {type(save_model).__name__}")
                    model_state_dict = save_model.state_dict()
            except Exception as e:
                if master_process:
                    print(f"[Finalization] Warning: Failed to gather state_dict: {e}")
                model_state_dict = None

            if _should_save_finalized:
                # 更新 best 记录
                if _finalize_wiki_ppl is not None:
                    best_wiki_ppl = _finalize_wiki_ppl
                best_val_loss = _finalize_val_loss

                if master_process:
                    print("[Finalization] Saving finalized checkpoint for retrain...")

                if master_process and model_state_dict is not None:
                    try:
                        finalized_checkpoint = {
                            'model_state_dict': model_state_dict,
                            'iter_num': iter_num,
                            'args': vars(args),
                            'finalization_done': True,
                            'finalization_iter': finalization_iter,
                        }
                        
                        # 保存 channel pruning mask 状态（如果启用）
                        # 重要：Finalization 后的 channel_masks 用于评估时恢复 MLP forward patch
                        if channel_pruning_enabled and channel_mask_state is not None:
                            finalized_checkpoint['channel_masks'] = {
                                i: {
                                    "mask": channel_mask_state.masks[i].mask.cpu(),
                                    "temperature": channel_mask_state.masks[i].temperature,
                                    "hardening_x": channel_mask_state.masks[i].hardening_x,
                                }
                                for i in range(channel_mask_state.num_layers)
                            }
                            print(f"[Finalization] Included channel_masks ({channel_mask_state.num_layers} layers)")

                        # Save to target_dir/model.pt (main checkpoint)
                        finalized_path = os.path.join(target_dir, "model.pt")
                        torch.save(finalized_checkpoint, finalized_path)
                        print(f"[Finalization] Saved finalized checkpoint to {finalized_path}")

                        # Also save eval.json with final metrics
                        try:
                            eval_data = {
                                'iter': iter_num,
                                'train_loss': float(losses.get('train', 0)),
                                'val_loss': float(losses.get('val', 0)),
                                'wiki_ppl': float(wiki_ppl) if wiki_ppl != float('inf') else None,
                                'finalization_done': True,
                            }
                            with open(os.path.join(target_dir, 'eval.json'), 'w') as f:
                                json.dump(eval_data, f, indent=2)
                        except Exception as e:
                            print(f"[Finalization] Warning: failed to save eval.json: {e}")

                        # Clean up
                        del finalized_checkpoint
                    except Exception as e:
                        print(f"[Finalization] Warning: failed to save finalized checkpoint: {e}")

            # Release model_state_dict memory (outside of _should_save_finalized block)
            if model_state_dict is not None:
                del model_state_dict

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Synchronize after checkpoint save
            if ddp:
                try:
                    dist.barrier()
                except Exception:
                    pass
            
            # ========== Channel Pruning: 导出物理裁剪后的模型 ==========
            if channel_pruning_enabled and channel_mask_state is not None:
                if master_process:
                    print("[Channel Pruning] 导出裁剪后的模型...")
                try:
                    # 移除 PCA hooks（导出前必须清理）
                    if channel_score_computer is not None:
                        channel_score_computer.remove_pca_hooks()
                    
                    # 恢复原始 MLP forward（导出前必须恢复）
                    if channel_original_forwards is not None:
                        restore_model_mlp_forward(
                            train_student,
                            channel_original_forwards,
                            channel_pruning_config
                        )
                    
                    from contextlib import nullcontext
                    full_param_ctx = (
                        FSDP.summon_full_params(
                            train_student,
                            writeback=False,
                            rank0_only=True
                        )
                        if (FSDP is not None and isinstance(train_student, FSDP))
                        else nullcontext()
                    )
                    with full_param_ctx:
                        # 计算参数减少比例
                        if master_process:
                            reduction_stats = compute_param_reduction(
                                train_student, channel_mask_state, channel_pruning_config
                            )
                            print(f"[Channel Pruning] FFN 参数减少: {reduction_stats['ffn_param_reduction']*100:.1f}%")
                            print(f"[Channel Pruning] FFN 剩余比例: {reduction_stats['ffn_remaining_ratio']*100:.1f}%")
                        
                        # 导出路径
                        ch_export_path = args.channel_export_path
                        if ch_export_path is None:
                            ch_export_path = os.path.join(target_dir, "channel_pruned_model")
                        
                        if master_process:
                            export_pruned_model(
                                train_student,
                                channel_mask_state,
                                channel_pruning_config,
                                ch_export_path
                            )
                            print(f"[Channel Pruning] 裁剪模型已保存到 {ch_export_path}")
                            
                            # 保存 channel mask 状态
                            ch_mask_save = {
                                i: {
                                    "mask": channel_mask_state.masks[i].mask.cpu(),
                                    "temperature": channel_mask_state.masks[i].temperature,
                                    "hardening_x": channel_mask_state.masks[i].hardening_x,
                                }
                                for i in range(channel_mask_state.num_layers)
                            }
                            torch.save(ch_mask_save, os.path.join(target_dir, "channel_masks.pt"))
                            print(f"[Channel Pruning] Channel masks 已保存到 {target_dir}/channel_masks.pt")
                    
                except Exception as e:
                    if master_process:
                        print(f"[Channel Pruning] 导出失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            extra = int(args.final_finetune_iters)
            if extra > 0:
                new_max = iter_num + extra
                max_iters = new_max
                if master_process:
                    try:
                        pbar.total = new_max
                        pbar.set_description("Finetuning (post-finalize)")
                        pbar.refresh()
                    except Exception:
                        pass
                    print(f"[Finalization] continuing training for {extra} more iters until {new_max}")
                continue
            else:
                if master_process:
                    print(args)
                break
        else:
            if master_process:
                pbar.close()
                print(args)
            break

# 停止数据预取器
if train_prefetcher is not None:
    train_prefetcher.stop()
    if master_process:
        print("[Prefetcher] Stopped.")

if ddp:
    destroy_process_group()
