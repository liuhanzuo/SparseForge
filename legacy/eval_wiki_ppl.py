"""
独立的 Wiki PPL 评估脚本
用于评估 checkpoint 的 WikiText-2 PPL，使用和 train_universal 一样的评估方法。
同时支持 lm_eval harness 评估（BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA）。

使用方法：
    # 评估单个 checkpoint (仅 Wiki PPL) - 单卡
    python eval_wiki_ppl.py --ckpt_path model.pt --model_path <base_model_path>
    
    # 对比两个 checkpoint
    python eval_wiki_ppl.py --ckpt_path model.pt --ckpt_path2 retrain_best.pt --model_path <base_model_path>
    
    # 同时运行 lm_eval benchmarks
    python eval_wiki_ppl.py --ckpt_path model.pt --model_path <base_model_path> --run_lm_eval
    
    # 指定特定的 lm_eval tasks
    python eval_wiki_ppl.py --ckpt_path model.pt --model_path <base_model_path> --run_lm_eval --lm_eval_tasks boolq,hellaswag,arc_easy
    
    # 评估原始模型 (不需要 checkpoint)
    python eval_wiki_ppl.py --eval_base_model --model_path models/facebook--opt-2.7b --run_lm_eval
    
    # 对比原始模型和 checkpoint
    python eval_wiki_ppl.py --eval_base_model --ckpt_path model.pt --model_path models/facebook--opt-2.7b --run_lm_eval
    
    # 使用 HuggingFace token (避免速率限制)
    python eval_wiki_ppl.py --eval_base_model --model_path models/facebook--opt-2.7b --run_lm_eval --hf_token YOUR_HF_TOKEN
    
    # ================== 多卡并行评估 (加速 7B+ 大模型) ==================
    # 使用 torchrun 启动多卡评估 (推荐用于 7B+ 模型)
    torchrun --nproc_per_node=4 eval_wiki_ppl.py --eval_base_model --model_path Llama-2-7b-hf/ --block_size 4096
    
    # 多卡评估 checkpoint
    torchrun --nproc_per_node=8 eval_wiki_ppl.py --ckpt_path retrain_best.pt --model_path Llama-2-7b-hf/ --block_size 4096
    
    # 多卡评估 + lm_eval (lm_eval 只在 rank 0 运行)
    torchrun --nproc_per_node=4 eval_wiki_ppl.py --eval_base_model --model_path Llama-2-7b-hf/ --run_lm_eval --lm_eval_batch_size 32
    
    # 也可以用 --distributed 显式启用分布式模式 (自动检测 torchrun 环境)
    torchrun --nproc_per_node=4 eval_wiki_ppl.py --distributed --eval_base_model --model_path Llama-2-7b-hf/
"""
import argparse
import gc
import os
import sys

# Ensure parent directory is on sys.path so that `sparseforge` and `channel_pruning` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import eval_ppl, eval_ppl_distributed, get_raw_model
from sparse_modeling import SparseLinearConfig
from model_factory import detect_model_type, SUPPORTED_MODEL_TYPES
from channel_pruning import (
    ChannelPruningConfig, ChannelMaskState, patch_model_mlp_forward
)

# 分布式相关
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def print_model_parameters(model: torch.nn.Module, prefix: str = ""):
    """输出模型的参数量统计信息
    
    Args:
        model: 要统计的模型
        prefix: 输出前缀（用于区分不同模型）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 格式化为可读的数字 (B/M/K)
    def fmt(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return str(n)
    
    tag = f" ({prefix})" if prefix else ""
    print(f"[MODEL PARAMS]{tag} Total: {fmt(total_params)} ({total_params:,})")
    print(f"[MODEL PARAMS]{tag} Trainable: {fmt(trainable_params)} ({trainable_params:,})")
    print(f"[MODEL PARAMS]{tag} Non-trainable: {fmt(non_trainable_params)} ({non_trainable_params:,})")
    
    # 区分 mask buffer 和实际参数
    from sparse_modeling import SparseLinear
    mask_params = 0
    slorb_params = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            if hasattr(module, 'mask') and module.mask is not None:
                mask_params += module.mask.numel()
            if hasattr(module, 'SLoRB_Weight'):
                slorb_params += module.SLoRB_Weight.numel()
            if hasattr(module, 'x_proj'):
                slorb_params += module.x_proj.numel()
    if mask_params > 0 or slorb_params > 0:
        effective = total_params - mask_params
        print(f"[MODEL PARAMS]{tag} Effective (excl. mask buffers): {fmt(effective)} ({effective:,})")
        if slorb_params > 0:
            print(f"[MODEL PARAMS]{tag} SLoRB params: {fmt(slorb_params)} ({slorb_params:,})")


def setup_distributed():
    """初始化分布式环境"""
    if not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # 由 torchrun 启动
            dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            return rank, world_size, local_rank
    elif dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
        return rank, world_size, local_rank
    return 0, 1, 0

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_rank():
    """检查是否是主进程"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def detect_model_type_from_path(model_path: str) -> str:
    """从模型路径推断模型类型，使用 model_factory 的统一检测逻辑"""
    return detect_model_type(model_path)

def run_lm_eval_benchmarks(
    model: torch.nn.Module,
    model_path: str,
    device: str = "cuda:0",
    tasks: Optional[List[str]] = None,
    batch_size: int = 4,
    num_fewshot: int = 0,
    hf_token: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """
    运行 lm_eval harness 评估
    
    Args:
        model: 要评估的模型
        model_path: 基础模型路径 (用于获取 tokenizer)
        device: 设备
        tasks: 要评估的任务列表，默认为所有标准 benchmarks
        batch_size: 批大小
        num_fewshot: few-shot 样本数
        hf_token: HuggingFace token (避免速率限制)
    
    Returns:
        任务名到准确率的字典
    """
    # 默认任务列表
    if tasks is None:
        tasks = [
            "boolq",           # BoolQ
            "rte",             # RTE  
            "hellaswag",       # HellaSwag
            "winogrande",      # WinoGrande
            "arc_easy",        # ARC-e
            "arc_challenge",   # ARC-c
            "openbookqa",      # OBQA
        ]
    
    print(f"[lm_eval] Running benchmarks: {tasks}")
    print(f"[lm_eval] batch_size={batch_size}, num_fewshot={num_fewshot}")
    
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoTokenizer
        import os
        
        # 设置环境变量以启用 tqdm 进度条
        os.environ["TQDM_DISABLE"] = "0"
        
        # 设置 HuggingFace token (避免速率限制)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print(f"[lm_eval] Using HuggingFace token for authentication")
        
        # 获取 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 确保模型有 device 属性 (lm_eval HFLM 需要)
        if not hasattr(model, 'device'):
            # 从模型参数推断 device
            try:
                model.device = next(model.parameters()).device
            except StopIteration:
                model.device = torch.device(device)
        
        # 诊断：检查 SparseLinear 的模式
        from sparse_modeling import SparseLinear
        sparse_modes = {}
        for name, module in model.named_modules():
            if isinstance(module, SparseLinear):
                mode = getattr(module, 'mode', 'unknown')
                sparse_modes[mode] = sparse_modes.get(mode, 0) + 1
        if sparse_modes:
            print(f"[lm_eval] SparseLinear modes: {sparse_modes}")
        
        # 重要优化：对于 OPTSparse/LlamaSparse，直接使用内部的 HuggingFace 模型
        # 这样可以避免额外的包装层开销，并且 lm_eval 可以直接使用原生接口
        # 注意：不能对原生 HuggingFace CausalLM 模型提取内部 model，因为内部 model (如 LlamaModel)
        # 没有 lm_head，其 forward 返回 BaseModelOutputWithPast 而不是 CausalLMOutput
        eval_model = model
        
        # 检查是否是我们自定义的 Sparse 包装类（有 lm_head 在内部 model 中）
        # 注意：GPT (nanoGPT 风格) 是自定义实现，不是 HuggingFace 包装类，需要单独处理
        is_hf_sparse_wrapper = hasattr(model, '__class__') and model.__class__.__name__ in [
            'OPTSparse', 'LlamaSparse', 'QwenSparse', 'MistralSparse', 'DeepSeekMoESparse'
        ]
        # GPT2 模型在本项目中的类名是 'GPT' (来自 model.py 的 nanoGPT 实现)
        is_gpt2_sparse = hasattr(model, '__class__') and model.__class__.__name__ == 'GPT'
        
        if is_hf_sparse_wrapper and hasattr(model, 'model') and hasattr(model.model, 'forward'):
            # 对于基于 HuggingFace 的 Sparse 包装类，可以使用内部的 HuggingFace 模型
            inner_model = model.model
            if hasattr(inner_model, 'config') and hasattr(inner_model, 'forward'):
                print(f"[lm_eval] Using inner HuggingFace model for faster evaluation")
                eval_model = inner_model
                # 确保内部模型也在正确的设备上
                if not hasattr(eval_model, 'device'):
                    eval_model.device = next(eval_model.parameters()).device
        elif is_gpt2_sparse:
            # GPT (nanoGPT) 是自定义实现，返回 (logits, loss)
            # 需要包装 forward 输出
            print(f"[lm_eval] GPT (nanoGPT) detected, will wrap forward output")
        else:
            # 对于原生 HuggingFace CausalLM 模型，直接使用，不提取内部 model
            print(f"[lm_eval] Using model directly (native HuggingFace CausalLM)")
        
        # 确保模型有 tie_weights 方法 (lm_eval HFLM 需要)
        if not hasattr(eval_model, 'tie_weights'):
            eval_model.tie_weights = lambda: None
        
        # 检查是否需要包装 forward 输出
        # GPT2Sparse 返回 (logits, loss) tuple，需要包装
        # 其他自定义 Sparse 模型如果没有使用内部 HF 模型，也可能需要包装
        needs_wrapper = False
        if is_gpt2_sparse:
            # GPT2Sparse 总是需要包装（返回 tuple）
            needs_wrapper = True
        elif eval_model is model:  # 如果没有使用内部模型
            # 检查是否是其他自定义模型（可能返回 tuple）
            if hasattr(model, '__class__') and model.__class__.__name__ in ['OPTSparse', 'LlamaSparse']:
                needs_wrapper = True
        
        if needs_wrapper:
            original_forward = eval_model.forward
            # 确定 autocast dtype
            autocast_dtype_wrapper = dtype if dtype is not None else torch.bfloat16
            
            class ModelOutputWrapper:
                """包装模型输出，确保有 logits 属性"""
                def __init__(self, logits):
                    self.logits = logits
            
            def wrapped_forward(*args, **kwargs):
                # 尝试禁用 KV cache，避免 DynamicCache 兼容性问题 (如 DeepSeek-MoE)
                # 注意：某些自定义模型的 forward 不接受 use_cache 参数
                # 使用 autocast 确保 dtype 一致性
                try:
                    kwargs['use_cache'] = False
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype_wrapper):
                        output = original_forward(*args, **kwargs)
                except TypeError as te:
                    if 'use_cache' in str(te):
                        # 模型不支持 use_cache 参数，移除后重试
                        kwargs.pop('use_cache', None)
                        with torch.autocast(device_type='cuda', dtype=autocast_dtype_wrapper):
                            output = original_forward(*args, **kwargs)
                    else:
                        raise te
                # 如果输出是 tuple，取第一个元素作为 logits
                if isinstance(output, tuple):
                    return ModelOutputWrapper(output[0])
                # 如果已经有 logits 属性，直接返回
                elif hasattr(output, 'logits'):
                    return output
                # 否则假设输出本身就是 logits
                else:
                    return ModelOutputWrapper(output)
            
            eval_model.forward = wrapped_forward
        else:
            # 即使不需要 wrapper，也需要禁用 use_cache 以避免 DynamicCache 兼容性问题
            # 为 DeepSeek-MoE 等使用自定义 modeling 文件的模型
            original_forward = eval_model.forward
            # 确定 autocast dtype
            autocast_dtype = dtype if dtype is not None else torch.bfloat16
            def no_cache_forward(*args, **kwargs):
                # 注意：某些自定义模型的 forward 不接受 use_cache 参数
                # 使用 autocast 确保 dtype 一致性
                try:
                    kwargs['use_cache'] = False
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        return original_forward(*args, **kwargs)
                except TypeError as te:
                    if 'use_cache' in str(te):
                        # 模型不支持 use_cache 参数，移除后重试
                        kwargs.pop('use_cache', None)
                        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                            return original_forward(*args, **kwargs)
                    else:
                        raise te
            eval_model.forward = no_cache_forward
        
        # 创建 HFLM wrapper
        lm = HFLM(
            pretrained=eval_model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=str(device),
        )
        
        # 运行评估 (verbosity="INFO" 显示进度)
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            verbosity="INFO",  # 显示进度信息
        )
        
        # 提取结果
        # 对于 HellaSwag 和 ARC-Challenge，标准 metric 是 acc_norm（长度归一化准确率）
        # 这与 CAST 等论文的评测标准一致
        _PREFER_ACC_NORM = {"hellaswag", "arc_challenge"}

        def _get_task_acc(task_name, task_result):
            """获取任务的准确率指标，HellaSwag/ARC-c 优先使用 acc_norm"""
            if task_name in _PREFER_ACC_NORM:
                return task_result.get("acc_norm,none",
                    task_result.get("acc_norm",
                    task_result.get("acc,none",
                    task_result.get("acc", None))))
            return task_result.get("acc,none",
                task_result.get("acc",
                task_result.get("acc_norm,none",
                task_result.get("acc_norm", None))))

        lm_eval_results = {}
        for task_name in tasks:
            if task_name in results.get("results", {}):
                task_result = results["results"][task_name]
                acc = _get_task_acc(task_name, task_result)
                if acc is not None:
                    acc_pct = acc * 100 if acc <= 1 else acc
                    lm_eval_results[task_name] = acc_pct
                    metric_used = "acc_norm" if task_name in _PREFER_ACC_NORM else "acc"
                    print(f"  - {task_name}: {acc_pct:.2f}% ({metric_used})")
                else:
                    print(f"  - {task_name}: (no accuracy metric found)")
        
        # 计算平均值
        if lm_eval_results:
            mean_acc = sum(lm_eval_results.values()) / len(lm_eval_results)
            lm_eval_results["mean"] = mean_acc
            print(f"  - Mean: {mean_acc:.2f}%")
        
        return lm_eval_results
        
    except ImportError as e:
        print(f"[lm_eval] WARNING: lm_eval not available: {e}")
        print("[lm_eval] Please install lm_eval: pip install lm_eval")
        return {}
    except Exception as e:
        print(f"[lm_eval] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}

def load_base_model(model_path: str, device: str = "cuda:0"):
    """
    加载原始 HuggingFace 模型 (不带任何 checkpoint)
    
    Args:
        model_path: 模型路径
        device: 设备
    
    Returns:
        加载好的模型
    """
    print(f"[INFO] Loading base model from: {model_path}")
    
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Base model loaded successfully")
    
    return model

def load_base_model_distributed(model_path: str, local_rank: int = 0):
    """
    加载原始 HuggingFace 模型用于分布式评估
    
    Args:
        model_path: 模型路径
        local_rank: 本地 rank
    
    Returns:
        加载好的模型
    """
    if is_main_rank():
        print(f"[INFO] Loading base model from: {model_path}")
    
    from transformers import AutoModelForCausalLM
    
    device = f"cuda:{local_rank}"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    model = model.to(device)
    model.eval()
    
    if is_main_rank():
        print(f"[INFO] Base model loaded successfully on device {device}")
    
    return model


def _restore_channel_masks(
    model: torch.nn.Module,
    channel_masks_data: dict,
    model_type: str,
    args_dict: dict,
    verbose: bool = True
) -> torch.nn.Module:
    """从 checkpoint 中恢复 channel pruning 的 MLP forward patch。
    
    训练时 channel pruning 通过 patch_model_mlp_forward() 修改 MLP forward，
    在中间层激活上乘以 channel mask。评估时必须恢复这个 patch，否则被剪枝的
    通道仍然会产出垃圾激活值，导致 PPL 飙升。
    
    Args:
        model: 已加载权重的模型
        channel_masks_data: checkpoint 中的 channel_masks 字典
        model_type: 模型类型 (gpt2, llama, opt, qwen, ...)
        args_dict: checkpoint 中保存的训练参数
        verbose: 是否打印详细信息
        
    Returns:
        patch 后的模型
    """
    try:
        # 从 args 获取 channel pruning 配置
        ffn_keep_ratio = args_dict.get('ffn_keep_ratio', 0.5)
        
        # 创建 ChannelPruningConfig
        ch_config = ChannelPruningConfig(
            model_type=model_type,
            ffn_keep_ratio=ffn_keep_ratio,
            importance_metric=args_dict.get('channel_importance_metric', 'hessian_obd'),
            mask_update_period=args_dict.get('channel_mask_update_period', 50),
            mask_lr=args_dict.get('channel_mask_lr', 0.1),
            sparsity_warmup_steps=args_dict.get('channel_sparsity_warmup_steps', 500),
            hardening_start_step=args_dict.get('channel_hardening_start_step', 0),
            hardening_duration=args_dict.get('channel_hardening_duration', 5000),
            temp_init=args_dict.get('channel_temp_init', 1.0),
            temp_min=args_dict.get('channel_temp_min', 0.05),
            temp_decay=args_dict.get('channel_temp_decay', 0.97),
        )
        
        # 创建 ChannelMaskState（注意：此时模型可能还在 CPU 上，
        # 后面 model.to(device) 后 mask 不会自动迁移，所以在 masked_forward 中
        # 需要确保 mask 与 hidden_states 在同一设备）
        model_device = next(model.parameters()).device
        ch_mask_state = ChannelMaskState(model, ch_config, device=model_device)
        
        # 从 checkpoint 恢复 mask 值
        num_restored = 0
        for layer_idx_str, mask_data in channel_masks_data.items():
            layer_idx = int(layer_idx_str)
            if layer_idx < ch_mask_state.num_layers:
                saved_mask = mask_data['mask']
                ch_mask_state.masks[layer_idx].mask = saved_mask.to(ch_mask_state.device)
                ch_mask_state.masks[layer_idx].temperature = mask_data.get('temperature', 0.05)
                # 评估时强制使用 hard mask (hardening_x=0)
                ch_mask_state.masks[layer_idx].hardening_x = 0.0
                num_restored += 1
        
        # 预先 finalize masks 为 binary，这样 get_effective_mask 直接返回 hard mask
        # 避免每次 forward 都重新计算 hard mask
        ch_mask_state.finalize_masks()
        
        if verbose:
            print(f"[Channel Pruning] Restored {num_restored} layer masks from checkpoint")
            # 打印 mask 稀疏度统计
            for i in range(min(3, ch_mask_state.num_layers)):
                mask = ch_mask_state.masks[i].mask
                kept = mask.sum().item()
                total = mask.numel()
                print(f"[Channel Pruning]   Layer {i}: {int(kept)}/{total} channels kept ({kept/total*100:.1f}%)")
            if ch_mask_state.num_layers > 3:
                print(f"[Channel Pruning]   ... ({ch_mask_state.num_layers - 3} more layers)")
        
        # Patch MLP forward 以应用 channel mask
        patch_model_mlp_forward(model, ch_mask_state, ch_config)  # 返回 (forwards, lora_modules)，eval 时不需要
        
        if verbose:
            print(f"[Channel Pruning] MLP forward patched with channel masks (hardening_x=0, hard mask)")
        
    except Exception as e:
        print(f"[WARN] Failed to restore channel masks: {e}")
        import traceback
        traceback.print_exc()
    
    return model


def optimize_sparse_model_for_inference(model: torch.nn.Module, use_slorb: bool = False) -> torch.nn.Module:
    """
    优化稀疏模型的推理速度
    
    问题：SparseLinear 在 sparse_forward 模式下，每次 forward 都要做 weight * mask 运算，
    即使 mask 已经固定不变，这会带来大量额外开销。
    
    解决方案：将 硬掩码(hard mask) 直接应用到权重上（weight = weight * hard_mask），
    然后切换到 dense_forward 模式，避免重复计算。
    
    重要：
    1. 必须使用硬掩码（0或1）而不是软掩码（0-1连续值），否则会导致权重被错误缩放！
    2. 如果启用了 SLoRB，不能切换到 dense_forward 模式，因为 SLoRB 只在 sparse_forward 中生效！
    
    Args:
        model: 稀疏模型
        use_slorb: 是否启用了 SLoRB（低秩恢复）
        
    Returns:
        优化后的模型
    """
    from sparse_modeling import SparseLinear
    
    optimized_count = 0
    total_sparse_layers = 0
    sample_sparsity = None
    
    # 如果启用了 SLoRB，不能切换到 dense_forward，因为 SLoRB 只在 sparse_forward 中应用
    can_use_dense_forward = not use_slorb
    
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear):
            total_sparse_layers += 1
            
            # 检查是否有 mask
            if hasattr(module, 'mask') and module.mask is not None:
                with torch.no_grad():
                    # 重要：必须使用硬掩码（binary mask），而不是软掩码！
                    # 软掩码是 0-1 之间的连续值，如果直接乘到权重上会错误缩放权重
                    # 硬掩码是 0 或 1 的二值掩码，才是正确的稀疏化方式
                    
                    # 检查 mask 是否已经是硬掩码（全是 0 或 1）
                    soft_mask = module.mask
                    is_already_binary = torch.all((soft_mask == 0) | (soft_mask == 1)).item()
                    
                    if is_already_binary:
                        # 已经是硬掩码，直接使用
                        hard_mask = soft_mask.to(dtype=module.weight.dtype)
                    else:
                        # 需要转换为硬掩码
                        # 使用 _hard_mask_from_soft 方法（与 forward 中 hardening_x=0 时一致）
                        if hasattr(module, '_hard_mask_from_soft'):
                            hard_mask = module._hard_mask_from_soft(soft_mask).to(dtype=module.weight.dtype)
                        else:
                            # fallback: 简单阈值化
                            hard_mask = (soft_mask > 0.5).to(dtype=module.weight.dtype)
                    
                    # 如果权重已经被硬掩码应用过（finalized checkpoint），避免重复乘导致二次稀疏
                    already_masked = False
                    if can_use_dense_forward and getattr(module, "_hardening_finalized", False):
                        weight_zero_ratio = (module.weight.data == 0).float().mean().item()
                        mask_zero_ratio = (hard_mask == 0).float().mean().item()
                        if weight_zero_ratio >= mask_zero_ratio - 1e-4:
                            already_masked = True
                            module.mode = "dense_forward"
                            module._inference_optimized = True
                            if sample_sparsity is None:
                                sample_sparsity = weight_zero_ratio
                    
                    if not already_masked:
                        if can_use_dense_forward:
                            # 没有 SLoRB: 将硬掩码应用到权重上，切换到 dense_forward 模式
                            module.weight.data.mul_(hard_mask)
                            module.mode = "dense_forward"
                        else:
                            # 有 SLoRB: 保持 sparse_forward 模式，但确保 mask 是硬掩码
                            # 将 mask 转换为硬掩码，这样 sparse_forward 时不需要每次做 hardening
                            module.mask.data.copy_(hard_mask)
                        
                        # 标记为已优化
                        module._inference_optimized = True
                        
                        # 记录一个样本的稀疏度
                        if sample_sparsity is None:
                            if can_use_dense_forward:
                                zeros = (module.weight.data == 0).sum().item()
                                total = module.weight.data.numel()
                            else:
                                zeros = (hard_mask == 0).sum().item()
                                total = hard_mask.numel()
                            sample_sparsity = zeros / total
                    
                optimized_count += 1
    
    if total_sparse_layers > 0:
        print(f"[INFO] Optimized {optimized_count}/{total_sparse_layers} SparseLinear layers for inference")
        if can_use_dense_forward:
            print(f"[INFO] Switched to dense_forward mode (hard mask pre-applied to weights)")
        else:
            print(f"[INFO] Keeping sparse_forward mode (SLoRB enabled, mask converted to hard mask)")
        if sample_sparsity is not None:
            print(f"[INFO] Sample layer sparsity: {sample_sparsity*100:.1f}% zeros")
    
    return model

def load_model_from_checkpoint(ckpt_path: str, model_path: str, device: str = "cuda:0"):
    """
    从 checkpoint 加载模型
    
    Args:
        ckpt_path: checkpoint 路径 (model.pt 或 retrain_best.pt)
        model_path: 基础模型路径 (用于初始化模型结构)
        device: 设备
    
    Returns:
        加载好的模型
    """
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 获取 state_dict - 支持多种格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # 可能整个 checkpoint 就是 state_dict
        state_dict = checkpoint
    
    # 获取训练参数
    args_dict = checkpoint.get('args', {})
    
    # 提取 channel pruning masks（如果有）
    channel_masks_data = checkpoint.get('channel_masks', None)
    if channel_masks_data is not None:
        print(f"[INFO] Found channel_masks in checkpoint ({len(channel_masks_data)} layers)")
    
    # 检查并移除 state_dict 中的 "student." 前缀
    # 这是因为训练时使用 Distill_Model 包装，key 会带有 student. 前缀
    # 但评估时创建的是裸模型，没有这个前缀
    if any(k.startswith('student.') for k in state_dict.keys()):
        print(f"[INFO] Detected 'student.' prefix in checkpoint keys, removing...")
        state_dict = {k.replace('student.', '', 1) if k.startswith('student.') else k: v 
                      for k, v in state_dict.items()}
    
    # 检查是否是 sparse 模型 - 检测 .mask 参数
    is_sparse = any('.mask' in k for k in state_dict.keys())
    
    print(f"[INFO] Detected sparse model: {is_sparse}")
    print(f"[INFO] State dict keys sample: {list(state_dict.keys())[:5]}")
    
    # 推断模型类型
    model_type = detect_model_type_from_path(model_path)
    print(f"[INFO] Detected model type: {model_type}")
    
    # 从 args 获取稀疏配置
    sparsity_ratio = args_dict.get('sparsity_ratio', 0.5)
    mask_type = args_dict.get('mask_type', 'unstructured')
    mask_metric = args_dict.get('mask_metric', 'hessian_obd')
    hard_mask_type = args_dict.get('hard_mask_type', 'topk')
    
    # 获取 SLoRB 配置 (低秩恢复)
    use_slorb = args_dict.get('SLoRB', False)
    slorb_k = args_dict.get('SLoRB_k', 64)
    slorb_init_type = args_dict.get('SLoRB_init_type', 'mean')
    
    print(f"[INFO] Sparsity config: ratio={sparsity_ratio}, mask_type={mask_type}, hard_mask_type={hard_mask_type}")
    if use_slorb:
        print(f"[INFO] SLoRB enabled: k={slorb_k}, init_type={slorb_init_type}")
    
    # 根据模型类型初始化
    if is_sparse:
        # 创建 SparseLinearConfig
        sparse_cfg = SparseLinearConfig(
            change_mask=False,  # 评估时不更新 mask
            mode="sparse_forward",
            mask_type=mask_type,
            mask_metric=mask_metric,
            sparsity_ratio=sparsity_ratio,
            hard_mask_type=hard_mask_type,
            # SLoRB 配置 (低秩恢复，推理时也需要)
            SLoRB=use_slorb,
            SLoRB_k=slorb_k,
            SLoRB_init_type=slorb_init_type,
        )
        
        if model_type == 'opt':
            from model_opt import OPTSparse
            model = OPTSparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'llama':
            from model_llama import LlamaSparse
            model = LlamaSparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'qwen':
            from model_qwen import QwenSparse
            model = QwenSparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'mistral':
            from model_mistral import MistralSparse
            model = MistralSparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'deepseek_moe':
            from model_deepseek_moe import DeepSeekMoESparse
            model = DeepSeekMoESparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'hunyuan':
            from model_hunyuan import HunyuanSparse
            model = HunyuanSparse.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        elif model_type == 'gpt2':
            from model import GPT
            model = GPT.from_pretrained(
                model_path,
                sparselinear_config=sparse_cfg,
                is_teacher=False,
            )
        else:
            raise ValueError(f"Unsupported sparse model type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}")
        
        # 过滤掉训练时特有的 keys，推理时不需要：
        # - hessian_diag: Hessian 对角估计，用于 mask 更新
        # 注意：如果启用了 SLoRB，则 SLoRB_Weight 和 x_proj 是推理时需要的！
        training_only_patterns = ['hessian_diag']
        if not use_slorb:
            # 只有在没有启用 SLoRB 时才过滤这些 keys
            training_only_patterns.extend(['SLoRB_Weight', 'x_proj'])
        keys_to_remove = [k for k in state_dict.keys() 
                          if any(pattern in k for pattern in training_only_patterns)]
        if keys_to_remove:
            print(f"[INFO] Filtering out {len(keys_to_remove)} training-only keys ({'/'.join(training_only_patterns)})")
            for k in keys_to_remove:
                del state_dict[k]
        
        # 加载权重
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        # missing keys 通常是 placeholder 参数（hessian_diag, frozen_mask_flags 等）
        # 由于 change_mask=False，模型用 placeholder 创建这些参数，这是预期行为
        if missing:
            # 区分真正缺失的 key 和 placeholder key
            placeholder_patterns = ['hessian_diag', 'frozen_mask_flags']
            real_missing = [k for k in missing if not any(p in k for p in placeholder_patterns)]
            placeholder_missing = len(missing) - len(real_missing)
            if placeholder_missing > 0:
                print(f"[INFO] {placeholder_missing} placeholder keys not loaded (expected, eval doesn't need them)")
            if real_missing:
                print(f"[WARN] Missing keys: {len(real_missing)}")
                for k in real_missing[:5]:
                    print(f"  - {k}")
                if len(real_missing) > 5:
                    print(f"  ... and {len(real_missing) - 5} more")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")
            if len(unexpected) > 5:
                print(f"  ... and {len(unexpected) - 5} more")
        
        # 设置所有 SparseLinear 的 hardening_x = 0 以使用 hard mask
        for module in model.modules():
            if hasattr(module, 'hardening_x'):
                module.hardening_x = 0.0
                module._hardening_finalized = True
        print("[INFO] Set hardening_x=0 for all SparseLinear layers (using hard masks)")
        
        # 优化稀疏模型推理速度：将 mask 应用到权重，切换到 dense_forward 模式
        model = optimize_sparse_model_for_inference(model, use_slorb=use_slorb)
        
    else:
        # 普通 HuggingFace 模型
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    
    # ===== 恢复 Channel Pruning Mask（如果 checkpoint 中有） =====
    # 必须在 model.to(device) 之后调用，确保 mask 和模型在同一设备
    if channel_masks_data is not None:
        model = _restore_channel_masks(model, channel_masks_data, model_type, args_dict)
    
    return model, args_dict

def load_model_from_checkpoint_distributed(ckpt_path: str, model_path: str, local_rank: int = 0):
    """
    从 checkpoint 加载模型用于分布式评估
    
    Args:
        ckpt_path: checkpoint 路径 (model.pt 或 retrain_best.pt)
        model_path: 基础模型路径 (用于初始化模型结构)
        local_rank: 本地 rank
    
    Returns:
        加载好的模型
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = f"cuda:{local_rank}"
    
    if rank == 0:
        print(f"[INFO] Loading checkpoint from: {ckpt_path}")
        print(f"[INFO] Serial loading mode: {world_size} ranks will load sequentially to avoid OOM")
    
    # =========================================================================
    # 串行加载：各 rank 按顺序加载 checkpoint 和基础模型
    # 避免所有 rank 同时 torch.load 大文件导致 CPU 内存爆炸 / I/O 卡死
    # =========================================================================
    checkpoint = None
    state_dict = None
    args_dict = {}
    channel_masks_data = None
    
    for loading_rank in range(world_size):
        if rank == loading_rank:
            if rank == 0:
                print(f"[INFO] Rank {rank}: loading checkpoint...")
            # 加载 checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # 获取 state_dict - 支持多种格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 获取训练参数
            args_dict = checkpoint.get('args', {})
            
            # 提取 channel pruning masks（如果有）
            channel_masks_data = checkpoint.get('channel_masks', None)
            if channel_masks_data is not None and rank == 0:
                print(f"[INFO] Found channel_masks in checkpoint ({len(channel_masks_data)} layers)")
            
            # 释放 checkpoint 容器（保留 state_dict、args_dict 和 channel_masks_data）
            if state_dict is not checkpoint:
                del checkpoint
            checkpoint = None
            gc.collect()
            
            if rank == 0:
                print(f"[INFO] Rank {rank}: checkpoint loaded successfully")
        
        # 等待当前 rank 加载完成后再让下一个 rank 开始
        if dist.is_initialized():
            dist.barrier()
    
    # 检查并移除 state_dict 中的 "student." 前缀
    # 这是因为训练时使用 Distill_Model 包装，key 会带有 student. 前缀
    # 但评估时创建的是裸模型，没有这个前缀
    if any(k.startswith('student.') for k in state_dict.keys()):
        if rank == 0:
            print(f"[INFO] Detected 'student.' prefix in checkpoint keys, removing...")
        state_dict = {k.replace('student.', '', 1) if k.startswith('student.') else k: v 
                      for k, v in state_dict.items()}
    
    # 检查是否是 sparse 模型
    is_sparse = any('.mask' in k for k in state_dict.keys())
    
    if rank == 0:
        print(f"[INFO] Detected sparse model: {is_sparse}")
    
    # 推断模型类型
    model_type = detect_model_type_from_path(model_path)
    if rank == 0:
        print(f"[INFO] Detected model type: {model_type}")
    
    # 从 args 获取稀疏配置
    sparsity_ratio = args_dict.get('sparsity_ratio', 0.5)
    mask_type = args_dict.get('mask_type', 'unstructured')
    mask_metric = args_dict.get('mask_metric', 'hessian_obd')
    hard_mask_type = args_dict.get('hard_mask_type', 'topk')
    
    # 获取 SLoRB 配置 (低秩恢复)
    use_slorb = args_dict.get('SLoRB', False)
    slorb_k = args_dict.get('SLoRB_k', 64)
    slorb_init_type = args_dict.get('SLoRB_init_type', 'mean')
    
    if rank == 0:
        print(f"[INFO] Sparsity config: ratio={sparsity_ratio}, mask_type={mask_type}, hard_mask_type={hard_mask_type}")
        if use_slorb:
            print(f"[INFO] SLoRB enabled: k={slorb_k}, init_type={slorb_init_type}")
    
    # =========================================================================
    # 串行初始化模型：各 rank 按顺序执行 from_pretrained + load_state_dict
    # 避免所有 rank 同时从共享文件系统加载模型权重导致 I/O 竞争和 CPU 内存爆炸
    # =========================================================================
    model = None
    
    for loading_rank in range(world_size):
        if rank == loading_rank:
            if rank == 0:
                print(f"[INFO] Rank {rank}: initializing model from {model_path}...")
            
            if is_sparse:
                sparse_cfg = SparseLinearConfig(
                    change_mask=False,
                    mode="sparse_forward",
                    mask_type=mask_type,
                    mask_metric=mask_metric,
                    sparsity_ratio=sparsity_ratio,
                    hard_mask_type=hard_mask_type,
                    # SLoRB 配置 (低秩恢复，推理时也需要)
                    SLoRB=use_slorb,
                    SLoRB_k=slorb_k,
                    SLoRB_init_type=slorb_init_type,
                )
                
                if model_type == 'opt':
                    from model_opt import OPTSparse
                    model = OPTSparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'llama':
                    from model_llama import LlamaSparse
                    model = LlamaSparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'qwen':
                    from model_qwen import QwenSparse
                    model = QwenSparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'mistral':
                    from model_mistral import MistralSparse
                    model = MistralSparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'deepseek_moe':
                    from model_deepseek_moe import DeepSeekMoESparse
                    model = DeepSeekMoESparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'hunyuan':
                    from model_hunyuan import HunyuanSparse
                    model = HunyuanSparse.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                elif model_type == 'gpt2':
                    from model import GPT
                    model = GPT.from_pretrained(model_path, sparselinear_config=sparse_cfg, is_teacher=False)
                else:
                    raise ValueError(f"Unsupported sparse model type: {model_type}")
                
                # 过滤掉训练时特有的 keys，推理时不需要：
                training_only_patterns = ['hessian_diag']
                if not use_slorb:
                    training_only_patterns.extend(['SLoRB_Weight', 'x_proj'])
                keys_to_remove = [k for k in state_dict.keys() 
                                  if any(pattern in k for pattern in training_only_patterns)]
                if keys_to_remove and rank == 0:
                    print(f"[INFO] Filtering out {len(keys_to_remove)} training-only keys ({'/'.join(training_only_patterns)})")
                for k in keys_to_remove:
                    del state_dict[k]
                
                # 加载权重
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
                if missing and rank == 0:
                    placeholder_patterns = ['hessian_diag', 'frozen_mask_flags']
                    real_missing = [k for k in missing if not any(p in k for p in placeholder_patterns)]
                    placeholder_missing = len(missing) - len(real_missing)
                    if placeholder_missing > 0:
                        print(f"[INFO] {placeholder_missing} placeholder keys not loaded (expected)")
                    if real_missing:
                        print(f"[WARN] Missing keys: {len(real_missing)}")
                if unexpected and rank == 0:
                    print(f"[WARN] Unexpected keys: {len(unexpected)}")
                
                # 设置 hardening_x = 0
                for module in model.modules():
                    if hasattr(module, 'hardening_x'):
                        module.hardening_x = 0.0
                        module._hardening_finalized = True
                
                if rank == 0:
                    print("[INFO] Set hardening_x=0 for all SparseLinear layers")
                
                # 优化推理速度
                model = optimize_sparse_model_for_inference(model, use_slorb=use_slorb)
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing and rank == 0:
                    print(f"[WARN] Missing keys: {len(missing)}")
                if unexpected and rank == 0:
                    print(f"[WARN] Unexpected keys: {len(unexpected)}")
            
            # 加载完成后立即释放 state_dict 减少内存占用
            del state_dict
            gc.collect()
            
            # 移到对应 GPU
            model = model.to(device)
            model.eval()
            
            # ===== 恢复 Channel Pruning Mask（如果 checkpoint 中有） =====
            # 必须在 model.to(device) 之后调用，确保 mask 与模型在同一设备
            if channel_masks_data is not None:
                model = _restore_channel_masks(model, channel_masks_data, model_type, args_dict, verbose=(rank == 0))
            
            if rank == 0:
                print(f"[INFO] Rank {rank}: model loaded to {device}")
        
        # 等待当前 rank 加载完成后再让下一个 rank 开始
        if dist.is_initialized():
            dist.barrier()
    
    return model, args_dict

def evaluate_checkpoint(
    ckpt_path: str,
    model_path: str,
    device: str = "cuda:0",
    block_size: int = 1024,
    run_lm_eval: bool = False,
    lm_eval_tasks: Optional[List[str]] = None,
    lm_eval_batch_size: int = 4,
    lm_eval_num_fewshot: int = 0,
    hf_token: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    评估单个 checkpoint 的 wiki PPL 和可选的 lm_eval benchmarks
    
    Returns:
        (wiki_ppl, lm_eval_results)
    """
    print("\n" + "=" * 60)
    print(f"Evaluating: {os.path.basename(ckpt_path)}")
    print("=" * 60)
    
    model, args_dict = load_model_from_checkpoint(ckpt_path, model_path, device)
    
    # 输出模型参数量
    print_model_parameters(model, prefix=os.path.basename(ckpt_path))
    
    # 计算 wiki PPL
    print("[INFO] Computing WikiText-2 PPL...")
    wiki_ppl = eval_ppl(
        model,
        bs=2,
        device=device,
        block_size=block_size,
        model_name_or_path=model_path,
    )
    
    print(f"\n[RESULT] Wiki PPL: {wiki_ppl:.4f}")
    
    # 运行 lm_eval benchmarks (如果启用)
    lm_eval_results = {}
    if run_lm_eval:
        print("\n[INFO] Running lm_eval benchmarks...")
        # 推断模型 dtype
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        lm_eval_results = run_lm_eval_benchmarks(
            model=model,
            model_path=model_path,
            device=device,
            tasks=lm_eval_tasks,
            batch_size=lm_eval_batch_size,
            num_fewshot=lm_eval_num_fewshot,
            hf_token=hf_token,
            dtype=model_dtype,
        )
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return wiki_ppl, lm_eval_results

def evaluate_checkpoint_distributed(
    ckpt_path: str,
    model_path: str,
    local_rank: int = 0,
    block_size: int = 1024,
    run_lm_eval: bool = False,
    lm_eval_tasks: Optional[List[str]] = None,
    lm_eval_batch_size: int = 4,
    lm_eval_num_fewshot: int = 0,
    hf_token: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    分布式评估单个 checkpoint 的 wiki PPL
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = f"cuda:{local_rank}"
    
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Evaluating: {os.path.basename(ckpt_path)}")
        print("=" * 60)
    
    model, args_dict = load_model_from_checkpoint_distributed(ckpt_path, model_path, local_rank)
    
    # 输出模型参数量 (仅 rank 0)
    if rank == 0:
        print_model_parameters(model, prefix=os.path.basename(ckpt_path))
    
    # 从 args_dict 获取训练时的 dtype 配置，确保评估时使用一致的精度
    ptdtype = None
    dtype_str = args_dict.get('dtype', 'auto')
    use_fsdp_mp = args_dict.get('fsdp_mixed_precision', False)
    if use_fsdp_mp or dtype_str in ('bfloat16', 'float16'):
        if dtype_str == 'float16':
            ptdtype = torch.float16
        else:
            # auto / bfloat16: 根据 GPU 能力选择
            ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if rank == 0:
            print(f"[INFO] Using ptdtype={ptdtype} for eval (from training config: dtype={dtype_str}, fsdp_mp={use_fsdp_mp})")
    
    # 计算 wiki PPL (使用分布式版本)
    if rank == 0:
        print("[INFO] Computing WikiText-2 PPL (distributed)...")
    
    wiki_ppl = eval_ppl_distributed(
        model,
        bs=2,
        device=device,
        block_size=block_size,
        model_name_or_path=model_path,
        ptdtype=ptdtype,
    )
    
    if rank == 0:
        print(f"\n[RESULT] Wiki PPL: {wiki_ppl:.4f}")
    
    # 运行 lm_eval benchmarks (如果启用) - 只在 rank 0 上运行
    # 重要：lm_eval 只在 rank 0 执行，其他 rank 需要提前释放模型并等待
    # 否则如果模型内部有 NCCL 通信，会导致 rank 不同步超时
    lm_eval_results = {}
    if run_lm_eval:
        # 在 lm_eval 之前先同步，确保所有 rank 都完成了 PPL 评估
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print("\n[INFO] Running lm_eval benchmarks (only on rank 0)...")
            lm_eval_results = run_lm_eval_benchmarks(
                model=model,
                model_path=model_path,
                device=device,
                tasks=lm_eval_tasks,
                batch_size=lm_eval_batch_size,
                num_fewshot=lm_eval_num_fewshot,
                hf_token=hf_token,
                dtype=ptdtype,
            )
        else:
            # 非 rank 0：提前释放模型，避免 lm_eval 期间的 NCCL 通信冲突
            del model
            torch.cuda.empty_cache()
            model = None  # 标记已释放
        
        # lm_eval 完成后同步
        if dist.is_initialized():
            dist.barrier()
    
    # 清理显存 (rank 0 或未运行 lm_eval 的情况)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    
    return wiki_ppl, lm_eval_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Wiki PPL and lm_eval benchmarks for checkpoints")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the first checkpoint (e.g., model.pt)"
    )
    parser.add_argument(
        "--ckpt_path2",
        type=str,
        default=None,
        help="Path to the second checkpoint for comparison (e.g., retrain_best.pt)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model (for tokenizer and config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Block size for evaluation (default: 2048, should match training block_size)"
    )
    # 分布式相关参数
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed evaluation for faster 7B+ model eval. Use with torchrun."
    )
    # lm_eval 相关参数
    parser.add_argument(
        "--run_lm_eval",
        action="store_true",
        help="Run lm_eval harness benchmarks (BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA)"
    )
    parser.add_argument(
        "--lm_eval_tasks",
        type=str,
        default=None,
        help="Comma-separated list of lm_eval tasks (default: boolq,rte,hellaswagwinogrande,arc_easy,arc_challenge,openbookqa)"
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        type=int,
        default=4,
        help="Batch size for lm_eval (default: 4)"
    )
    parser.add_argument(
        "--lm_eval_num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for lm_eval (default: 0, i.e., zero-shot)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON"
    )
    parser.add_argument(
        "--eval_base_model",
        action="store_true",
        help="Also evaluate the original base model (without any checkpoint)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for authentication (to avoid rate limiting)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="When both --eval_base_model and --ckpt_path are given, evaluate them in parallel "
             "on two GPUs (base model on cuda:0, checkpoint on cuda:1). "
             "Not compatible with --distributed (torchrun) mode."
    )
    
    args = parser.parse_args()
    
    # 解析 lm_eval tasks
    lm_eval_tasks = None
    if args.lm_eval_tasks:
        lm_eval_tasks = [t.strip() for t in args.lm_eval_tasks.split(",")]
    
    # 初始化分布式环境
    rank, world_size, local_rank = 0, 1, 0
    is_distributed = args.distributed or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    
    if is_distributed:
        rank, world_size, local_rank = setup_distributed()
        if rank == 0:
            print(f"[INFO] Distributed mode enabled: world_size={world_size}")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Wiki PPL Evaluation (Same method as train_universal)")
        if args.run_lm_eval:
            print("+ lm_eval Harness Benchmarks")
        if is_distributed:
            print(f"[MODE] Distributed evaluation with {world_size} GPUs")
        print("=" * 60)
        print(f"Base model path: {args.model_path}")
        print(f"Device: cuda:{local_rank}" if is_distributed else f"Device: {args.device}")
        print(f"Block size: {args.block_size}")
        if args.run_lm_eval:
            print(f"lm_eval tasks: {lm_eval_tasks or 'default (all 7 benchmarks)'}")
            print(f"lm_eval batch_size: {args.lm_eval_batch_size}")
            print(f"lm_eval num_fewshot: {args.lm_eval_num_fewshot}")
    
    results = {}
    all_lm_eval_results = {}
    
    # =========================================================================
    # 并行模式：base model 和 checkpoint 分别在两个 GPU 上同时评估
    # =========================================================================
    can_parallel = (
        args.parallel
        and not is_distributed
        and args.eval_base_model
        and args.ckpt_path
        and torch.cuda.device_count() >= 2
    )
    
    if can_parallel:
        print("\n[MODE] Parallel evaluation: base model on cuda:0, checkpoint on cuda:1")
        
        # 使用 multiprocessing 在两个 GPU 上并行评估
        result_queue = mp.Queue()
        
        def _eval_worker(queue, task_name, task_type, ckpt_path, model_path, device,
                         block_size, run_lm_eval, lm_eval_tasks, lm_eval_batch_size,
                         lm_eval_num_fewshot, hf_token):
            """在指定 GPU 上评估一个模型的 worker 进程"""
            try:
                torch.cuda.set_device(device)
                
                if task_type == 'base':
                    print(f"\n[GPU {device}] Loading base model...")
                    model = load_base_model(model_path, device)
                    print_model_parameters(model, prefix="Base Model")
                    
                    print(f"[GPU {device}] Computing WikiText-2 PPL for base model...")
                    wiki_ppl = eval_ppl(
                        model, bs=2, device=device,
                        block_size=block_size, model_name_or_path=model_path,
                    )
                    print(f"\n[GPU {device}] Base Model Wiki PPL: {wiki_ppl:.4f}")
                    
                    lm_eval_result = {}
                    if run_lm_eval:
                        print(f"[GPU {device}] Running lm_eval for base model...")
                        try:
                            model_dtype = next(model.parameters()).dtype
                        except StopIteration:
                            model_dtype = torch.float16
                        lm_eval_result = run_lm_eval_benchmarks(
                            model=model, model_path=model_path, device=device,
                            tasks=lm_eval_tasks, batch_size=lm_eval_batch_size,
                            num_fewshot=lm_eval_num_fewshot, hf_token=hf_token,
                            dtype=model_dtype,
                        )
                    
                    del model
                    torch.cuda.empty_cache()
                    queue.put((task_name, wiki_ppl, lm_eval_result))
                    
                else:  # checkpoint
                    print(f"\n[GPU {device}] Loading checkpoint: {ckpt_path}")
                    wiki_ppl, lm_eval_result = evaluate_checkpoint(
                        ckpt_path, model_path, device, block_size,
                        run_lm_eval=run_lm_eval, lm_eval_tasks=lm_eval_tasks,
                        lm_eval_batch_size=lm_eval_batch_size,
                        lm_eval_num_fewshot=lm_eval_num_fewshot,
                        hf_token=hf_token,
                    )
                    queue.put((task_name, wiki_ppl, lm_eval_result))
                    
            except Exception as e:
                import traceback
                print(f"\n[GPU {device}] ERROR in {task_type}: {e}")
                traceback.print_exc()
                queue.put((task_name, None, {}))
        
        # 启动两个进程
        common_kwargs = dict(
            block_size=args.block_size,
            run_lm_eval=args.run_lm_eval,
            lm_eval_tasks=lm_eval_tasks,
            lm_eval_batch_size=args.lm_eval_batch_size,
            lm_eval_num_fewshot=args.lm_eval_num_fewshot,
            hf_token=args.hf_token,
        )
        
        p_base = mp.Process(
            target=_eval_worker,
            args=(result_queue, "base_model", "base", None, args.model_path, "cuda:0"),
            kwargs=common_kwargs,
        )
        
        ckpt1_name = os.path.basename(args.ckpt_path)
        p_ckpt = mp.Process(
            target=_eval_worker,
            args=(result_queue, ckpt1_name, "checkpoint", args.ckpt_path, args.model_path, "cuda:1"),
            kwargs=common_kwargs,
        )
        
        p_base.start()
        p_ckpt.start()
        
        p_base.join()
        p_ckpt.join()
        
        # 收集结果
        while not result_queue.empty():
            task_name, wiki_ppl, lm_eval_result = result_queue.get()
            if wiki_ppl is not None:
                results[task_name] = wiki_ppl
            if lm_eval_result:
                all_lm_eval_results[task_name] = lm_eval_result
        
        # 如果还有 ckpt_path2，串行评估（复用 cuda:0）
        if args.ckpt_path2:
            ppl2, lm_eval2 = evaluate_checkpoint(
                args.ckpt_path2, args.model_path, "cuda:0", args.block_size,
                run_lm_eval=args.run_lm_eval, lm_eval_tasks=lm_eval_tasks,
                lm_eval_batch_size=args.lm_eval_batch_size,
                lm_eval_num_fewshot=args.lm_eval_num_fewshot,
                hf_token=args.hf_token,
            )
            ckpt2_name = os.path.basename(args.ckpt_path2)
            results[ckpt2_name] = ppl2
            if lm_eval2:
                all_lm_eval_results[ckpt2_name] = lm_eval2
    
    # =========================================================================
    # 串行模式（非并行时走此路径）
    # =========================================================================
    
    # 评估原始模型 (如果启用)
    if not can_parallel and args.eval_base_model:
        if rank == 0:
            print("\n" + "=" * 60)
            print("Evaluating: Base Model (Original)")
            print("=" * 60)
        
        if is_distributed:
            base_model = load_base_model_distributed(args.model_path, local_rank)
            device = f"cuda:{local_rank}"
            
            # 输出基础模型参数量 (仅 rank 0)
            if rank == 0:
                print_model_parameters(base_model, prefix="Base Model")
            
            if rank == 0:
                print("[INFO] Computing WikiText-2 PPL for base model (distributed)...")
            
            base_wiki_ppl = eval_ppl_distributed(
                base_model,
                bs=2,
                device=device,
                block_size=args.block_size,
                model_name_or_path=args.model_path,
            )
        else:
            base_model = load_base_model(args.model_path, args.device)
            device = args.device
            
            # 输出基础模型参数量
            print_model_parameters(base_model, prefix="Base Model")
            
            print("[INFO] Computing WikiText-2 PPL for base model...")
            base_wiki_ppl = eval_ppl(
                base_model,
                bs=2,
                device=device,
                block_size=args.block_size,
                model_name_or_path=args.model_path,
            )
        
        if rank == 0:
            print(f"\n[RESULT] Base Model Wiki PPL: {base_wiki_ppl:.4f}")
        results["base_model"] = base_wiki_ppl
        
        # 运行 lm_eval benchmarks (如果启用) - 只在 rank 0 执行
        # 分布式环境下需要特殊处理：非 rank 0 提前释放模型避免 NCCL 冲突
        if args.run_lm_eval:
            # 先同步，确保所有 rank 完成 PPL 评估
            if is_distributed:
                dist.barrier()
            
            if rank == 0:
                print("\n[INFO] Running lm_eval benchmarks for base model...")
                # 推断 base model dtype
                try:
                    base_model_dtype = next(base_model.parameters()).dtype
                except StopIteration:
                    base_model_dtype = torch.float16
                base_lm_eval = run_lm_eval_benchmarks(
                    model=base_model,
                    model_path=args.model_path,
                    device=device,
                    tasks=lm_eval_tasks,
                    batch_size=args.lm_eval_batch_size,
                    num_fewshot=args.lm_eval_num_fewshot,
                    hf_token=args.hf_token,
                    dtype=base_model_dtype,
                )
                if base_lm_eval:
                    all_lm_eval_results["base_model"] = base_lm_eval
            elif is_distributed:
                # 非 rank 0：提前释放模型，避免 lm_eval 期间 NCCL 通信冲突
                del base_model
                torch.cuda.empty_cache()
                base_model = None
            
            # lm_eval 完成后同步
            if is_distributed:
                dist.barrier()
        
        # 清理显存 (rank 0 或未运行 lm_eval 的情况)
        if base_model is not None:
            del base_model
            torch.cuda.empty_cache()
        
        # 强制垃圾回收，确保在加载 checkpoint 前释放所有内存
        gc.collect()
        torch.cuda.empty_cache()
    
    # 评估第一个 checkpoint (如果提供)
    if not can_parallel and args.ckpt_path:
        if is_distributed:
            ppl1, lm_eval1 = evaluate_checkpoint_distributed(
                args.ckpt_path,
                args.model_path,
                local_rank,
                args.block_size,
                run_lm_eval=args.run_lm_eval,
                lm_eval_tasks=lm_eval_tasks,
                lm_eval_batch_size=args.lm_eval_batch_size,
                lm_eval_num_fewshot=args.lm_eval_num_fewshot,
                hf_token=args.hf_token,
            )
        else:
            ppl1, lm_eval1 = evaluate_checkpoint(
                args.ckpt_path,
                args.model_path,
                args.device,
                args.block_size,
                run_lm_eval=args.run_lm_eval,
                lm_eval_tasks=lm_eval_tasks,
                lm_eval_batch_size=args.lm_eval_batch_size,
                lm_eval_num_fewshot=args.lm_eval_num_fewshot,
                hf_token=args.hf_token,
            )
        ckpt1_name = os.path.basename(args.ckpt_path)
        results[ckpt1_name] = ppl1
        if lm_eval1:
            all_lm_eval_results[ckpt1_name] = lm_eval1
    
    # 评估第二个 checkpoint (如果提供)
    if not can_parallel and args.ckpt_path2:
        if is_distributed:
            ppl2, lm_eval2 = evaluate_checkpoint_distributed(
                args.ckpt_path2,
                args.model_path,
                local_rank,
                args.block_size,
                run_lm_eval=args.run_lm_eval,
                lm_eval_tasks=lm_eval_tasks,
                lm_eval_batch_size=args.lm_eval_batch_size,
                lm_eval_num_fewshot=args.lm_eval_num_fewshot,
                hf_token=args.hf_token,
            )
        else:
            ppl2, lm_eval2 = evaluate_checkpoint(
                args.ckpt_path2,
                args.model_path,
                args.device,
                args.block_size,
                run_lm_eval=args.run_lm_eval,
                lm_eval_tasks=lm_eval_tasks,
                lm_eval_batch_size=args.lm_eval_batch_size,
                lm_eval_num_fewshot=args.lm_eval_num_fewshot,
                hf_token=args.hf_token,
            )
        ckpt2_name = os.path.basename(args.ckpt_path2)
        results[ckpt2_name] = ppl2
        if lm_eval2:
            all_lm_eval_results[ckpt2_name] = lm_eval2
    
    # 检查是否有结果可以展示 - 只在 rank 0 输出
    if rank == 0:
        if not results:
            print("\n[ERROR] No models to evaluate. Please provide --ckpt_path or --eval_base_model")
        else:
            # 打印对比结果
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"{'Checkpoint':<30} {'Wiki PPL':>15}")
            print("-" * 47)
            for name, ppl in results.items():
                print(f"{name:<30} {ppl:>15.4f}")
            
            if len(results) == 2:
                names = list(results.keys())
                diff = results[names[1]] - results[names[0]]
                pct = (diff / results[names[0]]) * 100
                print("-" * 47)
                print(f"{'Difference':<30} {diff:>+15.4f} ({pct:+.2f}%)")
            
            # 打印 lm_eval 结果
            if all_lm_eval_results:
                print("\n" + "=" * 60)
                print("lm_eval Benchmark Results")
                print("=" * 60)
                
                all_tasks = set()
                for lm_results in all_lm_eval_results.values():
                    all_tasks.update(lm_results.keys())
                all_tasks = sorted(all_tasks)
                
                header = f"{'Task':<20}"
                for ckpt_name in all_lm_eval_results.keys():
                    header += f" {ckpt_name[:15]:>15}"
                if len(all_lm_eval_results) == 2:
                    header += f" {'Diff':>10}"
                print(header)
                print("-" * len(header))
                
                ckpt_names = list(all_lm_eval_results.keys())
                for task in all_tasks:
                    row = f"{task:<20}"
                    values = []
                    for ckpt_name in ckpt_names:
                        val = all_lm_eval_results[ckpt_name].get(task, None)
                        if val is not None:
                            row += f" {val:>15.2f}%"
                            values.append(val)
                        else:
                            row += f" {'N/A':>15}"
                            values.append(None)
                    
                    if len(values) == 2 and all(v is not None for v in values):
                        diff = values[1] - values[0]
                        row += f" {diff:>+9.2f}%"
                    
                    print(row)
            
            print("=" * 60)
            
            # 保存 JSON 结果
            if args.output_json:
                output_data = {
                    "wiki_ppl": results,
                    "lm_eval": all_lm_eval_results,
                    "config": {
                        "model_path": args.model_path,
                        "block_size": args.block_size,
                        "lm_eval_tasks": lm_eval_tasks,
                        "lm_eval_batch_size": args.lm_eval_batch_size,
                        "lm_eval_num_fewshot": args.lm_eval_num_fewshot,
                        "distributed": is_distributed,
                        "world_size": world_size,
                    }
                }
                with open(args.output_json, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n[INFO] Results saved to: {args.output_json}")
    
    # 清理分布式环境
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
