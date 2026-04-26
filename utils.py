# utils.py
import os
import pickle
import datetime
import torch
import torch.nn as nn
import math
import numpy as np

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
except Exception:
    FSDP = None
    StateDictType = None

from sparse_modeling import SparseLinear


@torch.no_grad()
def get_raw_model(model):
    """
    Unwrap model from various wrappers (DDP, FSDP, Distill_Model).
    Returns the innermost student/base model.
    """
    # Handle FSDP wrapper
    if hasattr(model, '_fsdp_wrapped_module'):
        inner = model._fsdp_wrapped_module
        if hasattr(inner, 'student'):
            # Distill_Model inside FSDP
            student = inner.student
            # Student might also be FSDP-wrapped
            if hasattr(student, '_fsdp_wrapped_module'):
                return student._fsdp_wrapped_module
            return student
        return inner
    # Handle DDP wrapper
    if hasattr(model, "module"):
        if hasattr(model.module, "student"):
            return model.module.student
        return model.module
    # Handle Distill_Model
    if hasattr(model, "student"):
        student = model.student
        # Student might be FSDP-wrapped
        if hasattr(student, '_fsdp_wrapped_module'):
            return student._fsdp_wrapped_module
        return student
    return model


@torch.no_grad()
def set_model_mode(model, mode: str):
    raw = get_raw_model(model)
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            module.mode = mode


@torch.no_grad()
def sync_weight(model, device=None):
    # model.to(device) 在 main.py 已经做了也没关系
    if device is not None:
        model = model.to(device)
    raw = get_raw_model(model)
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            module.sync_weight()


@torch.no_grad()
def initialize_model(model):
    raw = get_raw_model(model)
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            module.initialize()
            if getattr(module, "SLoRB", False):
                module.init_SLoRB()


# -----------------------------
# EMA updates (called every step)
# -----------------------------
@torch.no_grad()
def update_model_grad_ema(model, update_hessian_with_grad2: bool = True):
    """
    Must be called after backward() and before optimizer.step().
    - updates grad_ema (if needed for mask_metric)
    - updates hessian_diag (if needed + update_hessian_with_grad2=True)
    - importance_ema REMOVED: scores computed on-the-fly in compute_gate_target()
    """
    raw = get_raw_model(model)
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            module.update_grad_hessian_ema(update_hessian_with_grad2=update_hessian_with_grad2)
            # update_importance_ema() removed - scores computed directly from grad_ema/hessian_diag


@torch.no_grad()
def update_mask_penalty_lr(model, step: int, max_steps: int, 
                           penalty_lr_min: float, penalty_lr_max: float, 
                           schedule: str = 'linear'):
    """
    Update mask_penalty_lr for all SparseLinear modules based on schedule.
    
    Args:
        model: The model containing SparseLinear modules
        step: Current training step
        max_steps: Total training steps
        penalty_lr_min: Starting penalty lr (at step=0)
        penalty_lr_max: Ending penalty lr (at step=max_steps)
        schedule: 'constant', 'linear', or 'cosine'
    
    Returns:
        current_penalty_lr: The computed penalty lr value
    """
    if schedule == 'constant':
        current_penalty_lr = penalty_lr_max
    elif schedule == 'linear':
        # 线性增长：从 min 到 max
        progress = min(1.0, step / max(1, max_steps))
        current_penalty_lr = penalty_lr_min + (penalty_lr_max - penalty_lr_min) * progress
    elif schedule == 'cosine':
        # 余弦调度：开始慢，后期快速增长（反向余弦退火）
        progress = min(1.0, step / max(1, max_steps))
        # cosine annealing: 0 -> 1，开始慢后期快
        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        current_penalty_lr = penalty_lr_min + (penalty_lr_max - penalty_lr_min) * cosine_factor
    else:
        current_penalty_lr = penalty_lr_max
    
    # 更新所有 SparseLinear 模块的 mask_penalty_lr
    raw = get_raw_model(model)
    for name, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            module.mask_penalty_lr = current_penalty_lr
    
    return current_penalty_lr


@torch.no_grad()
def calculate_model_mask(model, step=0, lambda_mid: float = 0.0):
    """
    Call periodically (your main.py already calls every output_flip_every),
    but SparseLinear internally only updates if step % mask_update_period==0.
    
    FSDP Compatibility (2024-02 fix):
    - mask 和 weight 都是 nn.Parameter，FSDP 会将同一 FSDP unit 内的所有 Parameter
      拼接成一个 FlatParameter 后均匀分片到各 rank。不同 Parameter 在 FlatParameter 
      中占据不同偏移，导致 sharded 状态下 weight.shape != mask.shape。
    - 因此必须在 summon_full_params() 上下文中执行 mask 更新，以恢复完整的 2D shape。
    - summon_full_params 是 collective 操作，所有 rank 必须一起调用。
      calculate_model_mask 在训练循环中所有 rank 同步调用，不会死锁。
    
    GLU Joint Mask Support:
    - For SwiGLU architectures (LLaMA, Qwen, Mistral), gate_proj and up_proj are element-wise
      multiplied. When glu_joint_mask=True, we compute a JOINT mask from combined scores
      and apply the SAME mask to both layers, ensuring aligned pruning.
    """
    from contextlib import nullcontext
    
    raw = get_raw_model(model)
    
    # ========== 早期退出：warmup 期间跳过昂贵的 summon_full_params ==========
    # 在 sparsity_warmup_steps 之前，所有 SparseLinear.update_mask() 都会直接 return。
    # 此时执行 summon_full_params（fully_sharded 模式下需要 all_gather 所有参数）
    # 是完全浪费的，而且在多 GPU 环境下开销巨大（可能导致看起来"卡住"）。
    # 检查是否所有 module 都会跳过更新，如果是则直接返回。
    _any_would_update = False
    for _name, _mod in raw.named_modules():
        if isinstance(_mod, SparseLinear):
            if (not _mod.change_mask) or (_mod.mask_type == "none"):
                continue
            if step < int(_mod.cfg.sparsity_warmup_steps):
                continue
            K = max(1, int(_mod._mask_update_period(step)))
            if step % K != 0:
                continue
            # 这个 module 会真正执行 mask 更新
            _any_would_update = True
            break
    
    if not _any_would_update:
        # 没有任何 module 需要更新 mask，跳过 summon_full_params
        return
    
    # ========== 检测 FSDP，准备 summon_full_params 上下文 ==========
    fsdp_target = None
    if FSDP is not None:
        # model 可能是：FSDP(student)、Distill_Model 包含 FSDP(student)、或非 FSDP
        if isinstance(model, FSDP):
            fsdp_target = model
        elif hasattr(model, 'module') and isinstance(model.module, FSDP):
            fsdp_target = model.module
        elif hasattr(model, 'student') and isinstance(model.student, FSDP):
            fsdp_target = model.student
        # 也检查 raw model 的外层包裹
        if fsdp_target is None:
            _m = model
            while hasattr(_m, '_fsdp_wrapped_module'):
                fsdp_target = _m
                break
    
    if fsdp_target is not None:
        # writeback=True：mask 更新（in-place .data 修改）需要写回到 sharded 参数
        # recurse=True：递归恢复所有子模块的参数
        fsdp_ctx = FSDP.summon_full_params(fsdp_target, writeback=True, recurse=True)
    else:
        fsdp_ctx = nullcontext()
    
    # ========== GLU Joint Mask: Identify gate/up pairs ==========
    # Collect all SparseLinear modules first
    sparse_modules_dict = {}
    for name, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            sparse_modules_dict[name] = module
    
    # Check if any module has glu_joint_mask enabled
    glu_joint_enabled = any(
        getattr(m.cfg, 'glu_joint_mask', False) 
        for m in sparse_modules_dict.values()
    )
    
    # Identify gate/up pairs if glu_joint_mask is enabled
    glu_pairs = {}  # {gate_name: up_name} mapping
    if glu_joint_enabled:
        glu_pairs = _identify_glu_pairs(sparse_modules_dict)
        # Track which modules are handled by joint update
        joint_handled = set()
        for gate_name, up_name in glu_pairs.items():
            joint_handled.add(gate_name)
            joint_handled.add(up_name)
    else:
        joint_handled = set()
    
    # ========== 在 summon_full_params 上下文中执行 mask 更新 ==========
    # summon_full_params 恢复所有 Parameter 到完整的原始 shape（2D），
    # 使得 weight.shape == mask.shape == (out_features, in_features)，
    # 所有操作（block reshape, N:M 分组等）都能正确执行。
    #
    # CRITICAL: 在 summon 上下文中，所有 rank 看到完全相同的完整参数，
    # 不需要 all_reduce 来同步 tau。而且如果某些 rank 因为 mask.dim()!=2
    # 或 shape mismatch 而跳过 update_mask，就不会参与 all_reduce，
    # 导致 NCCL collective mismatch → 死锁/超时。
    # 所以必须在 summon 内部禁用 _tau_unstructured 的 all_reduce。
    for module in sparse_modules_dict.values():
        module._in_summon_context = True
    try:
        with fsdp_ctx:
            # First pass: handle GLU joint pairs
            if glu_joint_enabled and glu_pairs:
                for gate_name, up_name in glu_pairs.items():
                    if gate_name not in sparse_modules_dict or up_name not in sparse_modules_dict:
                        continue
                    gate_module = sparse_modules_dict[gate_name]
                    up_module = sparse_modules_dict[up_name]
                    _update_glu_joint_mask(gate_module, up_module, step, lambda_mid)
        
            # Second pass: handle remaining modules
            for name, module in raw.named_modules():
                if isinstance(module, SparseLinear):
                    if name in joint_handled:
                        continue  # Already handled
                    module.update_mask(step, lambda_mid=lambda_mid)
    finally:
        # 恢复 flag，确保非 summon 上下文中 _tau_unstructured 正常做 all_reduce
        for module in sparse_modules_dict.values():
            module._in_summon_context = False


def _identify_glu_pairs(sparse_modules_dict):
    """Identify gate_proj/up_proj pairs in GLU architectures.
    
    GLU models (LLaMA, Qwen, Mistral) use:
        intermediate = silu(gate_proj(x)) * up_proj(x)
    
    The gate and up projections are element-wise multiplied, so they must
    have aligned pruning patterns for efficient computation.
    
    Layer naming patterns:
        - LLaMA/Qwen/Mistral: model.layers.{i}.mlp.gate_proj / up_proj
        - Some variants: layers.{i}.mlp.w1 (gate) / w3 (up)
    
    Args:
        sparse_modules_dict: {name: SparseLinear} mapping
        
    Returns:
        {gate_name: up_name} mapping of paired layers
    """
    import re
    pairs = {}
    
    # Pattern 1: gate_proj / up_proj naming
    gate_pattern = re.compile(r'(.+\.mlp)\.gate_proj$')
    for name in sparse_modules_dict:
        match = gate_pattern.match(name)
        if match:
            prefix = match.group(1)
            up_name = f"{prefix}.up_proj"
            if up_name in sparse_modules_dict:
                # Check if glu_joint_mask is enabled for this pair
                gate_mod = sparse_modules_dict[name]
                if getattr(gate_mod.cfg, 'glu_joint_mask', False):
                    pairs[name] = up_name
    
    # Pattern 2: w1 (gate) / w3 (up) naming (some model variants)
    w1_pattern = re.compile(r'(.+\.mlp)\.w1$')
    for name in sparse_modules_dict:
        match = w1_pattern.match(name)
        if match:
            prefix = match.group(1)
            w3_name = f"{prefix}.w3"
            if w3_name in sparse_modules_dict:
                gate_mod = sparse_modules_dict[name]
                if getattr(gate_mod.cfg, 'glu_joint_mask', False):
                    pairs[name] = w3_name
    
    return pairs


@torch.no_grad()
def _update_glu_joint_mask(gate_module, up_module, step, lambda_mid):
    """Update masks for a GLU gate/up pair using JOINT importance scores.
    
    The key insight: gate_proj and up_proj outputs are element-wise multiplied,
    so pruning them independently causes misalignment. We compute a joint mask:
    
    1. Compute importance scores for both layers
    2. Combine: joint_score = gate_score + up_score (element-wise)
    3. Compute target mask G from joint_score using existing mask update logic
    4. Apply the SAME mask to both gate_proj and up_proj
    
    This ensures that the same weight positions are pruned in both layers,
    avoiding the "wasted computation" problem where one is pruned but not the other.
    
    Args:
        gate_module: SparseLinear for gate_proj
        up_module: SparseLinear for up_proj
        step: Current training step
        lambda_mid: Mid-penalty coefficient
    """
    # Basic checks
    if not gate_module.change_mask or gate_module.mask_type == "none":
        gate_module.weight.flipped_mask = 0
        gate_module.weight.init_flipped_mask = 0
        up_module.weight.flipped_mask = 0
        up_module.weight.init_flipped_mask = 0
        return
    
    if step < int(gate_module.cfg.sparsity_warmup_steps):
        return
    
    K = max(1, int(gate_module._mask_update_period(step)))
    if step % K != 0:
        return
    
    # Shape check
    if gate_module.mask.shape != up_module.mask.shape:
        import warnings
        warnings.warn(
            f"[GLU Joint] Shape mismatch: gate={gate_module.mask.shape}, up={up_module.mask.shape}. "
            f"Falling back to independent update."
        )
        gate_module.update_mask(step, lambda_mid=lambda_mid)
        up_module.update_mask(step, lambda_mid=lambda_mid)
        return
    
    # Save previous masks for flip rate calculation
    prev_gate = gate_module.mask.detach().clone()
    prev_up = up_module.mask.detach().clone()
    
    # ========== Compute joint importance scores ==========
    gate_scores = _compute_importance_scores(gate_module)
    up_scores = _compute_importance_scores(up_module)
    
    # Joint score: simple sum (both contribute equally)
    # Alternative strategies: max, weighted sum, geometric mean
    joint_scores = gate_scores + up_scores
    
    # ========== Compute joint target mask G ==========
    # Use gate_module's compute_gate_target but with joint scores
    # We need to temporarily replace the importance computation
    G = _compute_joint_gate_target(gate_module, joint_scores, step)
    
    # ========== Apply EMA update with joint G to both modules ==========
    a = float(gate_module.cfg.mask_lr)
    
    # Apply mid-penalty if needed (same logic as in SparseLinear.update_mask)
    if lambda_mid is not None and float(lambda_mid) != 0.0:
        G = _apply_mid_penalty_to_gate(gate_module, G, lambda_mid)
    
    # Multiplicative binarization push
    d = float(getattr(gate_module.cfg, 'mask_binarize_decay', 0.0) or 0.0)
    if d != 0.0:
        lt = G < 0.5
        gt = G > 0.5
        G = G.clone()
        if lt.any():
            G[lt] = G[lt] * (1.0 - d)
        if gt.any():
            G[gt] = G[gt] * (1.0 + d)
    
    # EMA update: apply SAME G to both gate and up
    gate_module.mask.mul_(1 - a).add_(G, alpha=a).clamp_(0.0, 1.0)
    up_module.mask.mul_(1 - a).add_(G, alpha=a).clamp_(0.0, 1.0)
    
    # Update hardening_x for both modules
    start = int(getattr(gate_module.cfg, 'mask_hardening_start', 0) or 0)
    duration = int(getattr(gate_module.cfg, 'mask_hardening_duration', 0) or 0)
    if duration <= 0:
        new_hx = 0.0 if step >= start and start > 0 else 1.0
    else:
        if step < start:
            new_hx = 1.0
        elif step >= start + duration:
            new_hx = 0.0
        else:
            progress = float(step - start) / float(max(1, duration))
            new_hx = max(0.0, min(1.0, 1.0 - progress))
    
    gate_module.hardening_x = new_hx
    up_module.hardening_x = new_hx
    
    # Calculate flip rates
    hard_prev_gate = (gate_module._hard_mask_from_soft(prev_gate) > 0.5)
    hard_now_gate = (gate_module._hard_mask_from_soft(gate_module.mask) > 0.5)
    gate_module.weight.flipped_mask = int((hard_prev_gate ^ hard_now_gate).sum().item())
    gate_module.weight.init_flipped_mask = 0
    
    hard_prev_up = (up_module._hard_mask_from_soft(prev_up) > 0.5)
    hard_now_up = (up_module._hard_mask_from_soft(up_module.mask) > 0.5)
    up_module.weight.flipped_mask = int((hard_prev_up ^ hard_now_up).sum().item())
    up_module.weight.init_flipped_mask = 0
    
    # Keep pointers
    gate_module.weight.mask = gate_module.mask
    up_module.weight.mask = up_module.mask


def _compute_importance_scores(module):
    """Compute importance scores for a SparseLinear module.
    
    Mirrors the score computation in SparseLinear.compute_gate_target().
    """
    eps = 1e-8
    metric = module.mask_metric
    W = module.weight.detach()
    
    def get_hessian():
        if hasattr(module, '_hessian_diag_local') and module._hessian_diag_local.shape == W.shape:
            return module._hessian_diag_local.detach()
        if not module._is_placeholder('hessian_diag') and module.hessian_diag.shape == W.shape:
            return module.hessian_diag.detach()
        return None
    
    if metric == "magnitude":
        scores = W.abs()
    elif metric == "movement":
        if not module._is_placeholder('grad_ema') and module.grad_ema.shape == W.shape:
            g = module.grad_ema.detach()
            scores = (W * g).abs() / max(float(module.cfg.temperature), eps)
        else:
            scores = W.abs()
    elif metric == "hessian_obd":
        H = get_hessian()
        if H is not None:
            scores = (H + eps) * (W * W)
        else:
            scores = W * W
    elif metric == "hessian_ratio":
        H = get_hessian()
        if H is not None:
            scores = W.abs() / torch.sqrt(H + eps)
        else:
            scores = W.abs()
    elif metric == "hessian":
        H = get_hessian()
        if H is not None:
            scores = H
        else:
            scores = W.abs()
    elif metric == "wanda":
        if not module._is_placeholder('scaler_row'):
            scaler = module.scaler_row.detach()
            scores = W.abs() * torch.sqrt(scaler.reshape(1, -1) + eps)
        else:
            scores = W.abs()
    else:
        scores = W.abs()
    
    return scores


def _compute_joint_gate_target(module, scores, step):
    """Compute target mask G from joint scores.
    
    Similar to SparseLinear.compute_gate_target() but uses pre-computed joint scores.
    """
    if scores.numel() == 0:
        return module.mask
    
    # Normalize scores
    scores_fp32 = scores.float()
    mu = scores_fp32.mean()
    sigma = scores_fp32.std(unbiased=False) + 1e-6
    scores = ((scores_fp32 - mu) / sigma).to(module.mask.dtype)
    
    T = module._temperature(step)
    Tt = torch.tensor(T, device=scores.device, dtype=torch.float32)
    
    frozen = (
        module.frozen_mask_flags.bool()
        if not module._is_placeholder('frozen_mask_flags') else torch.zeros_like(module.mask, dtype=torch.bool)
    )
    
    if module.mask_type == "unstructured":
        sparsity = module._effective_sparsity(step)
        tau = module._tau_unstructured(scores, sparsity)
        G = torch.sigmoid((scores.float() - tau) / (Tt + 1e-8)).to(scores.dtype)
    
    elif module.mask_type == "structured":
        N, M = int(module.N), int(module.M)
        out_dim, in_dim = scores.shape
        assert in_dim % M == 0, f"in_features ({in_dim}) must be divisible by M ({M})."
        groups = in_dim // M
        s = scores.float().view(out_dim, groups, M)
        topk = torch.topk(s, k=N, dim=-1, largest=True)
        topv = topk.values
        topi = topk.indices
        tau_g = topv[..., -1].unsqueeze(-1)
        if getattr(module.cfg, 'structured_exact', False):
            Gg = torch.zeros_like(s, dtype=scores.dtype)
            Gg.scatter_(-1, topi, 1.0)
            G = Gg.view(out_dim, in_dim)
        else:
            G = torch.sigmoid((s - tau_g) / (Tt + 1e-8)).view(out_dim, in_dim).to(scores.dtype)
    
    else:
        # Default to unstructured behavior
        sparsity = module._effective_sparsity(step)
        tau = module._tau_unstructured(scores, sparsity)
        G = torch.sigmoid((scores.float() - tau) / (Tt + 1e-8)).to(scores.dtype)
    
    # β structural mixing (if enabled)
    beta_s = module._beta_structural(step)
    if beta_s > 0.0 and scores.dim() == 2:
        N_s, M_s = int(module.N), int(module.M)
        out_dim_s, in_dim_s = scores.shape
        in_full_s = (in_dim_s // M_s) * M_s
        if in_full_s > 0:
            sc_core = scores.detach().float()[:, :in_full_s]
            groups_s = in_full_s // M_s
            sc_grouped = sc_core.view(out_dim_s, groups_s, M_s)
            topi_s = torch.topk(sc_grouped, k=N_s, dim=-1, largest=True).indices
            target_nm = torch.zeros_like(sc_grouped, dtype=G.dtype)
            target_nm.scatter_(-1, topi_s, 1.0)
            target_nm = target_nm.view(out_dim_s, in_full_s)
            
            if in_full_s < in_dim_s:
                target_full = G.clone()
                target_full[:, :in_full_s] = target_nm
            else:
                target_full = target_nm
            
            G = ((1.0 - beta_s) * G.float() + beta_s * target_full.float()).to(G.dtype)
    
    if frozen.any():
        G = G.clone()
        G[frozen] = module.mask[frozen]
    
    return G


def _apply_mid_penalty_to_gate(module, G, lambda_mid):
    """Apply mid-penalty correction to target gate G.
    
    Mirrors the mid-penalty logic in SparseLinear.update_mask().
    """
    pen_scale = float(module.mask_penalty_lr)
    mode = str(getattr(module.cfg, 'mask_penalty_mode', 'mid') or 'mid')
    
    if mode == 'structured_topn' and module.mask.dim() == 2 and G.dim() == 2:
        N, M = int(module.N), int(module.M)
        out_dim, in_dim = module.mask.shape
        if in_dim % M == 0:
            groups = in_dim // M
            sm = G.detach().float().view(out_dim, groups, M)
            topi = torch.topk(sm, k=N, dim=-1, largest=True).indices
            target = torch.zeros_like(sm, dtype=G.dtype)
            target.scatter_(-1, topi, 1.0)
            target = target.view(out_dim, in_dim)
            delta = (module.mask.to(dtype=G.dtype) - target)
            G = G - (pen_scale * float(lambda_mid)) * delta
        else:
            mode = 'mid'
    
    if mode == 'mid':
        pen_grad = (1.0 - 2.0 * module.mask).to(dtype=G.dtype)
        G = G - (pen_scale * float(lambda_mid)) * pen_grad
    
    return G.clamp(0.0, 1.0)


def _find_fsdp_wrapper_for_module(fsdp_root, target_module):
    """
    Find the FSDP wrapper that directly wraps the given target module.
    In FSDP, each transformer block is typically wrapped as a separate FSDP unit.
    We want to find the smallest FSDP unit that contains our target_module.
    
    Returns the FSDP wrapper, or None if not found.
    """
    if FSDP is None:
        return None
    
    # Strategy: Walk the module tree and find FSDP wrappers that contain target_module
    # We want the innermost (smallest) FSDP wrapper
    
    candidates = []
    
    def _search(module, depth=0):
        if module is target_module:
            return True
        
        contains_target = False
        for child in module.children():
            if _search(child, depth + 1):
                contains_target = True
                break
        
        if contains_target and isinstance(module, FSDP):
            candidates.append((depth, module))
        
        return contains_target
    
    _search(fsdp_root)
    
    if candidates:
        # Return the deepest (innermost) FSDP wrapper
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    # If no FSDP wrapper found, return the root if it's FSDP
    if isinstance(fsdp_root, FSDP):
        return fsdp_root
    
    return None


@torch.no_grad()
def calculate_flip_rate(model):
    raw = get_raw_model(model)
    flipped = 0
    init_flipped = 0
    total = 0
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            flipped += int(getattr(module.weight, "flipped_mask", 0))
            init_flipped += int(getattr(module.weight, "init_flipped_mask", 0))
            total += int(getattr(module.weight, "param_count", module.weight.numel()))
    ratio = flipped / max(total, 1)
    init_ratio = init_flipped / max(total, 1)
    return flipped, ratio, init_flipped, init_ratio


# -----------------------------
# Hutchinson Hessian diagonal
# -----------------------------
def _rademacher_like(t: torch.Tensor) -> torch.Tensor:
    return (torch.randint_like(t, low=0, high=2, dtype=torch.int8).float() * 2 - 1).to(dtype=t.dtype, device=t.device)


def update_hessian_hutchinson(model, loss):
    """
    Update per-layer Hessian diagonal using Hutchinson estimator:
      diag(H) ≈ E[(H v) ⊙ v], v ~ Rademacher
    Must be called before loss.backward() while graph is alive.

    FSDP-safe:
      - In FSDP mode, weights are sharded across ranks, so hvp shapes differ per rank.
      - We skip all_reduce and update only the local shard of hessian_diag.
      - Each rank independently estimates its local portion of the Hessian diagonal.
    """
    raw = get_raw_model(model)

    # Device for DDP tensors (ok_tensor / reductions). Prefer loss.device.
    try:
        device = loss.device
    except Exception:
        device = None
    if device is None:
        try:
            device = next(raw.parameters()).device
        except Exception:
            device = torch.device("cpu")

    params = []
    owners = []
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            params.append(module.weight)
            owners.append(module)
    if not params:
        return

    # DDP/FSDP setup
    dist = None
    world = 1
    rank = 0
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
        else:
            dist = None
    except Exception:
        dist = None

    # Detect FSDP mode: if any weight shape != hessian_diag shape, we're in FSDP sharded mode
    is_fsdp_sharded = False
    for module in owners:
        if module.weight.shape != module.hessian_diag.shape:
            is_fsdp_sharded = True
            break

    # Compute Hutchinson estimate
    hvps = None
    vs = []
    try:
        grads = torch.autograd.grad(
            loss, params,
            create_graph=True, retain_graph=True, allow_unused=True
        )

        dot = None
        for g, p in zip(grads, params):
            if g is None:
                vs.append(None)
                continue
            v = _rademacher_like(p)
            vs.append(v)
            term = (g * v).sum()
            dot = term if dot is None else (dot + term)

        if dot is None:
            # All grads were None; nothing to do
            hvps = [None] * len(params)
        else:
            hvps = torch.autograd.grad(
                dot, params,
                retain_graph=True, allow_unused=True
            )

    except RuntimeError as e:
        raise RuntimeError(
            f"[Hutchinson] Double-backward failed: {e}\n"
            f"This usually means Flash Attention (SDPA) is enabled. "
            f"Set --eager_attention True to use Hutchinson Hessian estimation."
        ) from e

    # Update hessian_diag for each module
    for module, hvp, v in zip(owners, hvps, vs):
        if hvp is None or v is None:
            continue  # Skip modules with no gradient

        # Compute local Hessian diagonal estimate
        diag_est = (hvp * v).detach()

        # Get the beta for EMA
        beta = float(getattr(module.cfg, "beta", 0.99))

        if is_fsdp_sharded:
            # FSDP mode: weight is sharded, shapes don't match hessian_diag
            # We need to update only the corresponding shard of hessian_diag
            # 
            # In FSDP fully_sharded mode:
            # - weight has shape [local_rows, cols] or similar (sharded)
            # - hessian_diag has shape [full_rows, cols] (full size, as it's a frozen buffer)
            # 
            # Strategy: Update hessian_diag locally without all_reduce.
            # Each rank updates its own estimate. The hessian_diag buffer
            # should be kept in sync via FSDP's buffer broadcasting mechanism,
            # or we accept that each rank has its own local estimate.
            #
            # For FSDP with frozen parameters (like hessian_diag), the buffer
            # is typically NOT sharded, so we can't directly index into it.
            # Instead, we'll use a different approach: flatten and match elements.
            
            hvp_flat = hvp.reshape(-1)
            v_flat = v.reshape(-1)
            diag_flat = (hvp_flat * v_flat).detach()
            
            # Clamp to non-negative
            diag_flat = torch.clamp(diag_flat, min=0.0)
            
            # hessian_diag is now a frozen nn.Parameter (not buffer), so FSDP shards it.
            # In FSDP fully_sharded mode, hessian_diag should have the same shape as weight
            # (both are sharded), so we can update directly. However, we keep the fallback
            # logic for safety in case shapes don't match unexpectedly.
            
            if module.hessian_diag.numel() == hvp.numel():
                # Same size - can update directly (both are sharded by FSDP)
                hess_flat = module.hessian_diag.reshape(-1)
                hess_flat.mul_(beta).add_(diag_flat, alpha=1 - beta)
            else:
                # Different sizes - unexpected case (hessian_diag full, weight sharded)
                # This shouldn't happen with our current design (hessian_diag is a frozen Parameter).
                # But keep as fallback for safety.
                if not hasattr(module, '_hessian_diag_local'):
                    # Create a local buffer matching weight shape
                    module._hessian_diag_local = torch.zeros_like(hvp.detach())
                
                local_diag = module._hessian_diag_local
                if local_diag.shape == diag_flat.reshape(hvp.shape).shape:
                    local_diag.mul_(beta).add_(diag_flat.reshape(hvp.shape), alpha=1 - beta)
        else:
            # Non-FSDP mode (DDP or single GPU): shapes match, can use all_reduce
            diag_est = torch.clamp(diag_est, min=0.0)
            
            # All-reduce average across ranks to reduce variance + keep consistent
            if dist is not None and world > 1:
                dist.all_reduce(diag_est, op=dist.ReduceOp.SUM)
                diag_est.div_(world)
            
            # EMA update
            module.hessian_diag.mul_(beta).add_(diag_est, alpha=1 - beta)



# -----------------------------
# (Optional) mid penalty / hardening
# -----------------------------
@torch.no_grad()
def mid_penalty(model, lambda_mid: float = 0.0):
    """
    Encourage mask away from 0.5:
      penalty = mean(mask*(1-mask))
    For continuous-mask pruning, this is the correct proxy (no need to use importance_ema).
    
    FSDP-safe: uses sum/count approach with all_reduce to handle sharded masks.
    """
    raw = get_raw_model(model)
    
    # Get device from model parameters
    device = None
    for p in raw.parameters():
        device = p.device
        break
    if device is None:
        device = torch.device("cpu")
    
    if lambda_mid <= 0:
        return torch.tensor(0.0, device=device)
    
    # Use sum/count approach for FSDP compatibility
    total_penalty = torch.zeros((), device=device, dtype=torch.float64)
    total_count = torch.zeros((), device=device, dtype=torch.float64)
    
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            m = module.mask
            if m.numel() == 0:
                continue
            # Ensure mask is on the correct device
            if m.device != device:
                m = m.to(device)
            # Check for NaN/Inf in mask (can happen with uninitialized FSDP shards)
            if torch.isnan(m).any() or torch.isinf(m).any():
                continue  # Skip this module's contribution
            # Compute penalty: m * (1 - m), which is 0 at m=0 or m=1, max at m=0.5
            local_penalty = (m * (1 - m)).sum()
            total_penalty += local_penalty.double()
            total_count += float(m.numel())
    
    # DDP/FSDP all-reduce to get global sum and count
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(total_penalty, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    except Exception:
        pass
    
    # Compute mean penalty
    count_v = float(total_count.item())
    if count_v <= 0:
        return torch.tensor(0.0, device=device)
    
    mean_penalty = total_penalty / count_v
    result = lambda_mid * mean_penalty
    
    # Final safety check
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=device)
    
    return result.to(dtype=torch.float32)


@torch.no_grad()
def sparsity_penalty(model, target_sparsity: float = 0.5, alpha: float = 0.0):
    """
    Compute global hard sparsity and return (penalty_tensor, current_sparsity_float).
    
    对于 2:4 (structured) 稀疏，使用 hard mask (每4个元素取 top-2) 来计算稀疏度，
    而不是简单的 mask < 0.5 阈值判断。这样计算出的稀疏度更准确地反映实际推理时的稀疏度。
    
    Penalty = alpha * (current_sparsity - target_sparsity)^2.
    This is DDP-safe: counts are all-reduced across ranks.
    """
    raw = get_raw_model(model)
    device = None
    for p in raw.parameters():
        device = p.device
        break
    if device is None:
        device = torch.device("cpu")

    cnt = torch.zeros((), device=device, dtype=torch.float64)
    tot = torch.zeros((), device=device, dtype=torch.float64)

    for _, module in raw.named_modules():
        if not isinstance(module, SparseLinear):
            continue
        m = module.mask.detach()
        if m.numel() == 0:
            continue
        if m.device != device:
            m = m.to(device)
        
        # 获取 mask_type 来决定如何计算稀疏度
        mask_type = getattr(module, 'mask_type', 'unstructured')
        
        if mask_type == 'structured' and m.dim() == 2:
            # 2:4 稀疏: 使用 hard mask (topk) 计算
            # 在每 M=4 个元素的 group 中，bottom-2 被视为 pruned
            N, M = int(getattr(module, 'N', 2)), int(getattr(module, 'M', 4))
            out_dim, in_dim = m.shape
            in_full = (in_dim // M) * M
            if in_full > 0:
                core = m[:, :in_full].float()
                groups = in_full // M
                grouped = core.view(out_dim, groups, M)
                # 每个 group 取 top-N，其余为 pruned
                # 使用 topk 找出每个 group 的 bottom-(M-N) 个元素
                # pruned_count = (M - N) * out_dim * groups
                pruned_per_group = M - N
                tot += float(out_dim * in_full)
                cnt += float(out_dim * groups * pruned_per_group)
                # 尾部部分（不能被 M 整除的）使用阈值法
                if in_full < in_dim:
                    tail = m[:, in_full:]
                    tot += float(tail.numel())
                    cnt += (tail < 0.5).double().sum()
            else:
                # fallback to threshold
                tot += float(m.numel())
                cnt += (m < 0.5).double().sum()
        else:
            # unstructured 或其他类型: 使用 mask < 0.5 阈值
            tot += float(m.numel())
            cnt += (m < 0.5).double().sum()

    # DDP all-reduce
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    except Exception:
        pass

    tot_v = float(tot.item())
    if tot_v <= 0:
        cur = 0.0
    else:
        cur = float((cnt / tot).item())

    pen = torch.tensor(0.0, device=device)
    if alpha != 0.0:
        diff = cur - float(target_sparsity)
        pen = torch.tensor(float(alpha * (diff * diff)), device=device)

    return pen, cur


@torch.no_grad()
def nm_2_4_tile_stats(model, debug=False):
    """
    统计 2:4 稀疏中每个长度为 4 的 tile 里：
    - 较大的 2 个 mask 值的平均值 (top2_avg)
    - 较小的 2 个 mask 值的平均值 (bot2_avg)
    - soft-hard gap: top2_avg 和 1.0 的差距，bot2_avg 和 0.0 的差距
    
    这些指标可以用于诊断 soft mask 是否正确收敛到 hard 2:4 约束。
    理想情况下，top2_avg 应该接近 1.0，bot2_avg 应该接近 0.0。
    
    Returns:
        dict: {
            'top2_avg': float,      # 每个 tile 中较大 2 个 mask 的全局平均值
            'bot2_avg': float,      # 每个 tile 中较小 2 个 mask 的全局平均值
            'top2_gap': float,      # 1.0 - top2_avg，越小越好
            'bot2_gap': float,      # bot2_avg - 0.0，越小越好
            'total_gap': float,     # top2_gap + bot2_gap，总体 soft-hard gap
        }
    """
    raw = get_raw_model(model)
    
    if debug:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[nm_2_4_tile_stats DEBUG] model type: {type(model)}")
            print(f"[nm_2_4_tile_stats DEBUG] raw model type: {type(raw)}")
            # 打印所有模块类型
            module_types = {}
            for name, mod in raw.named_modules():
                t = type(mod).__name__
                module_types[t] = module_types.get(t, 0) + 1
            print(f"[nm_2_4_tile_stats DEBUG] module types: {module_types}")
    
    device = None
    for p in raw.parameters():
        device = p.device
        break
    if device is None:
        device = torch.device("cpu")

    top2_sum = torch.zeros((), device=device, dtype=torch.float64)
    bot2_sum = torch.zeros((), device=device, dtype=torch.float64)
    tile_count = torch.zeros((), device=device, dtype=torch.float64)

    M = 4  # tile size for 2:4 sparsity
    N = 2  # keep top-2
    
    sparse_linear_count = 0
    skipped_masks = []
    processed_count = 0
    for name, module in raw.named_modules():
        if not isinstance(module, SparseLinear):
            continue
        sparse_linear_count += 1
        
        # 统计所有 SparseLinear 层 (调用方已检查 hard_mask_type)
        m = module.mask.detach()
        
        if m.numel() == 0:
            skipped_masks.append((name, m.shape, m.dim()))
            continue
        if m.device != device:
            m = m.to(device)
        
        # FSDP 兼容：不再强制 reshape 为 2D。
        # 2:4 tile 统计只需要沿 in_features 维度每 4 个元素分组。
        # 关键思路：
        #   - 非 FSDP (2D mask): 直接按行的 in_dim 维度分组
        #   - FSDP 分片 (1D mask): mask 按行优先 flatten，连续的 M 个元素
        #     仍属于同一行的同一 tile（只要 in_features 能被 M 整除），
        #     所以直接在 1D 上每 M 个元素分组即可。
        #     各 rank 统计各自的 shard，最后 all_reduce 汇总。
        out_f = getattr(module, 'out_features', None)
        in_f = getattr(module, 'in_features', None)
        
        if m.dim() == 2:
            # 非 FSDP 或未分片：标准 2D 路径
            out_dim, in_dim = m.shape
            in_full = (in_dim // M) * M
            if in_full == 0:
                skipped_masks.append((name, m.shape, m.dim()))
                continue
            core = m[:, :in_full].float()
            groups = in_full // M
            grouped = core.view(out_dim, groups, M)  # (out_dim, groups, M)
        elif m.dim() == 1:
            # FSDP 分片后的 1D mask
            # 先尝试 reshape 回 2D（完整 mask 或整行分片）
            reshaped = False
            if out_f is not None and in_f is not None:
                expected_numel = out_f * in_f
                if m.numel() == expected_numel:
                    m = m.view(out_f, in_f)
                    reshaped = True
                elif m.numel() < expected_numel and m.numel() % in_f == 0:
                    local_out = m.numel() // in_f
                    m = m.view(local_out, in_f)
                    reshaped = True
            
            if reshaped:
                # 成功 reshape 为 2D，走标准路径
                out_dim, in_dim = m.shape
                in_full = (in_dim // M) * M
                if in_full == 0:
                    skipped_masks.append((name, m.shape, m.dim()))
                    continue
                core = m[:, :in_full].float()
                groups = in_full // M
                grouped = core.view(out_dim, groups, M)
            else:
                # 无法 reshape 为 2D（FSDP 混合分片/跨参数边界切割）
                # 直接在 1D 上每 M 个元素分组。
                # 注意：如果 in_features 不能被 M 整除，行尾会有 padding 元素混入，
                # 但对于大模型（in_f 通常是 128 的倍数）这几乎不会发生。
                # 即使有少量边界 tile 跨行，对全局统计的影响也微乎其微。
                flat = m.float()
                n_full = (flat.numel() // M) * M
                if n_full == 0:
                    skipped_masks.append((name, m.shape, m.dim()))
                    continue
                grouped = flat[:n_full].view(-1, M)  # (num_tiles, M)
        else:
            skipped_masks.append((name, m.shape, m.dim()))
            continue
        
        if debug and sparse_linear_count <= 3:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[nm_2_4_tile_stats DEBUG] module: {name}, mask.shape(orig): {module.mask.shape}, "
                      f"grouped.shape: {grouped.shape}, out_f: {out_f}, in_f: {in_f}")
        
        # 排序每个 tile 中的 M 个元素（最后一维）
        sorted_vals, _ = torch.sort(grouped, dim=-1, descending=True)
        # top-2: sorted_vals[..., :N], bot-2: sorted_vals[..., N:]
        top2_vals = sorted_vals[..., :N]
        bot2_vals = sorted_vals[..., N:]
        
        # 累加
        num_tiles = grouped.shape[0] if grouped.dim() == 2 else grouped.shape[0] * grouped.shape[1]
        top2_sum += top2_vals.sum()
        bot2_sum += bot2_vals.sum()
        tile_count += float(num_tiles)
        processed_count += 1

    # FSDP 模式下 mask 是分片参数，各 rank 只持有部分数据，
    # 需要 all_reduce 汇总所有 rank 的统计量。
    # 非 FSDP (DDP) 模式下 mask 在各 rank 是完整副本，也需要 all_reduce 后除以 world_size，
    # 或者只在 rank0 统计。为简化，统一使用 all_reduce。
    try:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        is_fsdp = isinstance(model, FSDP) or (hasattr(model, 'module') and isinstance(model.module, FSDP))
        
        if dist.is_initialized():
            if is_fsdp:
                # FSDP: 各 rank 持有不同的 shard，需要 all_reduce 汇总
                dist.all_reduce(top2_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(bot2_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(tile_count, op=dist.ReduceOp.SUM)
            else:
                # DDP: 各 rank 持有完整副本，不需要 all_reduce（避免重复计数）
                pass
    except Exception:
        pass

    total_tiles = float(tile_count.item())
    if debug:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[nm_2_4_tile_stats DEBUG] sparse_linear_count: {sparse_linear_count}, processed: {processed_count}, total_tiles: {total_tiles}")
            if skipped_masks:
                print(f"[nm_2_4_tile_stats DEBUG] skipped {len(skipped_masks)} masks, first 3: {skipped_masks[:3]}")
    
    if total_tiles <= 0:
        if debug:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[nm_2_4_tile_stats DEBUG] WARNING: total_tiles=0, returning zeros!")
        return {
            'top2_avg': 0.0,
            'bot2_avg': 0.0,
            'top2_gap': 0.0,
            'bot2_gap': 0.0,
            'total_gap': 0.0,
        }

    # 每个 tile 有 N=2 个 top 和 M-N=2 个 bot
    top2_avg = float(top2_sum.item()) / (total_tiles * N)
    bot2_avg = float(bot2_sum.item()) / (total_tiles * (M - N))
    
    top2_gap = 1.0 - top2_avg
    bot2_gap = bot2_avg  # 因为理想值是 0
    total_gap = top2_gap + bot2_gap

    return {
        'top2_avg': top2_avg,
        'bot2_avg': bot2_avg,
        'top2_gap': top2_gap,
        'bot2_gap': bot2_gap,
        'total_gap': total_gap,
    }


@torch.no_grad()
def harden_fraction(model, fraction: float = 0.2, band_low: float = 0.4, band_high: float = 0.6):
    """
    Optional: force a fraction of near-boundary entries to 0/1 and freeze them.
    NOTE: 这会“显式二值化”。你目前 hardening=0，所以不会触发。
    """
    raw = get_raw_model(model)
    candidates = []
    for _, module in raw.named_modules():
        if isinstance(module, SparseLinear):
            m = module.mask
            # frozen_mask_flags 可能是 float dtype（FSDP 要求统一 dtype），
            # 需要先转为 bool 才能用 ~ 操作
            frozen = module.frozen_mask_flags.bool()
            sel = (~frozen) & (m > band_low) & (m < band_high)
            if sel.any():
                diff = (m[sel] - 0.5).abs()
                idx = torch.nonzero(sel, as_tuple=False)
                candidates.append((module, idx, diff))

    if not candidates:
        return 0

    all_diff = torch.cat([c[2] for c in candidates])
    k = int(all_diff.numel() * fraction)
    if k <= 0:
        return 0

    thr = torch.topk(all_diff, k, largest=False).values.max()
    hardened = 0
    for module, idx, diff in candidates:
        pick = diff <= thr
        if pick.any():
            hidx = idx[pick]
            # round and freeze
            v = module.mask[hidx[:, 0], hidx[:, 1]]
            module.mask[hidx[:, 0], hidx[:, 1]] = (v > 0.5).to(module.mask.dtype)
            module.frozen_mask_flags[hidx[:, 0], hidx[:, 1]] = True
            hardened += int(pick.sum().item())
    return hardened


# -----------------------------
# Evaluation / Calibration (WANDA)
# -----------------------------
@torch.no_grad()
def eval_ppl(model, bs=2, device="cuda:0", block_size=1024, model_name_or_path=None):
    """
    Compute perplexity on WikiText-2.
    Auto-detects model type and uses appropriate tokenizer:
    - GPT2-family: tiktoken "gpt2"
    - LLaMA-family: transformers LlamaTokenizer
    
    Args:
        model_name_or_path: Path to model directory for loading tokenizer (e.g., "NousResearch/Llama-2-7b-hf")
    """
    
    # Detect model type and get vocab_size
    # Use get_raw_model to unwrap FSDP/DDP wrappers
    try:
        raw_model = get_raw_model(model)
        config = None
        
        # Try to get config from various model structures
        if hasattr(raw_model, 'config'):
            config = raw_model.config
        elif hasattr(raw_model, 'model') and hasattr(raw_model.model, 'config'):
            # OPTSparse.model = OPTForCausalLM
            config = raw_model.model.config
        
        if config is not None:
            vocab_size = getattr(config, 'vocab_size', 50257)
            model_type = getattr(config, 'model_type', 'unknown')
            # Try to get model path from config
            if model_name_or_path is None:
                model_name_or_path = getattr(config, '_name_or_path', None)
        else:
            vocab_size = 50257
            model_type = 'unknown'
    except Exception as e:
        print(f"[eval_ppl] WARNING: Could not determine model type: {e}")
        return float('inf')
    
    # Load appropriate tokenizer
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "wikitext", "wikitext-2-raw-v1", "wiki.test.raw")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    try:
        model_type_lower = model_type.lower()
        if 'llama' in model_type_lower:
            # LLaMA model: try loading tokenizer from model path
            try:
                from transformers import AutoTokenizer
                tokenizer_path = model_name_or_path or "meta-llama/Llama-2-7b-hf"
                # Handle relative path (local model directory)
                if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                    # Check if it's a HuggingFace model ID (contains '/') or local relative path
                    if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                        # Might be HF model ID, try as-is first
                        pass
                    else:
                        # Try to resolve relative path from script directory
                        resolved_path = os.path.join(base_dir, tokenizer_path)
                        if os.path.exists(resolved_path):
                            tokenizer_path = resolved_path
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                testenc = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e1:
                print(f"[eval_ppl] WARNING: Failed to load LLaMA tokenizer from '{tokenizer_path}': {e1}")
                print(f"[eval_ppl] Skipping WikiText eval. Use validation loss instead.")
                return float('inf')
        elif 'opt' in model_type_lower:
            # OPT model: use OPT tokenizer (GPT2TokenizerFast compatible)
            try:
                from transformers import AutoTokenizer
                tokenizer_path = model_name_or_path or "facebook/opt-2.7b"
                # Handle relative path (local model directory)
                if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                    # Check if it's a HuggingFace model ID (contains '/') or local relative path
                    if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                        # Might be HF model ID like 'facebook/opt-2.7b', try as-is first
                        pass
                    else:
                        # Try to resolve relative path from script directory
                        resolved_path = os.path.join(base_dir, tokenizer_path)
                        if os.path.exists(resolved_path):
                            tokenizer_path = resolved_path
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                testenc = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e1:
                print(f"[eval_ppl] WARNING: Failed to load OPT tokenizer from '{tokenizer_path}': {e1}")
                print(f"[eval_ppl] Skipping WikiText eval. Use validation loss instead.")
                return float('inf')
        elif 'qwen' in model_type_lower:
            # Qwen model: use Qwen tokenizer (requires trust_remote_code=True)
            try:
                from transformers import AutoTokenizer
                tokenizer_path = model_name_or_path or "Qwen/Qwen2-1.5B"
                # Handle relative path (local model directory)
                if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                    # Check if it's a HuggingFace model ID (contains '/') or local relative path
                    if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                        # Might be HF model ID like 'Qwen/Qwen2-1.5B', try as-is first
                        pass
                    else:
                        # Try to resolve relative path from script directory
                        resolved_path = os.path.join(base_dir, tokenizer_path)
                        if os.path.exists(resolved_path):
                            tokenizer_path = resolved_path
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                testenc = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e1:
                print(f"[eval_ppl] WARNING: Failed to load Qwen tokenizer from '{tokenizer_path}': {e1}")
                print(f"[eval_ppl] Skipping WikiText eval. Use validation loss instead.")
                return float('inf')
        elif 'deepseek' in model_type_lower:
            # DeepSeek MoE model: use DeepSeek tokenizer (requires trust_remote_code=True)
            try:
                from transformers import AutoTokenizer
                tokenizer_path = model_name_or_path or "deepseek-ai/deepseek-moe-16b-base"
                # Handle relative path (local model directory)
                if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                    # Check if it's a HuggingFace model ID (contains '/') or local relative path
                    if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                        # Might be HF model ID like 'deepseek-ai/deepseek-moe-16b-base', try as-is first
                        pass
                    else:
                        # Try to resolve relative path from script directory
                        resolved_path = os.path.join(base_dir, tokenizer_path)
                        if os.path.exists(resolved_path):
                            tokenizer_path = resolved_path
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                testenc = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e1:
                print(f"[eval_ppl] WARNING: Failed to load DeepSeek tokenizer from '{tokenizer_path}': {e1}")
                print(f"[eval_ppl] Skipping WikiText eval. Use validation loss instead.")
                return float('inf')
        else:
            # GPT2 or default: use GPT2 tokenizer
            if tiktoken is None:
                raise RuntimeError(
                    "Missing optional dependency 'tiktoken'. Install it (e.g. `pip install tiktoken`) "
                    "or run in an environment that has it before calling eval_ppl() for GPT2 tokenization."
                )
            tokenizer = tiktoken.get_encoding("gpt2")
            testenc = tokenizer.encode_ordinary(text)
    except Exception as e:
        print(f"[eval_ppl] ERROR: Failed to tokenize: {e}")
        return float('inf')
    
    # Validate token range
    testenc_arr = np.array(testenc, dtype=np.int64)
    max_token = testenc_arr.max()
    min_token = testenc_arr.min()
    
    if max_token >= vocab_size or min_token < 0:
        print(f"[eval_ppl] WARNING: Tokens out of range [{min_token}, {max_token}] for vocab_size={vocab_size}")
        print(f"[eval_ppl] Skipping eval due to tokenizer mismatch")
        return float('inf')

    # Resolve device safely. In distributed, prefer the model's parameter device.
    inferred_device = None
    try:
        inferred_device = next(get_raw_model(model).parameters()).device
    except Exception:
        inferred_device = None
    if inferred_device is not None:
        if device is None:
            device = inferred_device
        else:
            try:
                dev = torch.device(device) if not isinstance(device, torch.device) else device
                # If caller passed a mismatched cuda index, override to model device.
                if dev.type == 'cuda' and inferred_device.type == 'cuda' and dev.index != inferred_device.index:
                    device = inferred_device
                else:
                    device = dev
            except Exception:
                device = inferred_device

    model.eval()
    testenc = torch.tensor(testenc, dtype=torch.long, device=device).unsqueeze(0)
    nsamples = testenc.numel() // block_size

    nlls = []
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    from tqdm import tqdm
    pbar = tqdm(total=nsamples, desc="[eval_ppl]", disable=False)
    
    for i in range(0, nsamples, bs):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * block_size):(j * block_size)].to(device)
        inputs = inputs.reshape(j - i, block_size)

        try:
            # 禁用 KV cache，避免 DynamicCache 兼容性问题 (如 DeepSeek-MoE)
            # 注意：某些自定义模型的 forward 不接受 use_cache 参数
            try:
                forward_out = model(inputs, use_cache=False)
            except TypeError as te:
                if 'use_cache' in str(te):
                    # 模型不支持 use_cache 参数，直接调用
                    forward_out = model(inputs)
                else:
                    raise te
            # Handle different output formats:
            # 1. tuple/list: first element is logits
            # 2. ModelOutput (HuggingFace): has .logits attribute
            # 3. raw tensor: use directly
            if isinstance(forward_out, (tuple, list)):
                lm_logits = forward_out[0]
            elif hasattr(forward_out, 'logits'):
                # HuggingFace ModelOutput (e.g., CausalLMOutputWithPast)
                lm_logits = forward_out.logits
            else:
                lm_logits = forward_out
        except Exception as e:
            print(f"[eval_ppl] ERROR in forward: {e}")
            pbar.close()
            model.train()
            torch.cuda.empty_cache()
            return float('inf')
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # 立即转为 float 值并释放中间张量
        nlls.append(loss.float().item())
        total_tokens += int((block_size - 1) * (j - i))
        
        # 及时释放显存
        del shift_logits, shift_labels, loss, inputs, lm_logits, forward_out
        
        pbar.update(j - i)
        
        # 每隔几个 batch 清理一次缓存
        if (i // bs) % 10 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    if total_tokens <= 0:
        ppl_value = float('inf')
    else:
        # nlls 现在是 float 列表
        ppl_value = float(math.exp(sum(nlls) / float(total_tokens)))
    torch.cuda.empty_cache()
    model.train()
    return ppl_value


@torch.no_grad()
def eval_ppl_distributed(model, bs=2, device="cuda:0", block_size=1024, model_name_or_path=None, ptdtype=None):
    """
    Distributed eval_ppl for FSDP/DDP: all ranks participate with data sharding.
    Auto-detects model type and uses appropriate tokenizer:
    - GPT2-family: tiktoken "gpt2"
    - LLaMA-family: transformers LlamaTokenizer
    
    Args:
        model_name_or_path: Path to model directory for loading tokenizer (e.g., "NousResearch/Llama-2-7b-hf")
    """
    
    import torch.distributed as dist
    
    # Check if distributed context is active
    is_distributed = dist.is_available() and dist.is_initialized()
    if not is_distributed:
        # Fallback to single-GPU eval
        return eval_ppl(model, bs=bs, device=device, block_size=block_size, model_name_or_path=model_name_or_path)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # CRITICAL: In FSDP mode, ALL ranks must execute the same code path.
    # No early returns allowed! Use error_flag instead.
    # Initialize error_flag BEFORE any conditional logic to ensure ALL ranks have it
    # Use the LOCAL rank's device to ensure correct placement for distributed collectives
    local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    error_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    error_flag = torch.tensor([0], device=error_device, dtype=torch.int32)
    
    # Detect model type and vocab size
    # Use get_raw_model to unwrap FSDP/DDP wrappers
    try:
        raw_model = get_raw_model(model)
        config = None
        
        # Try to get config from various model structures
        if hasattr(raw_model, 'config'):
            config = raw_model.config
        elif hasattr(raw_model, 'model') and hasattr(raw_model.model, 'config'):
            # OPTSparse.model = OPTForCausalLM
            config = raw_model.model.config
        
        if config is not None:
            vocab_size = getattr(config, 'vocab_size', 50257)
            model_type = getattr(config, 'model_type', 'unknown')
            if model_name_or_path is None:
                model_name_or_path = getattr(config, '_name_or_path', None)
            if rank == 0:
                print(f"[eval_ppl_distributed] Detected model_type='{model_type}', vocab_size={vocab_size}, path={model_name_or_path}")
        else:
            vocab_size = 50257
            model_type = 'unknown'
    except Exception as e:
        # ALL ranks print for diagnosis
        print(f"[eval_ppl_distributed][rank{rank}] WARNING: Could not determine model type: {e}")
        error_flag[0] = 1
    
    # All ranks load test data independently (file I/O overhead is negligible)
    text = None
    if error_flag[0] == 0:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "data", "wikitext", "wikitext-2-raw-v1", "wiki.test.raw")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            if rank == 0:
                print(f"[eval_ppl_distributed] WikiText loaded: {len(text)} chars from {file_path}")
        except Exception as e:
            # ALL ranks print so we can identify which node fails
            print(f"[eval_ppl_distributed][rank{rank}] ERROR: Cannot load WikiText file '{file_path}': {e}")
            error_flag[0] = 1
    
    # Load appropriate tokenizer based on model type
    testenc = None
    if error_flag[0] == 0:
        try:
            model_type_lower = model_type.lower()
            if 'llama' in model_type_lower:
                # LLaMA model: try loading tokenizer from model path
                try:
                    from transformers import AutoTokenizer
                    tokenizer_path = model_name_or_path or "meta-llama/Llama-2-7b-hf"
                    # Handle relative path (local model directory)
                    if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                        if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                            pass  # Might be HF model ID
                        else:
                            resolved_path = os.path.join(base_dir, tokenizer_path)
                            if os.path.exists(resolved_path):
                                tokenizer_path = resolved_path
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                    testenc = tokenizer.encode(text, add_special_tokens=False)
                except Exception as e1:
                    if rank == 0:
                        print(f"[eval_ppl_distributed] WARNING: Failed to load LLaMA tokenizer from '{tokenizer_path}': {e1}")
                        print(f"[eval_ppl_distributed] Skipping WikiText eval. Use validation loss instead.")
                    error_flag[0] = 1
            elif 'opt' in model_type_lower:
                # OPT model: use OPT tokenizer (GPT2TokenizerFast compatible)
                try:
                    from transformers import AutoTokenizer
                    tokenizer_path = model_name_or_path or "facebook/opt-2.7b"
                    # Handle relative path (local model directory)
                    if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                        if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                            pass  # Might be HF model ID like 'facebook/opt-2.7b'
                        else:
                            resolved_path = os.path.join(base_dir, tokenizer_path)
                            if os.path.exists(resolved_path):
                                tokenizer_path = resolved_path
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                    testenc = tokenizer.encode(text, add_special_tokens=False)
                except Exception as e1:
                    if rank == 0:
                        print(f"[eval_ppl_distributed] WARNING: Failed to load OPT tokenizer from '{tokenizer_path}': {e1}")
                        print(f"[eval_ppl_distributed] Skipping WikiText eval. Use validation loss instead.")
                    error_flag[0] = 1
            elif 'qwen' in model_type_lower:
                # Qwen model: use Qwen tokenizer (requires trust_remote_code=True)
                try:
                    from transformers import AutoTokenizer
                    tokenizer_path = model_name_or_path or "Qwen/Qwen2-1.5B"
                    # Handle relative path (local model directory)
                    if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                        if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                            pass  # Might be HF model ID like 'Qwen/Qwen2-1.5B'
                        else:
                            resolved_path = os.path.join(base_dir, tokenizer_path)
                            if os.path.exists(resolved_path):
                                tokenizer_path = resolved_path
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                    testenc = tokenizer.encode(text, add_special_tokens=False)
                except Exception as e1:
                    if rank == 0:
                        print(f"[eval_ppl_distributed] WARNING: Failed to load Qwen tokenizer from '{tokenizer_path}': {e1}")
                        print(f"[eval_ppl_distributed] Skipping WikiText eval. Use validation loss instead.")
                    error_flag[0] = 1
            elif 'deepseek' in model_type_lower:
                # DeepSeek MoE model: use DeepSeek tokenizer (requires trust_remote_code=True)
                try:
                    from transformers import AutoTokenizer
                    tokenizer_path = model_name_or_path or "deepseek-ai/deepseek-moe-16b-base"
                    # Handle relative path (local model directory)
                    if tokenizer_path and not tokenizer_path.startswith('/') and not tokenizer_path.startswith('http'):
                        if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
                            pass  # Might be HF model ID like 'deepseek-ai/deepseek-moe-16b-base'
                        else:
                            resolved_path = os.path.join(base_dir, tokenizer_path)
                            if os.path.exists(resolved_path):
                                tokenizer_path = resolved_path
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=os.path.isdir(tokenizer_path))
                    testenc = tokenizer.encode(text, add_special_tokens=False)
                except Exception as e1:
                    if rank == 0:
                        print(f"[eval_ppl_distributed] WARNING: Failed to load DeepSeek tokenizer from '{tokenizer_path}': {e1}")
                        print(f"[eval_ppl_distributed] Skipping WikiText eval. Use validation loss instead.")
                    error_flag[0] = 1
            else:
                # GPT2 or default: use GPT2 tokenizer
                if tiktoken is None:
                    raise RuntimeError(
                        "Missing optional dependency 'tiktoken'. Install it (e.g. `pip install tiktoken`) "
                        "or run in an environment that has it before calling eval_ppl_distributed() for GPT2 tokenization."
                    )
                tokenizer = tiktoken.get_encoding("gpt2")
                testenc = tokenizer.encode_ordinary(text)
        except Exception as e:
            # ALL ranks print for diagnosis
            print(f"[eval_ppl_distributed][rank{rank}] ERROR: Failed to tokenize: {e}")
            error_flag[0] = 1

    # Resolve device safely. In distributed, prefer the model's parameter device.
    inferred_device = None
    if error_flag[0] == 0:
        try:
            inferred_device = next(get_raw_model(model).parameters()).device
        except Exception:
            inferred_device = None
        if inferred_device is not None:
            if device is None:
                device = inferred_device
            else:
                try:
                    dev = torch.device(device) if not isinstance(device, torch.device) else device
                    if dev.type == 'cuda' and inferred_device.type == 'cuda' and dev.index != inferred_device.index:
                        if rank == 0:
                            print(f"[eval_ppl_distributed] WARNING: overriding device {dev} -> {inferred_device} to match model params")
                        device = inferred_device
                    else:
                        device = dev
                except Exception:
                    device = inferred_device
    
    # CRITICAL: Synchronize error flag across ALL ranks before proceeding
    # This ensures all ranks see the same error state and take the same code path
    local_error = int(error_flag[0].item())
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    
    # If any rank encountered an error, ALL ranks return inf
    if error_flag[0] == 1:
        if rank == 0:
            print(f"[eval_ppl_distributed] ERROR: Some rank had error (local_error={local_error}). Returning inf.")
            print(f"[eval_ppl_distributed] Debug: tiktoken={tiktoken is not None}, text_loaded={text is not None}, testenc_ready={testenc is not None}")
        return float('inf')
    
    testenc = torch.tensor(testenc, dtype=torch.long, device=device).unsqueeze(0)
    nsamples = testenc.numel() // block_size
    
    # Validate token IDs: check for out-of-range values
    # CRITICAL: No early return here! Use error_flag to keep all ranks in sync
    try:
        max_token = testenc.max().item()
        min_token = testenc.min().item()
        if rank == 0:
            print(f"[eval_ppl] token range: [{min_token}, {max_token}], vocab_size: {vocab_size}")
        
        # If tokens exceed vocab size, skip eval - but sync across ALL ranks
        if max_token >= vocab_size or min_token < 0:
            if rank == 0:
                print(f"[eval_ppl] WARNING: tokens out of range [{min_token}, {max_token}] for vocab_size={vocab_size}")
                print(f"[eval_ppl] Skipping eval due to tokenizer mismatch")
            error_flag[0] = 1
    except Exception as e:
        if rank == 0:
            print(f"[eval_ppl] WARNING: failed to validate tokens: {e}")
        error_flag[0] = 1
    
    # Sync token validation error flag
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if error_flag[0] == 1:
        return float('inf')
    
    # CRITICAL FOR FSDP: ALL ranks must execute the SAME number of forward() calls.
    # FSDP's forward() involves all_gather for weight shards, so different iteration
    # counts across ranks will cause collective mismatch.
    # 
    # Solution: ALL ranks process ALL samples (no sharding). Only rank 0 computes PPL.
    # This is slightly less efficient but guarantees correctness with FSDP.
    
    # CRITICAL: Before entering the forward loop, sync to ensure all ranks are in the same state
    dist.barrier()
    
    model.eval()
    nlls = []
    token_sum = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    
    # Progress bar (only on rank 0)
    from tqdm import tqdm
    pbar = tqdm(total=nsamples, desc=f"[eval_ppl rank{rank}]", disable=(rank != 0))
    
    # ALL ranks process ALL samples to maintain FSDP collective consistency
    forward_error_occurred = False
    
    for i in range(0, nsamples, bs):
        if forward_error_occurred:
            # Even on error, we must continue the loop to match other ranks' forward() calls
            # Just skip the actual computation
            pbar.update(min(bs, nsamples - i))
            continue
        
        j = min(i + bs, nsamples)
        if j <= i:
            break
        
        inputs = testenc[:, (i * block_size):(j * block_size)].to(device)
        inputs = inputs.reshape(j - i, block_size)
        
        # Additional validation before forward
        if inputs.max() >= vocab_size or inputs.min() < 0:
            if rank == 0:
                print(f"[eval_ppl] ERROR: batch inputs out of range: [{inputs.min().item()}, {inputs.max().item()}]")
            forward_error_occurred = True
            del inputs
            pbar.update(j - i)
            continue  # Continue loop to maintain collective consistency
        
        try:
            # 禁用 KV cache，避免 DynamicCache 兼容性问题 (如 DeepSeek-MoE)
            # 注意：某些自定义模型的 forward 不接受 use_cache 参数
            # 在 FSDP MixedPrecision 下，模型参数可能是 bfloat16/float16，
            # 但 embedding 输出是 float32，需要 autocast 保证 dtype 一致
            import contextlib
            _autocast_dtype = None
            # 方法0: 使用调用方显式传入的 ptdtype（最可靠）
            if ptdtype is not None and ptdtype in (torch.bfloat16, torch.float16):
                _autocast_dtype = ptdtype
            # 方法1: 从 FSDP 对象检测 mixed_precision 配置
            if _autocast_dtype is None:
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    _fsdp_obj = None
                    if isinstance(model, FSDP):
                        _fsdp_obj = model
                    elif hasattr(model, 'module') and isinstance(model.module, FSDP):
                        _fsdp_obj = model.module
                    if _fsdp_obj is not None and hasattr(_fsdp_obj, 'mixed_precision') and _fsdp_obj.mixed_precision is not None:
                        _mp = _fsdp_obj.mixed_precision
                        if hasattr(_mp, 'param_dtype') and _mp.param_dtype is not None:
                            _autocast_dtype = _mp.param_dtype
                except Exception:
                    pass
            # 方法2: 从模型参数的实际 dtype 检测（fallback）
            if _autocast_dtype is None:
                try:
                    _param_dtype = next(get_raw_model(model).parameters()).dtype
                    if _param_dtype in (torch.bfloat16, torch.float16):
                        _autocast_dtype = _param_dtype
                except Exception:
                    pass
            _use_autocast = _autocast_dtype is not None
            if _use_autocast and i == 0 and rank == 0:
                print(f"[eval_ppl] Using autocast with dtype={_autocast_dtype} for FSDP MixedPrecision")
            _autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=_autocast_dtype) if _use_autocast else contextlib.nullcontext()
            
            with _autocast_ctx:
                try:
                    forward_out = model(inputs, use_cache=False)
                except TypeError as te:
                    if 'use_cache' in str(te):
                        # 模型不支持 use_cache 参数，直接调用
                        forward_out = model(inputs)
                    else:
                        raise te
            # 处理不同的输出格式：
            # 1. tuple/list: 取第一个元素 (自定义模型)
            # 2. CausalLMOutput 等 HuggingFace 输出: 取 .logits 属性
            # 3. 直接是 tensor: 直接使用
            if isinstance(forward_out, (tuple, list)):
                lm_logits = forward_out[0]
            elif hasattr(forward_out, 'logits'):
                lm_logits = forward_out.logits
            else:
                lm_logits = forward_out
        except Exception as e:
            # CRITICAL: In FSDP mode, we CANNOT break here - must continue to match other ranks
            print(f"[eval_ppl][rank{rank}] forward error at sample {i}: {e}")
            forward_error_occurred = True
            lm_logits = None
            del inputs
            pbar.update(j - i)
            continue  # Continue loop to maintain collective consistency
        
        # Only rank 0 accumulates loss (to avoid redundant computation)
        # All ranks execute forward() for FSDP consistency, but only rank 0 computes loss
        if rank == 0 and lm_logits is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            # 立即转为 float 并释放中间张量
            nlls.append(loss.float().item())
            token_sum += int((block_size - 1) * (j - i))
            del shift_logits, shift_labels, loss
        
        # 及时释放显存
        del inputs, lm_logits, forward_out
        
        pbar.update(j - i)
        
        # 每隔几个 batch 清理一次缓存
        if (i // bs) % 10 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    
    # CRITICAL: After the loop, sync error status across all ranks
    # Create the error tensor AFTER the loop, so all ranks create it at the same point
    forward_error = torch.tensor([1 if forward_error_occurred else 0], device=device, dtype=torch.int32)
    
    # Barrier to ensure all ranks have exited the loop
    try:
        dist.barrier()
    except Exception as e:
        if rank == 0:
            print(f"[eval_ppl] ERROR: Barrier failed: {e}")
        torch.cuda.empty_cache()
        model.train()
        return float('inf')
    
    # Check if any rank encountered a forward error
    dist.all_reduce(forward_error, op=dist.ReduceOp.MAX)
    if forward_error[0] == 1:
        if rank == 0:
            print(f"[eval_ppl] Forward errors detected on one or more ranks. Returning inf.")
        torch.cuda.empty_cache()
        model.train()
        return float('inf')
    
    # Compute PPL on rank 0 only (since only rank 0 accumulated loss)
    if rank == 0:
        if len(nlls) > 0 and token_sum > 0:
            # nlls 现在是 float 列表，直接求和
            total_loss = sum(nlls)
            ppl_value = float(math.exp(total_loss / float(token_sum)))
        else:
            ppl_value = float('inf')
    else:
        ppl_value = 0.0  # Will be overwritten by broadcast
    
    # Broadcast PPL from rank 0 to all ranks
    ppl_tensor = torch.tensor([ppl_value], device=device, dtype=torch.float32)
    dist.broadcast(ppl_tensor, src=0)
    
    torch.cuda.empty_cache()
    model.train()
    
    # Return on all ranks (same value)
    return float(ppl_tensor.item())


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def prepare_calibration_input(model, dataloader, device, nsamples):
    layers = model.transformer.h
    model = model.to(device)
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((nsamples, model.config.block_size, model.config.n_embd), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    return inps, outs


@torch.no_grad()
def add_calibration(model, nsamples=128, device="cuda",
                    calib_source="c4", synthetic_cache_path=None):
    """
    Only needed if mask_metric == 'wanda'.

    Args:
        calib_source: "c4" (默认，原始 C4 校准集) 或 "synthetic" (模型续写+PPL过滤)
        synthetic_cache_path: 当 calib_source="synthetic" 时，指定 .pkl 缓存路径。
            若为 None 则自动搜索 data/synthetic_calibration/ 下最新的缓存文件。
    """
    model = get_raw_model(model)
    for _, module in model.named_modules():
        if isinstance(module, SparseLinear):
            # initialize scaler_row to ones
            module.scaler_row = torch.ones_like(module.scaler_row, device=device)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if calib_source == "synthetic":
        # 加载 synthetic calibration 数据
        if synthetic_cache_path and os.path.exists(synthetic_cache_path):
            file_path = synthetic_cache_path
        else:
            # 自动搜索最新的缓存文件
            syn_dir = os.path.join(base_dir, "data", "synthetic_calibration")
            if not os.path.isdir(syn_dir):
                raise FileNotFoundError(
                    f"Synthetic calibration 目录不存在: {syn_dir}\n"
                    f"请先运行: python prepare_synthetic_calibration.py --model_path <model>"
                )
            pkl_files = sorted(
                [f for f in os.listdir(syn_dir) if f.endswith(".pkl")],
                key=lambda f: os.path.getmtime(os.path.join(syn_dir, f)),
                reverse=True,
            )
            if not pkl_files:
                raise FileNotFoundError(
                    f"在 {syn_dir} 中找不到 .pkl 缓存文件。\n"
                    f"请先运行: python prepare_synthetic_calibration.py --model_path <model>"
                )
            file_path = os.path.join(syn_dir, pkl_files[0])
        print(f"[add_calibration] 使用 synthetic 校准集: {file_path}")
    else:
        # 默认: 原始 C4 校准集
        file_path = os.path.join(base_dir, "data", "c4_dataset", "calibration_dataset.pkl")

    with open(file_path, "rb") as f:
        dataloader = pickle.load(f)

    inps, outs = prepare_calibration_input(model, dataloader, device, nsamples)
    layers = model.transformer.h

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[SparseLinear])

        def add_batch(name):
            def tmp(_, inp, out):
                subset[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        for h in handles:
            h.remove()

        inps, outs = outs, inps

    torch.cuda.empty_cache()
@torch.no_grad()
def mask_stats(model, bins: int = 100, eps: float = 1e-8,
               sample_per_layer: int = 4096):
    """
    Aggregate continuous mask statistics across all SparseLinear layers (DDP-safe).
    - Computes global mean/std/entropy and fractions (<0.5/<0.1/<0.01).
    - Approximates p10/p50/p90 via global histogram (all-reduced).

    Important:
      - Uses dist.all_reduce internally -> must be called on ALL ranks.
    """
    import math
    raw = get_raw_model(model)

    device = None
    for p in raw.parameters():
        device = p.device
        break
    if device is None:
        device = torch.device("cpu")

    # local accumulators (tensors on device for all_reduce)
    tot = torch.zeros((), device=device, dtype=torch.float64)
    sum_m = torch.zeros((), device=device, dtype=torch.float64)
    sum_lt05 = torch.zeros((), device=device, dtype=torch.float64)
    sum_gt05 = torch.zeros((), device=device, dtype=torch.float64)
    sum_m2 = torch.zeros((), device=device, dtype=torch.float64)
    sum_ent = torch.zeros((), device=device, dtype=torch.float64)
    cnt_lt05 = torch.zeros((), device=device, dtype=torch.float64)
    cnt_lt01 = torch.zeros((), device=device, dtype=torch.float64)
    cnt_lt001 = torch.zeros((), device=device, dtype=torch.float64)
    cnt_gt90 = torch.zeros((), device=device, dtype=torch.float64)
    cnt_gt99 = torch.zeros((), device=device, dtype=torch.float64)
    cnt_gt05 = torch.zeros((), device=device, dtype=torch.float64)

    # block_sparse 统计：较小一半 block 的均值 vs 较大一半 block 的均值
    blk_total_blocks = torch.zeros((), device=device, dtype=torch.float64)
    blk_sum_lower_half = torch.zeros((), device=device, dtype=torch.float64)
    blk_cnt_lower_half = torch.zeros((), device=device, dtype=torch.float64)
    blk_sum_upper_half = torch.zeros((), device=device, dtype=torch.float64)
    blk_cnt_upper_half = torch.zeros((), device=device, dtype=torch.float64)

    hist = torch.zeros((bins,), device=device, dtype=torch.float64)

    for _, module in raw.named_modules():
        if not isinstance(module, SparseLinear):
            continue
        m = module.mask.detach()
        if m.numel() == 0:
            continue
        if m.device != device:
            m = m.to(device)

        n = float(m.numel())
        tot += n
        sum_m += m.double().sum()
        sum_m2 += (m.double() * m.double()).sum()

        mm = m.clamp(eps, 1 - eps).double()
        ent = (-(mm * mm.log()) - ((1 - mm) * (1 - mm).log())).sum()
        sum_ent += ent

        sel_lt05 = (m < 0.5)
        sel_gt05 = (m > 0.5)
        cnt_lt05 += sel_lt05.double().sum()
        cnt_lt01 += (m < 0.1).double().sum()
        cnt_lt001 += (m < 0.01).double().sum()
        cnt_gt90 += (m > 0.9).double().sum()
        cnt_gt99 += (m > 0.99).double().sum()
        cnt_gt05 += (m > 0.5).double().sum()
        if sel_lt05.any():
            sum_lt05 += m[sel_lt05].double().sum()
        if sel_gt05.any():
            sum_gt05 += m[sel_gt05].double().sum()

        # histogram sampling (avoid full flatten cost)
        flat = m.reshape(-1)
        take = min(flat.numel(), int(sample_per_layer))
        if take > 0:
            idx = torch.randint(0, flat.numel(), (take,), device=device)
            samp = flat[idx].float()
            # histc returns float32; cast to float64 for stable reductions
            h = torch.histc(samp, bins=bins, min=0.0, max=1.0).double()
            hist += h

        # ── Block-level 统计（仅 block_sparse 模式） ──────────────────
        # 将 mask 按 block 取均值后，统计较小一半 block 和较大一半 block 的平均 mask 值。
        # 用于观察 mask 是否正确地向 0/1 两极拟合。
        hard_type = str(getattr(module, 'hard_mask_type', 'match') or 'match')
        if hard_type in ('block_sparse16', 'block_sparse32') and m.dim() == 2:
            blk_bs = 16 if hard_type == 'block_sparse16' else 32
            blk_out, blk_in = m.shape
            blk_out_full = (blk_out // blk_bs) * blk_bs
            blk_in_full = (blk_in // blk_bs) * blk_bs
            if blk_out_full > 0 and blk_in_full > 0:
                blk_core = m[:blk_out_full, :blk_in_full].float()
                blk_ob = blk_out_full // blk_bs
                blk_ib = blk_in_full // blk_bs
                blk_tiles = blk_core.view(blk_ob, blk_bs, blk_ib, blk_bs)
                blk_means = blk_tiles.mean(dim=(1, 3))  # (blk_ob, blk_ib)
                blk_flat = blk_means.flatten()  # 所有 block 的 mask 均值
                n_blocks = blk_flat.numel()
                blk_total_blocks += n_blocks
                # 按 mask 值排序，分成较小一半和较大一半
                sorted_blk, _ = blk_flat.sort()
                half = n_blocks // 2
                if half > 0:
                    lower_half = sorted_blk[:half]
                    upper_half = sorted_blk[half:]
                    blk_sum_lower_half += lower_half.double().sum()
                    blk_cnt_lower_half += half
                    blk_sum_upper_half += upper_half.double().sum()
                    blk_cnt_upper_half += (n_blocks - half)

    # DDP all_reduce
    dist = None
    world = 1
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            world = dist.get_world_size()
        else:
            dist = None
    except Exception:
        dist = None

    if dist is not None and world > 1:
        to_reduce = [
            tot, sum_m, sum_m2, sum_ent,
            cnt_lt05, cnt_lt01, cnt_lt001,
            cnt_gt90, cnt_gt99, cnt_gt05,
            sum_lt05, sum_gt05,
            blk_total_blocks, blk_sum_lower_half, blk_cnt_lower_half,
            blk_sum_upper_half, blk_cnt_upper_half,
        ]
        for t in to_reduce:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)

    tot_v = float(tot.item())
    if tot_v <= 0:
        return {
            "mask_mean": 1.0, "mask_std": 0.0, "mask_entropy": 0.0,
            "mask_hard_sparsity@0.5": 0.0, "mask_frac_lt_0.1": 0.0, "mask_frac_lt_0.01": 0.0,
            "mask_p10": 1.0, "mask_p50": 1.0, "mask_p90": 1.0,
        }

    mean = float((sum_m / tot).item())
    var = float((sum_m2 / tot).item()) - mean * mean
    std = math.sqrt(max(var, 0.0))

    entropy = float((sum_ent / tot).item())
    hard_sparsity = float((cnt_lt05 / tot).item())
    frac_lt01 = float((cnt_lt01 / tot).item())
    frac_lt001 = float((cnt_lt001 / tot).item())
    frac_gt90 = float((cnt_gt90 / tot).item())
    frac_gt99 = float((cnt_gt99 / tot).item())
    frac_gt05 = float((cnt_gt05 / tot).item())
    mean_lt05 = float((sum_lt05 / cnt_lt05).item()) if float(cnt_lt05.item()) > 0 else 0.0
    mean_gt05 = float((sum_gt05 / cnt_gt05).item()) if float(cnt_gt05.item()) > 0 else 0.0

    # quantiles from histogram
    def _q_from_hist(hist64: torch.Tensor, q: float) -> float:
        hsum = float(hist64.sum().item())
        if hsum <= 0:
            return mean
        cdf = (hist64.cumsum(0) / (hist64.sum() + 1e-12)).cpu()
        # first bin where cdf >= q
        idx = int(torch.searchsorted(cdf, torch.tensor([q])).item())
        idx = max(0, min(bins - 1, idx))
        # map bin index to value in [0,1]
        # use bin center
        return float((idx + 0.5) / bins)

    p10 = _q_from_hist(hist, 0.10)
    p50 = _q_from_hist(hist, 0.50)
    p90 = _q_from_hist(hist, 0.90)

    # block_sparse 统计：较小一半 / 较大一半 block 的平均 mask 值
    blk_total_v = float(blk_total_blocks.item())
    blk_lower_mean = (
        float((blk_sum_lower_half / blk_cnt_lower_half).item())
        if float(blk_cnt_lower_half.item()) > 0 else 0.0
    )
    blk_upper_mean = (
        float((blk_sum_upper_half / blk_cnt_upper_half).item())
        if float(blk_cnt_upper_half.item()) > 0 else 0.0
    )

    result = {
        "mask_mean": mean,
        "mask_std": std,
        "mask_entropy": entropy,
        "mask_hard_sparsity@0.5": hard_sparsity,
        "mask_frac_lt_0.1": frac_lt01,
        "mask_frac_lt_0.01": frac_lt001,
        "mask_frac_gt_0.9": frac_gt90,
        "mask_frac_gt_0.99": frac_gt99,
        "mask_frac_gt_0.5": frac_gt05,
        "mask_mean_lt_0.5": mean_lt05,
        "mask_mean_gt_0.5": mean_gt05,
        "mask_p10": p10,
        "mask_p50": p50,
        "mask_p90": p90,
    }
    # 仅当存在 block_sparse 层时才添加 block 统计指标
    if blk_total_v > 0:
        result["blk_lower_half_mean"] = blk_lower_mean
        result["blk_upper_half_mean"] = blk_upper_mean
        result["blk_total_blocks"] = blk_total_v
        result["blk_separation"] = blk_upper_mean - blk_lower_mean  # 分离度：越接近 1.0 越好
    return result


@torch.no_grad()
def log_mask_stats(model, step: int, wandb_run=None, prefix: str = "mask/",
                   bins: int = 100, sample_per_layer: int = 4096):
    """
    Rank0-only logging, but stats are GLOBAL (DDP all-reduced inside mask_stats()).
    Must be called on ALL ranks (due to all_reduce).
    """
    # compute global stats (all ranks participate)
    stats = mask_stats(model, bins=bins, sample_per_layer=sample_per_layer)
    log_dict = {prefix + k: v for k, v in stats.items()}
    log_dict["iter"] = step

    # Extract schedule scalars (beta_structural, hardening_x) from first SparseLinear
    raw_model = get_raw_model(model)
    for _, module in raw_model.named_modules():
        if isinstance(module, SparseLinear):
            log_dict[prefix + "beta_structural"] = float(module._beta_structural(step))
            log_dict[prefix + "hardening_x"] = float(getattr(module, 'hardening_x', 1.0))
            break

    # rank0 check for emitting logs
    is_rank0 = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            is_rank0 = (dist.get_rank() == 0)
    except Exception:
        pass
    if not is_rank0:
        return

    if wandb_run is not None:
        # supports both wandb module (wandb.log) and run object (run.log)
        if hasattr(wandb_run, "log"):
            try:
                wandb_run.log(log_dict, step=step)
            except TypeError:
                # some wandb Run objects may not accept step kw; fallback to no-step
                wandb_run.log(log_dict)
        else:
            try:
                import wandb
                try:
                    wandb.log(log_dict, step=step)
                except TypeError:
                    wandb.log(log_dict)
            except Exception:
                print(f"[mask_stats step={step}] {log_dict}")
    # else:
    #     print(
    #         f"[mask_stats step={step}] "
    #         + " ".join([f"{k}={v:.4f}" for k, v in stats.items()])
    #     )
