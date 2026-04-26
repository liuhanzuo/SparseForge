#!/usr/bin/env python3
"""
检测 checkpoint 中的稀疏度状态：
1. Mask 分布分析：mask 是否已经二值化？mask 的稀疏度是多少？
2. Weight 分布分析：weight 中零元素比例（判断 mask 是否已被 apply 到 weight 上）
3. 综合诊断：模型的真实稀疏状态

用法:
    python check_sparsity.py <ckpt_path>
    python check_sparsity.py out_llama/.../model_best.pt
"""

import torch
import sys
import os
import numpy as np
from collections import defaultdict

def analyze_checkpoint(ckpt_path: str):
    print(f"{'='*70}")
    print(f"  Sparsity & Weight Analysis Tool")
    print(f"{'='*70}")
    print(f"\nLoading checkpoint: {ckpt_path}")
    print(f"File size: {os.path.getsize(ckpt_path) / 1024**3:.2f} GB")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 获取 state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 获取训练参数
    args_dict = checkpoint.get('args', {})
    print(f"\n{'='*70}")
    print(f"  Training Args")
    print(f"{'='*70}")
    for key in ['sparsity_ratio', 'mask_type', 'hard_mask_type', 'mask_metric',
                 'SLoRB', 'SLoRB_k', 'SLoRB_init_type',
                 'total_iters', 'hardening_start_iter', 'hardening_end_iter',
                 'retrain_start_iter']:
        val = args_dict.get(key, 'N/A')
        print(f"  {key}: {val}")

    # 训练进度
    iter_num = checkpoint.get('iter_num', 'N/A')
    eval_count = checkpoint.get('eval_count', 'N/A')
    best_wiki_ppl = checkpoint.get('best_wiki_ppl', 'N/A')
    print(f"\n  iter_num (saved): {iter_num}")
    print(f"  eval_count: {eval_count}")
    print(f"  best_wiki_ppl: {best_wiki_ppl}")

    # ===== 分析所有 mask =====
    mask_keys = sorted([k for k in state_dict.keys() if k.endswith('.mask')])
    weight_keys = sorted([k for k in state_dict.keys() if k.endswith('.weight') and not 'SLoRB' in k and not 'x_proj' in k])

    # 找到 mask 对应的 weight key
    mask_to_weight = {}
    for mk in mask_keys:
        # mask key: model.model.layers.0.self_attn.q_proj.mask
        # weight key: model.model.layers.0.self_attn.q_proj.weight
        wk = mk.rsplit('.mask', 1)[0] + '.weight'
        if wk in state_dict:
            mask_to_weight[mk] = wk

    print(f"\n{'='*70}")
    print(f"  Mask Analysis ({len(mask_keys)} layers)")
    print(f"{'='*70}")

    total_mask_zeros = 0
    total_mask_elements = 0
    total_mask_binary = 0
    total_mask_soft = 0
    mask_sparsities = []

    for i, mk in enumerate(mask_keys):
        mask = state_dict[mk].float()
        n_elements = mask.numel()
        n_zeros = (mask == 0).sum().item()
        n_ones = (mask == 1).sum().item()
        n_binary = n_zeros + n_ones
        n_soft = n_elements - n_binary

        sparsity = n_zeros / n_elements * 100
        binary_ratio = n_binary / n_elements * 100

        # 用阈值 0.5 计算 hard mask 后的稀疏度
        hard_zeros = (mask <= 0.5).sum().item()
        hard_sparsity = hard_zeros / n_elements * 100

        total_mask_zeros += n_zeros
        total_mask_elements += n_elements
        total_mask_binary += n_binary
        total_mask_soft += n_soft
        mask_sparsities.append(sparsity)

        if i < 3 or i == len(mask_keys) - 1:
            print(f"\n  [{i}] {mk}")
            print(f"    Shape: {list(mask.shape)}, Elements: {n_elements:,}")
            print(f"    Exact 0: {n_zeros:,} ({sparsity:.2f}%) | Exact 1: {n_ones:,} ({n_ones/n_elements*100:.2f}%)")
            print(f"    Binary: {binary_ratio:.2f}% | Soft (0<x<1): {n_soft:,} ({n_soft/n_elements*100:.2f}%)")
            print(f"    Hard mask sparsity (threshold>0.5): {hard_sparsity:.2f}%")
            print(f"    Min={mask.min().item():.6f} Max={mask.max().item():.6f} Mean={mask.mean().item():.6f}")
            if n_soft > 0:
                soft_vals = mask[(mask > 0) & (mask < 1)]
                if len(soft_vals) > 0:
                    print(f"    Soft values: min={soft_vals.min().item():.6f} max={soft_vals.max().item():.6f} median={soft_vals.median().item():.6f}")
        elif i == 3:
            print(f"\n  ... (skipping {len(mask_keys) - 4} intermediate layers) ...")

    overall_mask_sparsity = total_mask_zeros / total_mask_elements * 100 if total_mask_elements > 0 else 0
    overall_binary_ratio = total_mask_binary / total_mask_elements * 100 if total_mask_elements > 0 else 0

    print(f"\n  --- Mask Summary ---")
    print(f"  Total mask elements: {total_mask_elements:,}")
    print(f"  Total exact zeros: {total_mask_zeros:,} ({overall_mask_sparsity:.2f}%)")
    print(f"  Binary ratio: {overall_binary_ratio:.2f}%")
    print(f"  Non-binary (soft) values: {total_mask_soft:,}")

    # ===== 分析 Weight 的零元素分布 =====
    print(f"\n{'='*70}")
    print(f"  Weight Zero-Element Analysis ({len(mask_to_weight)} layers with mask)")
    print(f"{'='*70}")

    total_w_zeros = 0
    total_w_elements = 0
    total_w_near_zeros = 0  # |w| < 1e-7
    weight_sparsities = []

    # 同时检测 mask 与 weight 零元素的对应关系
    mask_applied_count = 0
    mask_not_applied_count = 0

    for i, (mk, wk) in enumerate(mask_to_weight.items()):
        mask = state_dict[mk].float()
        weight = state_dict[wk].float()

        w_elements = weight.numel()
        w_zeros = (weight == 0).sum().item()
        w_near_zeros = (weight.abs() < 1e-7).sum().item()
        w_sparsity = w_zeros / w_elements * 100

        # 计算 mask==0 对应位置的 weight 是否也为 0
        mask_zero_positions = (mask == 0)
        mask_zeros_count = mask_zero_positions.sum().item()

        if mask_zeros_count > 0:
            weight_at_mask_zeros = weight[mask_zero_positions]
            weight_zeros_at_mask_zeros = (weight_at_mask_zeros == 0).sum().item()
            overlap_ratio = weight_zeros_at_mask_zeros / mask_zeros_count * 100
        else:
            overlap_ratio = -1  # mask 没有零元素

        # 计算 mask==1 对应位置，weight 为 0 的数量（不应该有）
        mask_one_positions = (mask == 1)
        mask_ones_count = mask_one_positions.sum().item()
        if mask_ones_count > 0:
            weight_zeros_at_mask_ones = (weight[mask_one_positions] == 0).sum().item()
            false_zero_ratio = weight_zeros_at_mask_ones / mask_ones_count * 100
        else:
            false_zero_ratio = -1

        total_w_zeros += w_zeros
        total_w_elements += w_elements
        total_w_near_zeros += w_near_zeros
        weight_sparsities.append(w_sparsity)

        if i < 3 or i == len(mask_to_weight) - 1:
            print(f"\n  [{i}] {wk}")
            print(f"    Shape: {list(weight.shape)}, Elements: {w_elements:,}")
            print(f"    Weight exact zeros: {w_zeros:,} ({w_sparsity:.2f}%)")
            print(f"    Weight near-zeros (|w|<1e-7): {w_near_zeros:,} ({w_near_zeros/w_elements*100:.2f}%)")
            if mask_zeros_count > 0:
                print(f"    Mask zeros: {mask_zeros_count:,} | Weight=0 at mask=0 positions: {weight_zeros_at_mask_zeros:,} ({overlap_ratio:.1f}%)")
            else:
                print(f"    Mask has NO exact zeros (mask not yet binary)")
            if mask_ones_count > 0 and false_zero_ratio > 0:
                print(f"    ⚠️ Weight=0 at mask=1 positions: {weight_zeros_at_mask_ones:,} ({false_zero_ratio:.2f}%)")
        elif i == 3:
            print(f"\n  ... (skipping {len(mask_to_weight) - 4} intermediate layers) ...")

    overall_w_sparsity = total_w_zeros / total_w_elements * 100 if total_w_elements > 0 else 0
    overall_w_near_zero = total_w_near_zeros / total_w_elements * 100 if total_w_elements > 0 else 0

    print(f"\n  --- Weight Summary ---")
    print(f"  Total weight elements: {total_w_elements:,}")
    print(f"  Total exact zeros: {total_w_zeros:,} ({overall_w_sparsity:.2f}%)")
    print(f"  Total near-zeros (|w|<1e-7): {total_w_near_zeros:,} ({overall_w_near_zero:.2f}%)")

    # ===== SLoRB 参数检查 =====
    slorb_keys = [k for k in state_dict.keys() if 'SLoRB' in k or 'x_proj' in k]
    if slorb_keys:
        print(f"\n{'='*70}")
        print(f"  SLoRB Parameters ({len(slorb_keys)} tensors)")
        print(f"{'='*70}")
        total_slorb_params = 0
        for k in slorb_keys[:4]:
            t = state_dict[k]
            total_slorb_params += t.numel()
            print(f"  {k}: shape={list(t.shape)}, dtype={t.dtype}, params={t.numel():,}")
        if len(slorb_keys) > 4:
            for k in slorb_keys[4:]:
                total_slorb_params += state_dict[k].numel()
            print(f"  ... ({len(slorb_keys) - 4} more)")
        print(f"  Total SLoRB params: {total_slorb_params:,} ({total_slorb_params/1e6:.2f}M)")

    # ===== 综合诊断 =====
    print(f"\n{'='*70}")
    print(f"  Diagnosis")
    print(f"{'='*70}")

    target_sparsity = args_dict.get('sparsity_ratio', 0.5)
    target_pct = target_sparsity * 100

    # 情况判定
    if overall_binary_ratio >= 99.0 and abs(overall_mask_sparsity - target_pct) < 5:
        # Mask 是二值的，且稀疏度接近目标
        if overall_w_sparsity >= target_pct - 5:
            print(f"  ✅ [CASE A] Mask 已二值化, 稀疏度正常 ({overall_mask_sparsity:.1f}%)")
            print(f"     Weight 零元素 {overall_w_sparsity:.1f}% → Mask 已被 apply 到 weight")
            print(f"     模型真实稀疏度: ~{target_pct:.0f}% ✓")
        else:
            print(f"  ✅ [CASE B] Mask 已二值化, 稀疏度正常 ({overall_mask_sparsity:.1f}%)")
            print(f"     Weight 零元素 {overall_w_sparsity:.1f}% → Mask 尚未 apply 到 weight")
            print(f"     模型真实稀疏度: ~{target_pct:.0f}% (通过 mask 实现) ✓")

    elif overall_binary_ratio >= 99.0 and overall_mask_sparsity < 5:
        # Mask 二值但全是 1
        if overall_w_sparsity >= target_pct - 5:
            print(f"  ✅ [CASE C] Mask 全为1, 但 Weight 已有 {overall_w_sparsity:.1f}% 零元素")
            print(f"     → Mask 已被 apply 到 weight, 然后 mask 被重置为全1")
            print(f"     → 这是 finalized checkpoint 的典型特征!")
            print(f"     模型真实稀疏度: ~{overall_w_sparsity:.0f}% (存在于 weight 中) ✓")
        else:
            print(f"  ❌ [CASE D] Mask 全为1, Weight 零元素仅 {overall_w_sparsity:.1f}%")
            print(f"     → 稀疏化可能未生效或训练异常")

    elif overall_binary_ratio < 99.0:
        # Mask 不是二值的（还在 soft mask 阶段）
        hard_sparsity_total = 0
        for mk in mask_keys:
            m = state_dict[mk].float()
            hard_sparsity_total += (m <= 0.5).sum().item()
        hard_sparsity_pct = hard_sparsity_total / total_mask_elements * 100

        if overall_w_sparsity >= target_pct - 5:
            print(f"  ⚠️ [CASE E] Mask 仍为软值 (binary {overall_binary_ratio:.1f}%), 但 Weight 已有 {overall_w_sparsity:.1f}% 零元素")
            print(f"     → Mask 可能在训练中被 apply 到了 weight, 然后继续更新 soft mask")
            print(f"     → 阈值化后 mask 稀疏度: {hard_sparsity_pct:.1f}%")
            print(f"     模型真实稀疏度: ~{overall_w_sparsity:.0f}% (存在于 weight 中)")
        else:
            print(f"  ⚠️ [CASE F] Mask 仍为软值 (binary {overall_binary_ratio:.1f}%)")
            print(f"     → Hardening 可能未完成, mask 未完全二值化")
            print(f"     → 阈值化后 mask 稀疏度: {hard_sparsity_pct:.1f}%")
            print(f"     → Weight 零元素: {overall_w_sparsity:.1f}%")
            if hard_sparsity_pct >= target_pct - 10:
                print(f"     → 但阈值化后稀疏度 {hard_sparsity_pct:.1f}% 接近目标 {target_pct:.0f}%,")
                print(f"       mask 学习方向正确, 只是 hardening 未彻底完成")

    # 关键统计对比表
    print(f"\n  {'─'*50}")
    print(f"  | 指标                    | 值          |")
    print(f"  {'─'*50}")
    print(f"  | 目标稀疏度              | {target_pct:.1f}%       |")
    print(f"  | Mask exact-zero 比例    | {overall_mask_sparsity:.2f}%     |")
    print(f"  | Mask 二值化比例         | {overall_binary_ratio:.2f}%     |")
    print(f"  | Weight exact-zero 比例  | {overall_w_sparsity:.2f}%     |")
    print(f"  | Weight near-zero 比例   | {overall_w_near_zero:.2f}%     |")
    print(f"  {'─'*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_sparsity.py <checkpoint_path>")
        print("例如: python check_sparsity.py out_llama/.../model_best.pt")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        sys.exit(1)

    analyze_checkpoint(ckpt_path)
