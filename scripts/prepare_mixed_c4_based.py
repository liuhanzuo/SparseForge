#!/usr/bin/env python3
"""
以 C4 为基准准备混合数据集

逻辑：
1. 首先处理 C4 数据，计算 C4 的 token 总数作为基准
2. 根据配比计算其他域需要的 token 数
   - 例如：C4 占 60%，则其他域总共需要 (40%/60%) * C4_tokens = 0.667 * C4_tokens
3. 其他域按各自比例分配（ArXiv 15%, GitHub 15%, StackExchange 10% -> 归一化为 37.5%, 37.5%, 25%）
4. 如果某域数据不足，复制补全

用法：
    # 默认：C4 60%, ArXiv 15%, GitHub 15%, StackExchange 10%
    python scripts/prepare_mixed_c4_based.py
    
    # 自定义配比
    python scripts/prepare_mixed_c4_based.py --c4_ratio 0.5 --arxiv_ratio 0.2 --github_ratio 0.2 --se_ratio 0.1
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# 全局 tokenizer
_TOKENIZER = None
_TOKENIZER_PATH = None


def get_tokenizer(tokenizer_path: str):
    """获取或初始化全局 tokenizer"""
    global _TOKENIZER, _TOKENIZER_PATH
    if _TOKENIZER is None or _TOKENIZER_PATH != tokenizer_path:
        print(f"Loading tokenizer from {tokenizer_path}...")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True
        )
        if _TOKENIZER.pad_token_id is None and _TOKENIZER.eos_token_id is not None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        if _TOKENIZER.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token_id")
        _TOKENIZER_PATH = tokenizer_path
        print(f"✓ Tokenizer loaded (vocab_size={len(_TOKENIZER)})")
    return _TOKENIZER


def read_jsonl_file(file_path: str, text_field: str = "text") -> Generator[str, None, None]:
    """读取 JSONL 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text and len(text.strip()) > 50:
                        yield text
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def read_json_gz_file(file_path: str, text_field: str = "text") -> Generator[str, None, None]:
    """读取 gzip 压缩的 JSON 文件（C4 格式）"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text and len(text.strip()) > 50:
                        yield text
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def collect_files(data_dir: str) -> Dict[str, List[str]]:
    """收集所有域的文件"""
    files = {}
    
    # C4 数据
    c4_dir = os.path.join(data_dir, "c4_dataset")
    c4_files = sorted([
        os.path.join(c4_dir, f"c4-train.{i:05d}-of-01024.json.gz")
        for i in range(128)
        if os.path.exists(os.path.join(c4_dir, f"c4-train.{i:05d}-of-01024.json.gz"))
    ])
    if c4_files:
        files["c4"] = c4_files
        print(f"Found {len(c4_files)} C4 files")
    
    # ArXiv
    arxiv_dir = os.path.join(data_dir, "mixed_raw", "arxiv")
    if os.path.isdir(arxiv_dir):
        arxiv_files = sorted([
            os.path.join(arxiv_dir, f)
            for f in os.listdir(arxiv_dir)
            if f.endswith('.jsonl')
        ])
        if arxiv_files:
            files["arxiv"] = arxiv_files
            print(f"Found {len(arxiv_files)} ArXiv files")
    
    # GitHub
    github_dir = os.path.join(data_dir, "mixed_raw", "github")
    if os.path.isdir(github_dir):
        github_files = sorted([
            os.path.join(github_dir, f)
            for f in os.listdir(github_dir)
            if f.endswith('.jsonl')
        ])
        if github_files:
            files["github"] = github_files
            print(f"Found {len(github_files)} GitHub files")
    
    # StackExchange
    se_dir = os.path.join(data_dir, "mixed_raw", "stackexchange")
    if os.path.isdir(se_dir):
        se_files = sorted([
            os.path.join(se_dir, f)
            for f in os.listdir(se_dir)
            if f.endswith('.jsonl')
        ])
        if se_files:
            files["stackexchange"] = se_files
            print(f"Found {len(se_files)} StackExchange files")
    
    # Wikipedia
    wiki_dir = os.path.join(data_dir, "mixed_raw", "wikipedia")
    if os.path.isdir(wiki_dir):
        wiki_files = sorted([
            os.path.join(wiki_dir, f)
            for f in os.listdir(wiki_dir)
            if f.endswith('.jsonl') or f.endswith('.json')
        ])
        if wiki_files:
            files["wikipedia"] = wiki_files
            print(f"Found {len(wiki_files)} Wikipedia files")
    
    return files


def tokenize_texts(texts: List[str], tokenizer_path: str) -> List[List[int]]:
    """分词"""
    tokenizer = get_tokenizer(tokenizer_path)
    encoded = tokenizer(texts, add_special_tokens=False)
    ids_list = encoded["input_ids"]
    eos = int(tokenizer.eos_token_id)
    ids_list = [ids + [eos] for ids in ids_list]
    return ids_list


def process_file_worker(args: Tuple[str, str, str]) -> Tuple[List[List[int]], str]:
    """处理单个文件的 worker"""
    file_path, text_field, tokenizer_path = args
    
    texts = []
    if file_path.endswith('.json.gz'):
        for text in read_json_gz_file(file_path, text_field):
            texts.append(text)
    else:
        for text in read_jsonl_file(file_path, text_field):
            texts.append(text)
    
    if not texts:
        return [], file_path
    
    # 分批分词
    all_ids = []
    batch_size = 1000
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        ids = tokenize_texts(batch_texts, tokenizer_path)
        all_ids.extend(ids)
    
    return all_ids, file_path


def process_domain(domain_name: str, files: List[str], tokenizer_path: str, num_workers: int) -> List[List[int]]:
    """处理一个域的所有文件"""
    print(f"\n--- Processing {domain_name.upper()} ({len(files)} files) ---")
    
    worker_args = [(f, "text", tokenizer_path) for f in files]
    all_ids = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_worker, arg): arg[0] for arg in worker_args}
        
        with tqdm(total=len(futures), desc=f"Processing {domain_name}") as pbar:
            for future in as_completed(futures):
                try:
                    ids_list, file_path = future.result()
                    all_ids.extend(ids_list)
                    pbar.set_postfix_str(f"seqs: {len(all_ids):,}")
                except Exception as e:
                    print(f"Error processing file: {e}")
                pbar.update(1)
    
    total_tokens = sum(len(ids) for ids in all_ids)
    print(f"  {domain_name}: {len(all_ids):,} sequences, {total_tokens:,} tokens")
    
    return all_ids


def sample_to_target(sequences: List[List[int]], target_tokens: int, seed: int = 42) -> List[List[int]]:
    """
    采样/复制序列以达到目标 token 数
    - 如果数据足够：随机采样
    - 如果数据不足：复制补全
    """
    random.seed(seed)
    
    total_available = sum(len(s) for s in sequences)
    
    if total_available == 0:
        return []
    
    # 数据足够：随机采样
    if total_available >= target_tokens:
        random.shuffle(sequences)
        sampled = []
        current_tokens = 0
        for seq in sequences:
            if current_tokens >= target_tokens:
                break
            sampled.append(seq)
            current_tokens += len(seq)
        print(f"    采样 {len(sampled):,} sequences ({current_tokens:,} tokens) from {len(sequences):,} available")
        return sampled
    
    # 数据不足：复制补全
    print(f"    数据不足 ({total_available:,} < {target_tokens:,})，复制补全...")
    sampled = list(sequences)  # 先全部使用
    current_tokens = total_available
    
    # 复制直到达到目标
    while current_tokens < target_tokens:
        random.shuffle(sequences)
        for seq in sequences:
            if current_tokens >= target_tokens:
                break
            sampled.append(seq)
            current_tokens += len(seq)
    
    print(f"    复制后: {len(sampled):,} sequences ({current_tokens:,} tokens)")
    return sampled


def write_memmap(all_ids: List[List[int]], output_path: str, desc: str = "writing") -> int:
    """写入 memmap 文件"""
    total_tokens = sum(len(ids) for ids in all_ids)
    print(f"\n{desc}: {total_tokens:,} tokens -> {output_path}")
    
    dtype = np.uint16
    mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    
    idx = 0
    for ids in tqdm(all_ids, desc=desc):
        arr = np.array(ids, dtype=dtype)
        mmap[idx:idx + len(arr)] = arr
        idx += len(arr)
    
    mmap.flush()
    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="以 C4 为基准准备混合数据集")
    parser.add_argument("--tokenizer", type=str, default="models/NousResearch--Llama-2-7b-hf")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/mixed_c4_based")
    parser.add_argument("--val_ratio", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=32)
    
    # 配比：这些比例决定各域在最终数据集中的占比
    parser.add_argument("--c4_ratio", type=float, default=0.60, help="C4 占比 (基准)")
    parser.add_argument("--arxiv_ratio", type=float, default=0.15, help="ArXiv 占比")
    parser.add_argument("--github_ratio", type=float, default=0.15, help="GitHub 占比")
    parser.add_argument("--se_ratio", type=float, default=0.10, help="StackExchange 占比")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 路径处理
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(project_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    tokenizer_path = os.path.join(project_dir, args.tokenizer) if not os.path.isabs(args.tokenizer) else args.tokenizer
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    
    if not args.force and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Output files already exist: {train_path}, {val_path}")
        print("Use --force to overwrite")
        return
    
    print("=" * 70)
    print("以 C4 为基准的混合数据集准备")
    print("=" * 70)
    print(f"配比: C4 {args.c4_ratio:.0%}, ArXiv {args.arxiv_ratio:.0%}, GitHub {args.github_ratio:.0%}, SE {args.se_ratio:.0%}")
    print(f"Tokenizer: {tokenizer_path}")
    print()
    
    # 加载 tokenizer
    tokenizer = get_tokenizer(tokenizer_path)
    
    # 收集文件
    print("\n[Step 1] 收集数据文件...")
    all_files = collect_files(data_dir)
    
    if "c4" not in all_files:
        print("ERROR: C4 数据不存在！")
        return
    
    # 首先处理 C4 数据
    print("\n[Step 2] 处理 C4 数据（作为基准）...")
    c4_data = process_domain("c4", all_files["c4"], tokenizer_path, args.num_workers)
    c4_tokens = sum(len(ids) for ids in c4_data)
    
    print(f"\n>>> C4 基准: {c4_tokens:,} tokens")
    
    # 计算其他域需要的 token 数
    # C4 占 c4_ratio，所以 C4_tokens = total * c4_ratio
    # total = C4_tokens / c4_ratio
    # 其他域 = total * 其他域比例 = C4_tokens / c4_ratio * 其他域比例
    
    total_tokens = c4_tokens / args.c4_ratio
    
    other_ratios = {
        "arxiv": args.arxiv_ratio,
        "github": args.github_ratio,
        "stackexchange": args.se_ratio,
    }
    
    # 过滤出实际存在的域
    available_others = {k: v for k, v in other_ratios.items() if k in all_files}
    
    print("\n[Step 3] 计算各域目标 token 数...")
    domain_targets = {"c4": c4_tokens}
    
    for domain, ratio in available_others.items():
        target = int(total_tokens * ratio)
        domain_targets[domain] = target
        print(f"  {domain}: {target:,} tokens (占比 {ratio:.0%})")
    
    # 处理其他域数据
    print("\n[Step 4] 处理其他域数据...")
    domain_data = {"c4": c4_data}
    
    for domain, files in all_files.items():
        if domain == "c4":
            continue
        if domain not in domain_targets:
            continue
        
        # 处理该域
        data = process_domain(domain, files, tokenizer_path, args.num_workers)
        
        # 采样/复制到目标数量
        target = domain_targets[domain]
        sampled = sample_to_target(data, target, args.seed + hash(domain) % 10000)
        domain_data[domain] = sampled
    
    # 合并所有数据
    print("\n[Step 5] 合并数据...")
    all_train_ids = []
    all_val_ids = []
    
    domain_stats = {}
    
    for domain, sequences in domain_data.items():
        # 划分 train/val
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * args.val_ratio))
        val_ids = sequences[:n_val]
        train_ids = sequences[n_val:]
        
        all_train_ids.extend(train_ids)
        all_val_ids.extend(val_ids)
        
        train_tokens = sum(len(ids) for ids in train_ids)
        val_tokens = sum(len(ids) for ids in val_ids)
        
        domain_stats[domain] = {
            "train_seqs": len(train_ids),
            "train_tokens": train_tokens,
            "val_seqs": len(val_ids),
            "val_tokens": val_tokens,
            "target_tokens": domain_targets.get(domain, 0)
        }
    
    # 计算实际比例
    total_train_tokens = sum(s['train_tokens'] for s in domain_stats.values())
    actual_ratios = {d: s['train_tokens'] / total_train_tokens for d, s in domain_stats.items()}
    
    # 最终打乱
    print("\n[Step 6] 打乱并写入文件...")
    random.shuffle(all_train_ids)
    random.shuffle(all_val_ids)
    
    # 写入文件
    train_tokens = write_memmap(all_train_ids, train_path, "Writing train.bin")
    val_tokens = write_memmap(all_val_ids, val_path, "Writing val.bin")
    
    # 保存元数据
    metadata = {
        "tokenizer": tokenizer_path,
        "vocab_size": len(tokenizer),
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_sequences": len(all_train_ids),
        "val_sequences": len(all_val_ids),
        "c4_base_tokens": c4_tokens,
        "requested_ratios": {
            "c4": args.c4_ratio,
            "arxiv": args.arxiv_ratio,
            "github": args.github_ratio,
            "stackexchange": args.se_ratio,
        },
        "actual_ratios": actual_ratios,
        "domain_stats": domain_stats,
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # dtype 文件
    dtype_path = os.path.join(output_dir, "dtype.txt")
    with open(dtype_path, 'w') as f:
        f.write("uint16")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"C4 基准: {c4_tokens:,} tokens")
    print(f"总训练: {train_tokens:,} tokens")
    print(f"总验证: {val_tokens:,} tokens")
    print()
    print("各域统计:")
    print(f"  {'域':<15s} {'目标比例':>10s} {'实际比例':>10s} {'Tokens':>15s}")
    print("  " + "-" * 50)
    
    target_ratios = {
        "c4": args.c4_ratio,
        "arxiv": args.arxiv_ratio,
        "github": args.github_ratio,
        "stackexchange": args.se_ratio,
    }
    
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        target_pct = target_ratios.get(domain, 0) * 100
        actual_pct = actual_ratios.get(domain, 0) * 100
        print(f"  {domain:<15s} {target_pct:>9.1f}% {actual_pct:>9.1f}% {stats['train_tokens']:>15,}")
    
    print()
    print(f"输出文件:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {metadata_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
