#!/usr/bin/env python3
"""
制作 FLAN 高比例的 Dolmino 混合数据集，用于 final finetune 阶段。
目标：拉高 BoolQ、HellaSwag、WinoGrande 等 downstream 分数。

输出 train.bin + val.bin (uint32 memmap)，兼容 retrain_universal.py。

用法:
    # 默认 FLAN-heavy 配比 (FLAN 35%, DCLM 45%, 其余 20%)
    python scripts/prepare_dolmino_flan_heavy.py \
        --raw_dir data/dolmino-mix-1124-raw/raw \
        --output_dir data/dolmino-flan-heavy \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --target_tokens 500M \
        --num_workers 32

    # 自定义比例
    python scripts/prepare_dolmino_flan_heavy.py \
        --raw_dir data/dolmino-mix-1124-raw/raw \
        --output_dir data/dolmino-flan-heavy \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --target_tokens 1B \
        --ratios flan:0.35 dclm:0.45 math:0.05 code:0.05 wiki:0.05 pes2o:0.05 \
        --num_workers 32

    # 如果 tokenizer 在本地
    python scripts/prepare_dolmino_flan_heavy.py \
        --raw_dir data/dolmino-mix-1124-raw/raw \
        --output_dir data/dolmino-flan-heavy \
        --tokenizer_path models/NousResearch--Llama-2-7b-hf \
        --target_tokens 500M \
        --num_workers 32
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import logging
import os
import random
import sys
import time
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dolmino_flan_heavy")


# ======================================================================
# 工具函数
# ======================================================================

def parse_token_count(s: str) -> int:
    """解析 token 数量字符串，如 '500M', '1B', '10G'。"""
    if isinstance(s, int):
        return s
    s = s.strip().upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'G': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def parse_ratios(ratio_strs: List[str]) -> Dict[str, float]:
    """解析比例字符串列表，如 ['flan:0.35', 'dclm:0.45']。"""
    ratios = {}
    for s in ratio_strs:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"比例格式错误: {s}，应为 'name:ratio'")
        name, ratio = parts[0].strip(), float(parts[1].strip())
        ratios[name] = ratio
    return ratios


def discover_subsets(raw_dir: str) -> Dict[str, List[str]]:
    """
    发现 raw 目录下的所有子集及其数据文件。
    
    Dolmino raw 目录结构:
        raw/data/<subset>/<files...>
    或直接:
        raw/<subset>/<files...>
    """
    subsets = {}
    
    # 支持的文件格式
    extensions = ["*.jsonl", "*.jsonl.gz", "*.json.gz", "*.json.zst", "*.parquet"]
    
    # 尝试 raw/data/<subset>/... 结构
    data_subdir = os.path.join(raw_dir, "data")
    search_base = data_subdir if os.path.isdir(data_subdir) else raw_dir
    
    # 遍历子目录
    for entry in sorted(os.listdir(search_base)):
        entry_path = os.path.join(search_base, entry)
        if not os.path.isdir(entry_path):
            continue
        
        # 收集该子集下的所有数据文件
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(entry_path, "**", ext), recursive=True))
        
        if files:
            files.sort()
            subsets[entry] = files
            logger.info(f"  子集 {entry}: {len(files)} 个文件")
    
    return subsets


def read_data_file(filepath: str) -> Generator[str, None, None]:
    """读取单个数据文件，返回文本迭代器。"""
    if filepath.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(filepath)
            df = table.to_pandas()
            for _, row in df.iterrows():
                text = row.get("text", "")
                if text and str(text).strip():
                    yield str(text)
        except ImportError:
            logger.warning(f"需要 pyarrow: {filepath}")
    elif filepath.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError:
            logger.error(f"需要 zstandard: pip install zstandard")
            return
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        text = d.get("text", "")
                        if text and text.strip():
                            yield text
                    except json.JSONDecodeError:
                        continue
    elif filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    text = d.get("text", "")
                    if text and text.strip():
                        yield text
                except json.JSONDecodeError:
                    continue
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    text = d.get("text", "")
                    if text and text.strip():
                        yield text
                except json.JSONDecodeError:
                    continue


# ======================================================================
# Tokenize worker (fork 模式多进程，主进程预加载 tokenizer)
# fork 后子进程直接继承父进程内存 (copy-on-write)，零启动开销
# ======================================================================

# 全局 tokenizer（主进程加载，fork 后子进程自动继承）
_G_TOKENIZER = None
_G_EOS_ID = None


def _init_global_tokenizer(tokenizer_path: str):
    """在主进程中加载 tokenizer 到全局变量。"""
    global _G_TOKENIZER, _G_EOS_ID
    if _G_TOKENIZER is not None:
        return
    from transformers import AutoTokenizer
    logger.info(f"  主进程加载 tokenizer: {tokenizer_path}")
    _G_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True,
    )
    if _G_TOKENIZER.pad_token is None:
        _G_TOKENIZER.pad_token = _G_TOKENIZER.eos_token
    _G_EOS_ID = _G_TOKENIZER.eos_token_id
    logger.info(f"  tokenizer 加载完成 (vocab_size={_G_TOKENIZER.vocab_size}, eos_id={_G_EOS_ID})")


def _tokenize_single_file(args_tuple):
    """
    fork 模式 worker：tokenize 单个文件，写临时 .npy 文件。
    直接使用从父进程继承的全局 _G_TOKENIZER（零开销）。
    
    Args:
        args_tuple: (file_idx, filepath, max_seq_len, tmp_dir, total_files)
    
    Returns:
        dict: {"file", "tmp_npy", "num_tokens", "num_docs"}
    """
    file_idx, filepath, max_seq_len, tmp_dir, total_files = args_tuple
    
    tokenizer = _G_TOKENIZER
    eos_id = _G_EOS_ID
    
    all_ids = []
    num_docs = 0
    t0 = time.time()
    
    # 批量 tokenize
    BATCH_SIZE = 256
    text_buffer = []
    
    for text in read_data_file(filepath):
        text_buffer.append(text)
        
        if len(text_buffer) >= BATCH_SIZE:
            encoded = tokenizer(text_buffer, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                if len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                if len(ids) == 0:
                    continue
                ids.append(eos_id)
                all_ids.extend(ids)
                num_docs += 1
            text_buffer = []
    
    # 处理剩余
    if text_buffer:
        encoded = tokenizer(text_buffer, add_special_tokens=False)
        for ids in encoded["input_ids"]:
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            if len(ids) == 0:
                continue
            ids.append(eos_id)
            all_ids.extend(ids)
            num_docs += 1
    
    num_tokens = len(all_ids)
    t1 = time.time()
    
    if num_tokens == 0:
        return {
            "file": filepath,
            "tmp_npy": None,
            "num_tokens": 0,
            "num_docs": 0,
        }
    
    # 写入临时 npy 文件（避免通过 IPC 传大量数据）
    tmp_npy = os.path.join(tmp_dir, f"tmp_{file_idx:05d}.npy")
    arr = np.array(all_ids, dtype=np.uint32)
    np.save(tmp_npy, arr)
    del all_ids, arr
    
    speed = num_tokens / max(t1 - t0, 0.01)
    print(
        f"  [{file_idx+1}/{total_files}] {os.path.basename(filepath)}: "
        f"{num_docs:,} docs, {num_tokens:,} tokens, "
        f"{speed:,.0f} tok/s, {t1-t0:.1f}s",
        flush=True,
    )
    
    return {
        "file": filepath,
        "tmp_npy": tmp_npy,
        "num_tokens": num_tokens,
        "num_docs": num_docs,
    }


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="制作 FLAN 高比例的 Dolmino 混合数据集 (final finetune 用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--raw_dir", type=str, required=True,
        help="Dolmino raw 数据目录 (如 data/dolmino-mix-1124-raw/raw)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录 (会生成 train.bin, val.bin, dtype.txt, metadata.json)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer 路径 (HuggingFace ID 或本地路径)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096,
        help="单个文档最大 token 长度 (默认 4096)",
    )
    parser.add_argument(
        "--target_tokens", type=str, default="500M",
        help="目标总 token 数 (如 '500M', '1B', '2B')",
    )
    parser.add_argument(
        "--ratios", nargs="*", default=None,
        help="子集比例，格式: name:ratio (如 flan:0.35 dclm:0.45)。不指定则用默认 FLAN-heavy 配比",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.001,
        help="验证集比例 (默认 0.1%%)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16,
        help="并行 worker 数",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--allow_repeat", action="store_true",
        help="数据不足时允许重复采样",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已有输出文件",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    target_tokens = parse_token_count(args.target_tokens)
    
    # ---- 默认 FLAN-heavy 配比 ----
    # 针对 BoolQ / HellaSwag / WinoGrande 优化
    DEFAULT_RATIOS = {
        "flan":  0.35,   # FLAN 指令数据 -> BoolQ, WinoGrande
        "dclm":  0.40,   # 高质量网页 -> HellaSwag, 通用 LM 能力
        "math":  0.05,   # 数学
        "code":  0.05,   # 代码
        "wiki":  0.05,   # 百科
        "pes2o": 0.05,   # 学术论文
        "reddit": 0.05,  # Reddit
    }
    
    if args.ratios:
        domain_ratios = parse_ratios(args.ratios)
    else:
        domain_ratios = DEFAULT_RATIOS.copy()
    
    # ---- 检查输出 ----
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    
    if not args.force and os.path.exists(train_path):
        logger.error(f"输出文件已存在: {train_path}，使用 --force 覆盖")
        return
    
    # ---- Step 1: 发现子集 ----
    logger.info("=" * 70)
    logger.info("Dolmino FLAN-Heavy Mix 数据集制作")
    logger.info("=" * 70)
    logger.info(f"Raw 目录: {args.raw_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"目标 tokens: {target_tokens:,}")
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    logger.info("")
    
    logger.info("[Step 1/5] 发现数据子集...")
    subsets = discover_subsets(args.raw_dir)
    
    if not subsets:
        logger.error(f"在 {args.raw_dir} 中未找到任何数据子集!")
        return
    
    logger.info(f"共发现 {len(subsets)} 个子集: {list(subsets.keys())}")
    
    # 归一化比例（只对存在的子集）
    available = set(subsets.keys())
    active_ratios = {}
    for name in available:
        # 模糊匹配：如果子集名包含 ratio key，就用那个比例
        matched = False
        for rkey, rval in domain_ratios.items():
            if rkey in name or name in rkey:
                active_ratios[name] = rval
                matched = True
                break
        if not matched:
            # 未匹配的子集给一个小比例
            active_ratios[name] = 0.02
    
    # 归一化
    total_ratio = sum(active_ratios.values())
    for k in active_ratios:
        active_ratios[k] /= total_ratio
    
    logger.info("\n目标混合比例:")
    for name in sorted(active_ratios.keys()):
        logger.info(f"  {name:15s}: {active_ratios[name]:.1%}")
    
    # 每个子集的目标 token 数
    subset_targets = {name: int(target_tokens * ratio) for name, ratio in active_ratios.items()}
    
    # ---- Step 2: Tokenize 每个子集 ----
    logger.info(f"\n[Step 2/5] Tokenize 所有子集 ({args.num_workers} workers)...")
    
    # 改用 npy 文件路径引用，不加载到内存（避免 OOM）
    subset_npy_files: Dict[str, List[Tuple[str, int]]] = {}  # name -> [(npy_path, num_tokens), ...]
    subset_stats: Dict[str, dict] = {}
    
    # 主进程预加载 tokenizer（fork 后子进程自动继承，零开销）
    _init_global_tokenizer(args.tokenizer_path)
    
    for subset_name, files in subsets.items():
        target_toks_for_subset = subset_targets.get(subset_name, 0)
        
        logger.info(f"\n--- {subset_name.upper()} ({len(files)} 文件, 目标 {target_toks_for_subset:,} tokens) ---")
        
        # 打乱文件顺序，避免偏向某些分片
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        # 创建临时目录存放 worker 输出的 npy 文件
        tmp_dir = os.path.join(args.output_dir, f"_tmp_{subset_name}")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # ---- 断点续传：检测已有的临时 npy 文件 ----
        existing_npys = {}
        for f_name in os.listdir(tmp_dir):
            if f_name.endswith(".npy"):
                npy_path = os.path.join(tmp_dir, f_name)
                try:
                    arr = np.load(npy_path)
                    existing_npys[f_name] = (npy_path, len(arr))
                    del arr
                except Exception:
                    # 损坏的 npy 文件，删除重做
                    os.remove(npy_path)
        
        if existing_npys:
            logger.info(f"  🔄 断点续传: 发现 {len(existing_npys)} 个已处理的 npy 文件")
        
        # 过滤掉已处理的文件
        pending_args = []
        resumed_results = []
        for i, fp in enumerate(shuffled_files):
            expected_npy = f"tmp_{i:05d}.npy"
            if expected_npy in existing_npys:
                npy_path, num_tokens = existing_npys[expected_npy]
                resumed_results.append({
                    "file": fp,
                    "tmp_npy": npy_path,
                    "num_tokens": num_tokens,
                    "num_docs": 0,  # 无法恢复 doc 数，但不影响最终结果
                })
            else:
                pending_args.append((i, fp, args.max_seq_len, tmp_dir, len(shuffled_files)))
        
        total_files = len(shuffled_files)
        actual_workers = min(args.num_workers, max(len(pending_args), 1))
        
        if pending_args:
            logger.info(f"  启动 {actual_workers} 个 worker 进程 (fork 模式), 待处理 {len(pending_args)}/{total_files} 文件...")
        else:
            logger.info(f"  所有 {total_files} 个文件已处理完毕 (断点续传)")
        
        all_results = list(resumed_results)
        # 早停阈值：已收集 token 数达到目标的 1.1 倍即可停止（留 10% 余量给采样打乱）
        early_stop_threshold = int(target_toks_for_subset * 1.1)
        resumed_tokens = sum(r["num_tokens"] for r in resumed_results)
        
        if resumed_tokens >= early_stop_threshold:
            logger.info(f"  ⏩ 断点续传已收集 {resumed_tokens:,} tokens >= 目标 {target_toks_for_subset:,}，跳过剩余文件")
        elif pending_args:
            if actual_workers > 1:
                # fork 模式：子进程直接继承父进程的 _G_TOKENIZER，零启动开销
                ctx = mp.get_context("fork")
                collected_tokens = resumed_tokens
                with ctx.Pool(processes=actual_workers) as pool:
                    for r in pool.imap_unordered(_tokenize_single_file, pending_args):
                        all_results.append(r)
                        collected_tokens += r["num_tokens"]
                        # 早停：已收集够目标 token 数，终止剩余任务
                        if collected_tokens >= early_stop_threshold:
                            logger.info(
                                f"  ⏩ 早停: 已收集 {collected_tokens:,} tokens >= "
                                f"目标 {target_toks_for_subset:,} (×1.1)，"
                                f"终止剩余 worker"
                            )
                            pool.terminate()
                            break
            else:
                collected_tokens = resumed_tokens
                for wa in pending_args:
                    r = _tokenize_single_file(wa)
                    all_results.append(r)
                    collected_tokens += r["num_tokens"]
                    if collected_tokens >= early_stop_threshold:
                        logger.info(
                            f"  ⏩ 早停: 已收集 {collected_tokens:,} tokens >= "
                            f"目标 {target_toks_for_subset:,} (×1.1)"
                        )
                        break
        
        # 收集 npy 文件路径和 token 数（不加载到内存！）
        total_docs = sum(r["num_docs"] for r in all_results)
        total_toks = sum(r["num_tokens"] for r in all_results)
        
        npy_file_list = []
        for r in all_results:
            if r["tmp_npy"] is not None and r["num_tokens"] > 0:
                npy_file_list.append((r["tmp_npy"], r["num_tokens"]))
        
        # 打乱 npy 文件顺序（用于后续采样）
        random.shuffle(npy_file_list)
        
        subset_npy_files[subset_name] = npy_file_list
        subset_stats[subset_name] = {
            "num_files": len(files),
            "num_docs": total_docs,
            "total_tokens": total_toks,
        }
        logger.info(f"  {subset_name}: {total_docs:,} docs, {total_toks:,} tokens, {len(npy_file_list)} npy 文件")
    
    # ---- Step 3: 按比例采样 (基于 npy 文件路径，不加载到内存) ----
    logger.info(f"\n[Step 3/5] 按比例采样...")
    
    # train/val 分别收集 (npy_path, num_tokens) 列表
    all_train_npys: List[Tuple[str, int]] = []
    all_val_npys: List[Tuple[str, int]] = []
    actual_stats = {}
    
    for subset_name, npy_list in subset_npy_files.items():
        target = subset_targets.get(subset_name, 0)
        if not npy_list or target == 0:
            continue
        
        available_tokens = sum(nt for _, nt in npy_list)
        
        # 打乱
        random.shuffle(npy_list)
        
        if available_tokens <= target:
            if args.allow_repeat and available_tokens < target:
                # 重复采样
                sampled = []
                current = 0
                while current < target:
                    random.shuffle(npy_list)
                    for item in npy_list:
                        if current >= target:
                            break
                        sampled.append(item)
                        current += item[1]
                logger.info(f"  {subset_name}: 数据不足，重复采样 {available_tokens:,} -> {current:,} tokens")
            else:
                sampled = list(npy_list)
                logger.info(f"  {subset_name}: 使用全部 {available_tokens:,} tokens (目标 {target:,})")
        else:
            # 采样到目标，在 token 级别精确截断
            sampled = []
            current = 0
            for npy_path, num_tokens in npy_list:
                if current >= target:
                    break
                remaining = target - current
                if num_tokens <= remaining:
                    # 整个 npy 文件都需要
                    sampled.append((npy_path, num_tokens))
                    current += num_tokens
                else:
                    # 最后一个 npy 文件需要截断
                    truncated_path = npy_path + ".trunc.npy"
                    arr = np.load(npy_path)
                    np.save(truncated_path, arr[:remaining])
                    del arr
                    sampled.append((truncated_path, remaining))
                    current += remaining
                    logger.info(f"  {subset_name}: 截断最后一个 npy 文件 {num_tokens:,} -> {remaining:,} tokens")
            logger.info(f"  {subset_name}: 采样 {current:,}/{available_tokens:,} tokens (目标 {target:,})")
        
        # Train/Val split: 按 token 数比例切分（而非按 npy 文件数）
        # 避免大文件子集只有少量 npy 时整个被分到 val
        random.shuffle(sampled)
        total_sampled_tokens = sum(nt for _, nt in sampled)
        val_token_target = max(1, int(total_sampled_tokens * args.val_ratio))
        
        train_items = []
        val_items = []
        val_collected = 0
        
        # 从末尾开始收集 val tokens
        for npy_path, num_tokens in reversed(sampled):
            if val_collected >= val_token_target:
                train_items.append((npy_path, num_tokens))
            else:
                need = val_token_target - val_collected
                if num_tokens <= need:
                    # 整个文件给 val
                    val_items.append((npy_path, num_tokens))
                    val_collected += num_tokens
                else:
                    # 需要拆分这个 npy 文件：前部分给 train，后部分给 val
                    arr = np.load(npy_path)
                    # val 部分
                    val_part_path = npy_path + ".val.npy"
                    np.save(val_part_path, arr[-need:])
                    val_items.append((val_part_path, need))
                    val_collected += need
                    # train 部分
                    train_part_path = npy_path + ".train.npy"
                    train_len = num_tokens - need
                    np.save(train_part_path, arr[:train_len])
                    train_items.append((train_part_path, train_len))
                    del arr
        
        # 反转 train_items 恢复原始顺序（因为是从 reversed 收集的）
        train_items.reverse()
        
        all_train_npys.extend(train_items)
        all_val_npys.extend(val_items)
        
        train_toks = sum(nt for _, nt in train_items)
        val_toks = sum(nt for _, nt in val_items)
        actual_stats[subset_name] = {
            "train_tokens": train_toks,
            "val_tokens": val_toks,
            "train_seqs": len(train_items),
            "val_seqs": len(val_items),
        }
    
    # ---- Step 4: Shuffle + 流式写入 ----
    logger.info(f"\n[Step 4/5] Shuffle + 流式写入 train.bin / val.bin...")
    
    random.shuffle(all_train_npys)
    random.shuffle(all_val_npys)
    
    def stream_write_from_npys(npy_items: List[Tuple[str, int]], output_path: str, desc: str) -> int:
        """从 npy 文件流式读取并写入 uint32 memmap 文件（内存友好）。"""
        total = sum(nt for _, nt in npy_items)
        logger.info(f"  {desc}: {total:,} tokens -> {output_path}")
        
        mm = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(total,))
        idx = 0
        for npy_path, num_tokens in tqdm(npy_items, desc=desc):
            arr = np.load(npy_path)  # 只加载一个 npy 文件
            if arr.dtype != np.uint32:
                arr = arr.astype(np.uint32)
            mm[idx:idx + len(arr)] = arr
            idx += len(arr)
            del arr  # 立即释放
        mm.flush()
        del mm
        return total
    
    train_tokens = stream_write_from_npys(all_train_npys, train_path, "Writing train.bin")
    val_tokens = stream_write_from_npys(all_val_npys, val_path, "Writing val.bin")
    
    # 写 dtype.txt
    dtype_path = os.path.join(args.output_dir, "dtype.txt")
    with open(dtype_path, "w") as f:
        f.write("uint32")
    
    # ---- Step 5: 保存元数据 ----
    logger.info(f"\n[Step 5/5] 保存元数据...")
    
    total_train = sum(s["train_tokens"] for s in actual_stats.values())
    actual_ratios = {
        name: s["train_tokens"] / total_train if total_train > 0 else 0
        for name, s in actual_stats.items()
    }
    
    metadata = {
        "description": "Dolmino FLAN-Heavy Mix for final finetune",
        "tokenizer": args.tokenizer_path,
        "max_seq_len": args.max_seq_len,
        "dtype": "uint32",
        "target_tokens": target_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_sequences": len(all_train_npys),
        "val_sequences": len(all_val_npys),
        "requested_ratios": {k: v for k, v in active_ratios.items()},
        "actual_ratios": actual_ratios,
        "subset_stats": actual_stats,
        "seed": args.seed,
    }
    
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # ---- 汇总 ----
    logger.info("\n" + "=" * 70)
    logger.info("完成!")
    logger.info("=" * 70)
    logger.info(f"Train tokens: {train_tokens:,} ({train_tokens * 4 / 1024**3:.2f} GB)")
    logger.info(f"Val tokens:   {val_tokens:,} ({val_tokens * 4 / 1024**3:.2f} GB)")
    logger.info("")
    logger.info("子集分布 (目标 vs 实际):")
    logger.info(f"  {'子集':<15s} {'目标%':>8s} {'实际%':>8s} {'Tokens':>15s}")
    logger.info("  " + "-" * 50)
    for name in sorted(actual_stats.keys()):
        target_pct = active_ratios.get(name, 0) * 100
        actual_pct = actual_ratios.get(name, 0) * 100
        toks = actual_stats[name]["train_tokens"]
        logger.info(f"  {name:<15s} {target_pct:>7.1f}% {actual_pct:>7.1f}% {toks:>15,}")
    
    logger.info("")
    logger.info(f"输出文件:")
    logger.info(f"  {train_path}")
    logger.info(f"  {val_path}")
    logger.info(f"  {dtype_path}")
    logger.info(f"  {meta_path}")
    logger.info("")
    logger.info("使用方式:")
    logger.info(f"  python retrain_universal.py --dataset {os.path.basename(args.output_dir)} ...")
    
    # 清理临时 npy 文件
    logger.info("\n清理临时 npy 文件...")
    cleaned = 0
    for subset_name in subset_npy_files:
        tmp_dir = os.path.join(args.output_dir, f"_tmp_{subset_name}")
        if os.path.isdir(tmp_dir):
            for f_name in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, f_name)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    cleaned += 1
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
    logger.info(f"  已清理 {cleaned} 个临时文件")


if __name__ == "__main__":
    main()
