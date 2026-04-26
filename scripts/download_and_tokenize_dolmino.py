#!/usr/bin/env python3
"""
下载并 tokenize Dolmino-Mix-1124 数据集 (OLMo2 CAST 退火阶段数据)。

使用 Llama2-7b 的 tokenizer 进行 tokenize，支持多种输出格式:
  - numpy memmap (.npy): 适合高效随机访问的预训练
  - jsonl: 每行一个 tokenized 样本，兼容项目现有流程

用法:
    # 下载 + tokenize 全量数据，输出 numpy memmap 格式
    python scripts/download_and_tokenize_dolmino.py \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --output_dir /apdcephfs/pig_data/dolmino-mix-1124-tokenized \
        --output_format numpy \
        --max_seq_len 4096 \
        --num_workers 16

    # 如果 tokenizer 在本地
    python scripts/download_and_tokenize_dolmino.py \
        --tokenizer_path /path/to/llama2-7b \
        --output_dir /apdcephfs/pig_data/dolmino-mix-1124-tokenized \
        --output_format numpy \
        --max_seq_len 4096

    # 输出 jsonl 格式
    python scripts/download_and_tokenize_dolmino.py \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --output_dir /apdcephfs/pig_data/dolmino-mix-1124-tokenized \
        --output_format jsonl \
        --max_seq_len 4096

    # 只下载不 tokenize
    python scripts/download_and_tokenize_dolmino.py \
        --download_only \
        --output_dir /apdcephfs/pig_data/dolmino-mix-1124-raw

    # 指定子集下载
    python scripts/download_and_tokenize_dolmino.py \
        --tokenizer_path meta-llama/Llama-2-7b-hf \
        --output_dir /apdcephfs/pig_data/dolmino-mix-1124-tokenized \
        --subsets dclm flan math \
        --output_format numpy

说明:
    Dolmino-Mix-1124 是 Allen AI 为 OLMo2 模型 CAST (Continual Annealing Stage Training)
    阶段准备的高质量混合数据集。包含 DCLM 网页数据、FLAN 指令数据、数学数据、代码数据、
    百科全书等多种高质量数据源。

    HuggingFace 地址: https://huggingface.co/datasets/allenai/dolmino-mix-1124
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import multiprocessing as mp
from multiprocessing import cpu_count
from functools import partial

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dolmino_tokenize")

# HuggingFace 数据集 ID
DATASET_ID = "allenai/dolmino-mix-1124"


# ======================================================================
# 下载
# ======================================================================

def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的字符串。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def download_dataset(
    output_dir: str,
    subsets: Optional[list[str]] = None,
    use_mirror: bool = False,
    max_size_gb: Optional[float] = None,
    download_workers: int = 8,
) -> str:
    """
    使用 huggingface_hub 下载 Dolmino-Mix-1124 数据集。

    当指定 max_size_gb 时，会先列出所有文件及大小，按文件名排序后
    累加下载，直到总大小达到限额为止。

    Args:
        output_dir: 下载目标目录
        subsets: 要下载的子集名称列表，None 表示全部下载
        use_mirror: 是否使用 HuggingFace 镜像站
        max_size_gb: 最大下载大小 (GB)，None 表示不限制
        download_workers: 多线程下载的并发数 (默认 8)

    Returns:
        下载后的本地路径
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("已设置 HuggingFace 镜像: https://hf-mirror.com")

    download_dir = os.path.join(output_dir, "raw")
    os.makedirs(download_dir, exist_ok=True)

    # ---- 无大小限制: 直接用 snapshot_download ----
    if max_size_gb is None:
        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": DATASET_ID,
            "repo_type": "dataset",
            "local_dir": download_dir,
        }
        if subsets:
            patterns = [f"*{s}*" for s in subsets]
            kwargs["allow_patterns"] = patterns
            logger.info(f"只下载子集: {subsets}, patterns: {patterns}")

        logger.info(f"开始下载 {DATASET_ID} 到 {download_dir} ...")
        t0 = time.time()
        local_path = snapshot_download(**kwargs)
        t1 = time.time()
        logger.info(f"下载完成! 耗时 {t1 - t0:.1f}s, 路径: {local_path}")
        return local_path

    # ---- 有大小限制: 先列文件，按大小筛选后逐个下载 ----
    from huggingface_hub import HfApi, hf_hub_download

    max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    logger.info(f"下载大小限制: {max_size_gb} GB ({_format_size(max_size_bytes)})")

    api = HfApi()
    logger.info(f"正在列出 {DATASET_ID} 的文件列表...")

    # 获取 repo 中所有文件信息
    all_files = []
    for entry in api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        recursive=True,
    ):
        # 只处理文件 (跳过目录)
        if not hasattr(entry, "size") or entry.size is None:
            continue
        rpath = entry.rfilename if hasattr(entry, "rfilename") else entry.path
        all_files.append((rpath, entry.size))

    logger.info(f"仓库共 {len(all_files)} 个文件")

    # 按子集过滤
    if subsets:
        filtered = []
        for rpath, size in all_files:
            if any(s in rpath for s in subsets):
                filtered.append((rpath, size))
        logger.info(f"子集过滤后: {len(filtered)} 个文件 (子集: {subsets})")
        all_files = filtered

    # 按文件名排序，保证确定性
    all_files.sort(key=lambda x: x[0])

    # 统计总大小
    total_repo_size = sum(s for _, s in all_files)
    logger.info(f"待下载文件总大小: {_format_size(total_repo_size)}")

    # 按大小限制筛选文件
    selected_files = []
    cumulative_size = 0
    for rpath, size in all_files:
        if cumulative_size + size > max_size_bytes:
            # 如果加上这个文件会超限，跳过
            # 但如果还没选任何文件，至少选一个
            if not selected_files:
                selected_files.append((rpath, size))
                cumulative_size += size
            break
        selected_files.append((rpath, size))
        cumulative_size += size

    logger.info(
        f"将下载 {len(selected_files)}/{len(all_files)} 个文件, "
        f"总大小: {_format_size(cumulative_size)} (限制: {max_size_gb} GB)"
    )

    # 显示选中文件的子集分布
    subset_sizes: dict[str, int] = {}
    for rpath, size in selected_files:
        # 从路径提取子集名 (如 data/dclm/... -> dclm)
        parts = rpath.split("/")
        subset_name = parts[1] if len(parts) > 1 else "other"
        subset_sizes[subset_name] = subset_sizes.get(subset_name, 0) + size
    for sname, ssize in sorted(subset_sizes.items(), key=lambda x: -x[1]):
        logger.info(f"  子集 {sname}: {_format_size(ssize)}")

    # 多线程并发下载
    logger.info(f"开始下载到 {download_dir} (并发线程数: {download_workers}) ...")
    t0 = time.time()

    # 线程安全的进度统计
    progress_lock = threading.Lock()
    progress = {
        "downloaded_size": 0,
        "completed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "failed_files": [],
    }
    total_count = len(selected_files)

    def _download_one_file(file_info):
        """下载单个文件的工作函数（在线程中执行）。"""
        rpath, size = file_info
        local_file = os.path.join(download_dir, rpath)

        # 如果文件已存在且大小匹配，跳过
        if os.path.exists(local_file) and os.path.getsize(local_file) == size:
            with progress_lock:
                progress["downloaded_size"] += size
                progress["completed_count"] += 1
                progress["skipped_count"] += 1
            return True

        # 带重试的下载
        max_retries = 3
        for attempt in range(max_retries):
            try:
                hf_hub_download(
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    filename=rpath,
                    local_dir=download_dir,
                )
                with progress_lock:
                    progress["downloaded_size"] += size
                    progress["completed_count"] += 1
                    cnt = progress["completed_count"]
                    dl_size = progress["downloaded_size"]

                # 每完成 50 个文件或第一个文件时打印进度
                if cnt % 50 == 0 or cnt == 1:
                    elapsed = time.time() - t0
                    speed = dl_size / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  [{cnt}/{total_count}] {rpath} "
                        f"(累计: {_format_size(dl_size)}, "
                        f"速度: {_format_size(int(speed))}/s)"
                    )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s 递增等待
                    logger.warning(
                        f"  下载失败 (第{attempt+1}次): {rpath}, "
                        f"错误: {e}, {wait_time}s 后重试..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"  下载失败 (已重试{max_retries}次): {rpath}, 错误: {e}"
                    )
                    with progress_lock:
                        progress["completed_count"] += 1
                        progress["failed_count"] += 1
                        progress["failed_files"].append(rpath)
                    return False

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {
            executor.submit(_download_one_file, f): f
            for f in selected_files
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                file_info = futures[future]
                logger.warning(f"  线程异常: {file_info[0]}, 错误: {e}")

    t1 = time.time()
    logger.info(
        f"下载完成! 共 {total_count} 个文件, "
        f"成功: {total_count - progress['failed_count']}, "
        f"跳过(已存在): {progress['skipped_count']}, "
        f"失败: {progress['failed_count']}, "
        f"总大小: {_format_size(progress['downloaded_size'])}, "
        f"耗时 {t1 - t0:.1f}s"
    )
    if progress["failed_files"]:
        logger.warning(f"  失败文件列表 ({len(progress['failed_files'])} 个):")
        for fp in progress["failed_files"][:20]:
            logger.warning(f"    {fp}")
        if len(progress["failed_files"]) > 20:
            logger.warning(f"    ... 共 {len(progress['failed_files'])} 个")
    logger.info(f"路径: {download_dir}")
    return download_dir


# ======================================================================
# Tokenize 工具函数
# ======================================================================

def init_tokenizer(tokenizer_path: str):
    """初始化 Llama2 tokenizer。"""
    from transformers import AutoTokenizer

    logger.info(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Llama2 tokenizer 信息
    logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    logger.info(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    logger.info(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    logger.info(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"  pad_token 未设置, 使用 eos_token 作为 pad_token")

    return tokenizer


def tokenize_text(text: str, tokenizer, max_seq_len: int, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """
    对单条文本进行 tokenize。

    Args:
        text: 输入文本
        tokenizer: tokenizer 实例
        max_seq_len: 最大序列长度
        add_bos: 是否在开头添加 BOS token
        add_eos: 是否在结尾添加 EOS token

    Returns:
        token id 列表
    """
    # 不让 tokenizer 自动加特殊 token，手动控制
    ids = tokenizer.encode(text, add_special_tokens=False)

    # 预留 BOS/EOS 的位置
    reserved = int(add_bos) + int(add_eos)
    if len(ids) > max_seq_len - reserved:
        ids = ids[: max_seq_len - reserved]

    if add_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    if add_eos and tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]

    return ids


# ======================================================================
# 流式 Tokenize (适合超大数据集)
# ======================================================================

def tokenize_streaming_to_numpy(
    tokenizer_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
    shard_size: int = 100_000_000,  # 每个 shard 的 token 数量 (约 400MB)
    subsets: Optional[list[str]] = None,
    use_mirror: bool = False,
):
    """
    流式下载 + tokenize，直接输出 numpy memmap 文件。
    不需要先下载到本地，节省磁盘空间。

    输出文件结构:
        output_dir/
            shard_000.npy    # 第一个 shard
            shard_001.npy    # 第二个 shard
            ...
            metadata.json    # 元数据 (总 token 数、shard 信息等)

    Args:
        tokenizer_path: Llama2 tokenizer 路径
        output_dir: 输出目录
        max_seq_len: 最大序列长度 (单个文档截断长度)
        shard_size: 每个 shard 的 token 数量
        subsets: 要处理的子集
        use_mirror: 是否使用 HF 镜像
    """
    from datasets import load_dataset

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    os.makedirs(output_dir, exist_ok=True)

    # 初始化 tokenizer
    tokenizer = init_tokenizer(tokenizer_path)

    # 流式加载数据集
    logger.info(f"流式加载数据集 {DATASET_ID} ...")
    load_kwargs = {
        "path": DATASET_ID,
        "streaming": True,
        "split": "train",
    }
    if subsets:
        # Dolmino-Mix-1124 可能有多个 config/subset
        # 尝试逐个加载并合并
        logger.info(f"指定子集: {subsets}")

    ds = load_dataset(**load_kwargs)

    # Tokenize 并写入 shard
    shard_idx = 0
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
    current_shard = np.memmap(shard_path, dtype=np.uint32, mode="w+", shape=(shard_size,))
    offset_in_shard = 0
    total_tokens = 0
    total_docs = 0
    shard_info = []

    t0 = time.time()
    log_interval = 50000

    for sample in ds:
        # Dolmino-Mix-1124 的主要文本字段是 "text"
        text = sample.get("text", "")
        if not text or not text.strip():
            continue

        ids = tokenize_text(text, tokenizer, max_seq_len)
        n = len(ids)

        if n == 0:
            continue

        # 检查是否需要开新 shard
        if offset_in_shard + n > shard_size:
            # 截断当前 shard 到实际大小
            current_shard.flush()
            actual_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
            # 重新创建精确大小的文件
            _truncate_memmap(actual_path, offset_in_shard)
            shard_info.append({
                "shard_idx": shard_idx,
                "file": f"shard_{shard_idx:04d}.npy",
                "num_tokens": offset_in_shard,
            })
            logger.info(
                f"  Shard {shard_idx} 完成: {offset_in_shard:,} tokens"
            )

            # 开新 shard
            shard_idx += 1
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
            current_shard = np.memmap(shard_path, dtype=np.uint32, mode="w+", shape=(shard_size,))
            offset_in_shard = 0

        current_shard[offset_in_shard: offset_in_shard + n] = ids
        offset_in_shard += n
        total_tokens += n
        total_docs += 1

        if total_docs % log_interval == 0:
            elapsed = time.time() - t0
            speed = total_tokens / elapsed
            logger.info(
                f"  已处理 {total_docs:,} 文档, {total_tokens:,} tokens "
                f"({speed:,.0f} tokens/s), 当前 shard {shard_idx}"
            )

    # 最后一个 shard
    if offset_in_shard > 0:
        current_shard.flush()
        actual_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.npy")
        _truncate_memmap(actual_path, offset_in_shard)
        shard_info.append({
            "shard_idx": shard_idx,
            "file": f"shard_{shard_idx:04d}.npy",
            "num_tokens": offset_in_shard,
        })

    t1 = time.time()

    # 保存元数据
    metadata = {
        "dataset": DATASET_ID,
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_seq_len": max_seq_len,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(shard_info),
        "shards": shard_info,
        "dtype": "uint32",
        "elapsed_seconds": t1 - t0,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Tokenize 完成!")
    logger.info(f"  总文档数: {total_docs:,}")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  Shard 数: {len(shard_info)}")
    logger.info(f"  耗时: {t1 - t0:.1f}s ({total_tokens / (t1 - t0):,.0f} tokens/s)")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  元数据: {meta_path}")
    logger.info("=" * 60)


def _truncate_memmap(path: str, num_tokens: int):
    """将 memmap 文件截断到实际大小。"""
    actual_bytes = num_tokens * 4  # uint32 = 4 bytes
    with open(path, "r+b") as f:
        f.truncate(actual_bytes)


# ======================================================================
# JSONL 格式输出 (兼容项目现有流程)
# ======================================================================

def tokenize_streaming_to_jsonl(
    tokenizer_path: str,
    output_dir: str,
    max_seq_len: int = 4096,
    subsets: Optional[list[str]] = None,
    use_mirror: bool = False,
):
    """
    流式下载 + tokenize，输出 JSONL 格式。
    每行一个 JSON 对象，包含 input_ids 和元信息。

    输出文件:
        output_dir/dolmino_tokenized.jsonl

    Args:
        tokenizer_path: Llama2 tokenizer 路径
        output_dir: 输出目录
        max_seq_len: 最大序列长度
        subsets: 要处理的子集
        use_mirror: 是否使用 HF 镜像
    """
    from datasets import load_dataset

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = init_tokenizer(tokenizer_path)

    logger.info(f"流式加载数据集 {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, streaming=True, split="train")

    output_path = os.path.join(output_dir, "dolmino_tokenized.jsonl")
    total_tokens = 0
    total_docs = 0
    t0 = time.time()
    log_interval = 50000

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if not text or not text.strip():
                continue

            ids = tokenize_text(text, tokenizer, max_seq_len)
            if not ids:
                continue

            record = {
                "input_ids": ids,
                "length": len(ids),
                "source": sample.get("source", sample.get("metadata", {}).get("source", "unknown")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_tokens += len(ids)
            total_docs += 1

            if total_docs % log_interval == 0:
                elapsed = time.time() - t0
                speed = total_tokens / elapsed
                logger.info(
                    f"  已处理 {total_docs:,} 文档, {total_tokens:,} tokens "
                    f"({speed:,.0f} tokens/s)"
                )

    t1 = time.time()

    # 保存元数据
    metadata = {
        "dataset": DATASET_ID,
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": max_seq_len,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "output_file": "dolmino_tokenized.jsonl",
        "elapsed_seconds": t1 - t0,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Tokenize 完成!")
    logger.info(f"  总文档数: {total_docs:,}")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  耗时: {t1 - t0:.1f}s ({total_tokens / (t1 - t0):,.0f} tokens/s)")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 60)


# ======================================================================
# 从本地已下载的文件 tokenize
# ======================================================================

def tokenize_local_files(
    tokenizer_path: str,
    input_dir: str,
    output_dir: str,
    max_seq_len: int = 4096,
    output_format: str = "numpy",
    num_workers: int = 1,
    shard_size: int = 100_000_000,
):
    """
    对本地已下载的 Dolmino-Mix-1124 文件进行 tokenize。
    支持 .jsonl, .jsonl.gz, .parquet 格式。

    Args:
        tokenizer_path: Llama2 tokenizer 路径
        input_dir: 本地数据目录
        output_dir: 输出目录
        max_seq_len: 最大序列长度
        output_format: 输出格式 ("numpy" 或 "jsonl")
        num_workers: 并行 worker 数
        shard_size: numpy 格式下每个 shard 的 token 数
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有数据文件
    patterns = ["**/*.jsonl", "**/*.jsonl.gz", "**/*.parquet", "**/*.json.gz", "**/*.json.zst"]
    data_files = []
    for pattern in patterns:
        data_files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))

    if not data_files:
        logger.error(f"在 {input_dir} 中未找到数据文件!")
        return

    data_files.sort()
    logger.info(f"找到 {len(data_files)} 个数据文件")
    for f in data_files[:10]:
        logger.info(f"  {f}")
    if len(data_files) > 10:
        logger.info(f"  ... 共 {len(data_files)} 个文件")

    tokenizer = init_tokenizer(tokenizer_path)

    if output_format == "numpy":
        _tokenize_local_to_numpy(tokenizer, data_files, output_dir, max_seq_len, shard_size, num_workers, tokenizer_path)
    else:
        _tokenize_local_to_jsonl(tokenizer, data_files, output_dir, max_seq_len, num_workers, tokenizer_path)


def _read_data_file(filepath: str):
    """读取单个数据文件，返回文本迭代器。支持 .jsonl, .jsonl.gz, .json.gz, .json.zst, .parquet 格式。"""
    import gzip

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
            logger.warning(f"需要 pyarrow 来读取 parquet 文件: {filepath}")
    elif filepath.endswith(".zst"):
        # Zstandard 压缩格式 (Dolmino-Mix-1124 使用此格式)
        try:
            import zstandard as zstd
        except ImportError:
            logger.error(
                f"需要 zstandard 库来读取 .zst 文件: {filepath}\n"
                f"请安装: pip install zstandard"
            )
            return
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                import io
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


def _tokenize_single_file_worker(args_tuple):
    """
    多进程 worker：对单个数据文件进行 tokenize，输出临时 .npy 文件。

    Args:
        args_tuple: (file_idx, filepath, tokenizer_path, max_seq_len, tmp_dir)

    Returns:
        dict: {"file": filepath, "tmp_npy": 临时文件路径, "num_tokens": token数, "num_docs": 文档数}
    """
    file_idx, filepath, tokenizer_path, max_seq_len, tmp_dir, total_files = args_tuple

    # 每个 worker 进程独立加载 tokenizer（避免跨进程序列化问题）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先收集所有 token ids 到内存列表，再一次性写入 npy
    all_ids = []
    num_docs = 0
    t0 = time.time()

    for text in _read_data_file(filepath):
        ids = tokenize_text(text, tokenizer, max_seq_len)
        if len(ids) == 0:
            continue
        all_ids.extend(ids)
        num_docs += 1

    num_tokens = len(all_ids)
    t1 = time.time()

    if num_tokens == 0:
        logger.info(
            f"  Worker [{file_idx+1}/{total_files}] {os.path.basename(filepath)}: "
            f"0 tokens, 跳过"
        )
        return {
            "file": filepath,
            "tmp_npy": None,
            "num_tokens": 0,
            "num_docs": 0,
        }

    # 写入临时 npy 文件
    tmp_npy = os.path.join(tmp_dir, f"tmp_{file_idx:05d}.npy")
    arr = np.array(all_ids, dtype=np.uint32)
    np.save(tmp_npy, arr)

    speed = num_tokens / max(t1 - t0, 0.01)
    logger.info(
        f"  Worker [{file_idx+1}/{total_files}] {os.path.basename(filepath)}: "
        f"{num_docs:,} docs, {num_tokens:,} tokens, "
        f"{speed:,.0f} tok/s, {t1-t0:.1f}s"
    )

    return {
        "file": filepath,
        "tmp_npy": tmp_npy,
        "num_tokens": num_tokens,
        "num_docs": num_docs,
    }


def _tokenize_local_to_numpy(
    tokenizer,
    data_files: list[str],
    output_dir: str,
    max_seq_len: int,
    shard_size: int,
    num_workers: int,
    tokenizer_path: str,
):
    """将本地文件 tokenize 为 numpy memmap 格式（支持多进程并行）。"""
    t0 = time.time()

    # ---- 第一阶段：多进程并行 tokenize 每个文件 ----
    tmp_dir = os.path.join(output_dir, "_tmp_tokenize")
    os.makedirs(tmp_dir, exist_ok=True)

    actual_workers = min(num_workers, len(data_files))
    logger.info(
        f"开始多进程 tokenize: {len(data_files)} 个文件, "
        f"{actual_workers} 个 worker 进程"
    )

    worker_args = [
        (i, fp, tokenizer_path, max_seq_len, tmp_dir, len(data_files))
        for i, fp in enumerate(data_files)
    ]

    if actual_workers > 1:
        # 使用 spawn 方式创建子进程，避免 fork 后 tokenizer 死锁
        ctx = mp.get_context("spawn")
        results = []
        with ctx.Pool(processes=actual_workers) as pool:
            for r in pool.imap_unordered(_tokenize_single_file_worker, worker_args):
                results.append(r)
    else:
        # 单进程模式，直接顺序执行
        results = [_tokenize_single_file_worker(a) for a in worker_args]

    # 按 file_idx 排序，保证 shard 合并顺序一致
    results.sort(key=lambda r: r["file"])

    t_tokenize = time.time()
    total_docs = sum(r["num_docs"] for r in results)
    total_tokens = sum(r["num_tokens"] for r in results)
    logger.info(
        f"Tokenize 阶段完成: {total_docs:,} docs, {total_tokens:,} tokens, "
        f"耗时 {t_tokenize - t0:.1f}s "
        f"({total_tokens / max(t_tokenize - t0, 1):,.0f} tokens/s)"
    )

    # ---- 第二阶段：多线程并行合并临时 npy 文件到最终 shard ----
    logger.info(f"开始合并到 shard (每个 shard {shard_size:,} tokens) ...")

    # 2a. 预计算每个 shard 的写入计划
    # write_op = (tmp_npy_path, tmp_offset, shard_offset, chunk_size)
    write_ops_by_shard = {}  # shard_idx -> list of write_ops
    shard_token_counts = {}  # shard_idx -> 实际 token 数
    shard_idx = 0
    offset_in_shard = 0

    valid_results = [r for r in results if r["tmp_npy"] is not None]

    for r in valid_results:
        tmp_offset = 0
        remaining = r["num_tokens"]

        while remaining > 0:
            space_in_shard = shard_size - offset_in_shard
            chunk = min(remaining, space_in_shard)

            if shard_idx not in write_ops_by_shard:
                write_ops_by_shard[shard_idx] = []
            write_ops_by_shard[shard_idx].append(
                (r["tmp_npy"], tmp_offset, offset_in_shard, chunk)
            )

            offset_in_shard += chunk
            tmp_offset += chunk
            remaining -= chunk

            shard_token_counts[shard_idx] = offset_in_shard

            # shard 满了，切换到下一个
            if offset_in_shard >= shard_size:
                shard_idx += 1
                offset_in_shard = 0

    num_shards = max(write_ops_by_shard.keys()) + 1 if write_ops_by_shard else 0
    logger.info(f"  预计算完成: {num_shards} 个 shard, {len(valid_results)} 个临时文件")

    # 2b. 定义单个 shard 的写入函数
    def _write_one_shard(sid):
        """写入单个 shard 的所有数据块（在线程中执行）。"""
        ops = write_ops_by_shard[sid]
        n_tokens = shard_token_counts[sid]
        s_path = os.path.join(output_dir, f"shard_{sid:04d}.npy")
        shard_mm = np.memmap(s_path, dtype=np.uint32, mode="w+", shape=(shard_size,))

        # 缓存已加载的 tmp 文件，避免同一个文件重复 np.load
        tmp_cache = {}
        for (tmp_npy, t_off, s_off, chunk) in ops:
            if tmp_npy not in tmp_cache:
                tmp_cache[tmp_npy] = np.load(tmp_npy)
            shard_mm[s_off: s_off + chunk] = tmp_cache[tmp_npy][t_off: t_off + chunk]

        shard_mm.flush()
        del tmp_cache  # 释放内存

        # 如果不是满 shard，截断
        if n_tokens < shard_size:
            _truncate_memmap(s_path, n_tokens)

        logger.info(f"  Shard {sid} 完成: {n_tokens:,} tokens")
        return {
            "shard_idx": sid,
            "file": f"shard_{sid:04d}.npy",
            "num_tokens": n_tokens,
        }

    # 2c. 多线程并行写入各个 shard
    shard_info = []
    merge_workers = min(num_shards, max(actual_workers, 4))
    if num_shards > 1:
        logger.info(f"  使用 {merge_workers} 个线程并行写入 {num_shards} 个 shard ...")
        with ThreadPoolExecutor(max_workers=merge_workers) as executor:
            futures = {
                executor.submit(_write_one_shard, sid): sid
                for sid in sorted(write_ops_by_shard.keys())
            }
            for future in as_completed(futures):
                shard_info.append(future.result())
    elif num_shards == 1:
        shard_info.append(_write_one_shard(0))

    # 按 shard_idx 排序
    shard_info.sort(key=lambda x: x["shard_idx"])

    # 清理临时文件
    for r in valid_results:
        try:
            os.remove(r["tmp_npy"])
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    t1 = time.time()

    metadata = {
        "dataset": DATASET_ID,
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_seq_len": max_seq_len,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(shard_info),
        "shards": shard_info,
        "dtype": "uint32",
        "elapsed_seconds": t1 - t0,
        "num_workers": actual_workers,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Tokenize 完成! (numpy memmap, {actual_workers} workers)")
    logger.info(f"  总文档数: {total_docs:,}")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  Shard 数: {len(shard_info)}")
    logger.info(f"  Tokenize 耗时: {t_tokenize - t0:.1f}s ({total_tokens / max(t_tokenize - t0, 1):,.0f} tokens/s)")
    logger.info(f"  合并 + 总耗时: {t1 - t0:.1f}s")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)


def merge_tmp_to_single_bin(
    tmp_dir: str,
    output_dir: str,
    output_filename: str = "train.bin",
    tokenizer_path: Optional[str] = None,
    max_seq_len: int = 4096,
    num_workers: int = 32,
):
    """
    将 _tmp_tokenize/ 目录中已有的临时 .npy 文件合并为单个 train.bin 文件。
    跳过 tokenize 阶段，直接合并。

    输出格式: 原始 uint32 二进制文件 (无 numpy header)，可直接 memmap 读取。

    Args:
        tmp_dir: 临时 npy 文件所在目录
        output_dir: 输出目录
        output_filename: 输出文件名 (默认 train.bin)
        tokenizer_path: tokenizer 路径 (仅用于写 metadata)
        max_seq_len: 最大序列长度 (仅用于写 metadata)
        num_workers: 并行读取线程数
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有临时 npy 文件
    npy_files = sorted(glob.glob(os.path.join(tmp_dir, "tmp_*.npy")))
    if not npy_files:
        logger.error(f"在 {tmp_dir} 中未找到临时 .npy 文件!")
        return

    logger.info(f"找到 {len(npy_files)} 个临时 .npy 文件")
    logger.info(f"  目录: {tmp_dir}")
    logger.info(f"  输出: {os.path.join(output_dir, output_filename)}")

    # 第一步：并行读取所有 npy 文件的大小（不加载数据）
    t0 = time.time()
    logger.info("扫描临时文件大小...")

    file_sizes = []  # (filepath, num_tokens)
    skipped = 0
    for f in npy_files:
        try:
            # numpy .npy 文件: 读 header 获取 shape，不加载数据
            arr = np.load(f, mmap_mode='r')
            file_sizes.append((f, len(arr)))
            del arr
        except (ValueError, Exception) as e:
            # 跳过损坏/不完整的文件（如写入中断导致文件头声明大小 > 实际文件大小）
            skipped += 1
            logger.warning(f"  跳过损坏文件: {os.path.basename(f)} ({e})")
    if skipped:
        logger.warning(f"  共跳过 {skipped} 个损坏文件, 有效文件 {len(file_sizes)} 个")

    total_tokens = sum(s for _, s in file_sizes)
    logger.info(f"  总 token 数: {total_tokens:,} ({total_tokens * 4 / 1024**3:.2f} GB)")

    # 第二步：创建输出文件并顺序写入
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"创建输出文件: {output_path} ({total_tokens * 4 / 1024**3:.2f} GB)")

    # 使用 memmap 创建目标文件
    out_mm = np.memmap(output_path, dtype=np.uint32, mode='w+', shape=(total_tokens,))

    # 顺序写入（保证文件顺序一致）
    offset = 0
    for i, (fpath, n_tokens) in enumerate(file_sizes):
        if n_tokens == 0:
            continue
        arr = np.load(fpath)
        out_mm[offset: offset + n_tokens] = arr
        offset += n_tokens

        if (i + 1) % 100 == 0 or (i + 1) == len(file_sizes):
            elapsed = time.time() - t0
            pct = offset / total_tokens * 100
            logger.info(
                f"  [{i+1}/{len(file_sizes)}] {pct:.1f}% 完成, "
                f"{offset:,}/{total_tokens:,} tokens, {elapsed:.1f}s"
            )

    out_mm.flush()
    del out_mm

    t1 = time.time()

    # 保存元数据
    metadata = {
        "dataset": DATASET_ID,
        "tokenizer": tokenizer_path or "unknown",
        "max_seq_len": max_seq_len,
        "total_tokens": total_tokens,
        "num_files_merged": len(npy_files),
        "output_file": output_filename,
        "dtype": "uint32",
        "elapsed_seconds": t1 - t0,
    }

    # 尝试获取 tokenizer 信息
    if tokenizer_path:
        try:
            tokenizer = init_tokenizer(tokenizer_path)
            metadata["vocab_size"] = tokenizer.vocab_size
            metadata["bos_token_id"] = tokenizer.bos_token_id
            metadata["eos_token_id"] = tokenizer.eos_token_id
        except Exception:
            pass

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"合并完成!")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  文件大小: {total_tokens * 4 / 1024**3:.2f} GB")
    logger.info(f"  合并文件数: {len(npy_files)}")
    logger.info(f"  耗时: {t1 - t0:.1f}s")
    logger.info(f"  输出文件: {output_path}")
    logger.info(f"  元数据: {meta_path}")
    logger.info("=" * 60)


def _tokenize_single_file_to_jsonl_worker(args_tuple):
    """
    多进程 worker：对单个数据文件进行 tokenize，输出临时 .jsonl 文件。
    """
    file_idx, filepath, tokenizer_path, max_seq_len, tmp_dir, total_files = args_tuple

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tmp_jsonl = os.path.join(tmp_dir, f"tmp_{file_idx:05d}.jsonl")
    num_docs = 0
    num_tokens = 0
    t0 = time.time()

    with open(tmp_jsonl, "w", encoding="utf-8") as out_f:
        for text in _read_data_file(filepath):
            ids = tokenize_text(text, tokenizer, max_seq_len)
            if not ids:
                continue
            record = {
                "input_ids": ids,
                "length": len(ids),
                "source": os.path.basename(filepath),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_tokens += len(ids)
            num_docs += 1

    t1 = time.time()
    speed = num_tokens / max(t1 - t0, 0.01)
    logger.info(
        f"  Worker [{file_idx+1}/{total_files}] {os.path.basename(filepath)}: "
        f"{num_docs:,} docs, {num_tokens:,} tokens, "
        f"{speed:,.0f} tok/s, {t1-t0:.1f}s"
    )

    return {
        "file": filepath,
        "tmp_jsonl": tmp_jsonl,
        "num_tokens": num_tokens,
        "num_docs": num_docs,
    }


def _tokenize_local_to_jsonl(
    tokenizer,
    data_files: list[str],
    output_dir: str,
    max_seq_len: int,
    num_workers: int,
    tokenizer_path: str,
):
    """将本地文件 tokenize 为 JSONL 格式（支持多进程并行）。"""
    t0 = time.time()

    tmp_dir = os.path.join(output_dir, "_tmp_tokenize")
    os.makedirs(tmp_dir, exist_ok=True)

    actual_workers = min(num_workers, len(data_files))
    logger.info(
        f"开始多进程 tokenize (JSONL): {len(data_files)} 个文件, "
        f"{actual_workers} 个 worker 进程"
    )

    worker_args = [
        (i, fp, tokenizer_path, max_seq_len, tmp_dir, len(data_files))
        for i, fp in enumerate(data_files)
    ]

    if actual_workers > 1:
        # 使用 spawn 方式创建子进程，避免 fork 后 tokenizer 死锁
        ctx = mp.get_context("spawn")
        results = []
        with ctx.Pool(processes=actual_workers) as pool:
            for r in pool.imap_unordered(_tokenize_single_file_to_jsonl_worker, worker_args):
                results.append(r)
        # 按 file_idx 排序，保证合并顺序一致
        results.sort(key=lambda r: r["file"])
    else:
        results = [_tokenize_single_file_to_jsonl_worker(a) for a in worker_args]

    t_tokenize = time.time()
    total_docs = sum(r["num_docs"] for r in results)
    total_tokens = sum(r["num_tokens"] for r in results)

    # 合并临时 jsonl 文件
    output_path = os.path.join(output_dir, "dolmino_tokenized.jsonl")
    logger.info(f"合并 {len(results)} 个临时文件到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for r in results:
            if r["num_docs"] == 0:
                continue
            with open(r["tmp_jsonl"], "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
            os.remove(r["tmp_jsonl"])

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    t1 = time.time()

    metadata = {
        "dataset": DATASET_ID,
        "tokenizer": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": max_seq_len,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "output_file": "dolmino_tokenized.jsonl",
        "elapsed_seconds": t1 - t0,
        "num_workers": actual_workers,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Tokenize 完成! (JSONL, {actual_workers} workers)")
    logger.info(f"  总文档数: {total_docs:,}")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  Tokenize 耗时: {t_tokenize - t0:.1f}s ({total_tokens / max(t_tokenize - t0, 1):,.0f} tokens/s)")
    logger.info(f"  合并 + 总耗时: {t1 - t0:.1f}s")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 60)


# ======================================================================
# 数据验证
# ======================================================================

def verify_output(output_dir: str, output_format: str = "numpy"):
    """验证 tokenize 输出的正确性。"""
    meta_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(meta_path):
        logger.error(f"元数据文件不存在: {meta_path}")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info("=" * 60)
    logger.info("验证 tokenize 输出")
    logger.info(f"  数据集: {metadata['dataset']}")
    logger.info(f"  Tokenizer: {metadata['tokenizer']}")
    logger.info(f"  Vocab size: {metadata['vocab_size']}")
    logger.info(f"  Max seq len: {metadata['max_seq_len']}")
    logger.info(f"  总文档数: {metadata['total_docs']:,}")
    logger.info(f"  总 token 数: {metadata['total_tokens']:,}")

    if output_format == "numpy" and "shards" in metadata:
        total_verified = 0
        for shard in metadata["shards"]:
            shard_path = os.path.join(output_dir, shard["file"])
            if os.path.exists(shard_path):
                data = np.memmap(shard_path, dtype=np.uint32, mode="r")
                total_verified += len(data)
                # 检查前几个 token
                logger.info(
                    f"  Shard {shard['shard_idx']}: {len(data):,} tokens, "
                    f"前10个: {data[:10].tolist()}"
                )
            else:
                logger.warning(f"  Shard 文件不存在: {shard_path}")

        logger.info(f"  验证 token 总数: {total_verified:,} (元数据: {metadata['total_tokens']:,})")
        if total_verified == metadata["total_tokens"]:
            logger.info("  ✅ 验证通过!")
        else:
            logger.warning("  ⚠️ Token 数量不匹配!")

    elif output_format == "jsonl":
        output_file = os.path.join(output_dir, metadata.get("output_file", "dolmino_tokenized.jsonl"))
        if os.path.exists(output_file):
            # 读取前几行验证
            with open(output_file) as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    d = json.loads(line)
                    logger.info(
                        f"  样本 {i}: length={d['length']}, "
                        f"source={d.get('source', 'N/A')}, "
                        f"前10个 token: {d['input_ids'][:10]}"
                    )
            logger.info("  ✅ JSONL 文件可读!")
        else:
            logger.warning(f"  输出文件不存在: {output_file}")

    logger.info("=" * 60)


# ======================================================================
# 主函数
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="下载并 tokenize Dolmino-Mix-1124 数据集 (OLMo2 CAST 退火阶段数据)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基本参数
    parser.add_argument(
        "--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf",
        help="Llama2 tokenizer 路径 (HuggingFace ID 或本地路径)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--output_format", type=str, default="numpy", choices=["numpy", "jsonl"],
        help="输出格式: numpy (memmap) 或 jsonl",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096,
        help="最大序列长度 (单个文档截断长度)",
    )

    # 下载参数
    parser.add_argument(
        "--download_only", action="store_true",
        help="只下载不 tokenize",
    )
    parser.add_argument(
        "--subsets", nargs="*", default=None,
        help="要处理的子集名称 (如 dclm flan math code wiki)",
    )
    parser.add_argument(
        "--use_mirror", action="store_true",
        help="使用 HuggingFace 镜像站 (国内加速)",
    )
    parser.add_argument(
        "--max_size_gb", type=float, default=None,
        help="最大下载大小 (GB), 例如 --max_size_gb 1024 表示不超过 1TB",
    )

    # 本地文件处理
    parser.add_argument(
        "--local_input_dir", type=str, default=None,
        help="本地已下载的数据目录 (跳过下载，直接 tokenize)",
    )

    # 性能参数
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="并行 worker 数 (用于本地文件处理)",
    )
    parser.add_argument(
        "--download_workers", type=int, default=8,
        help="下载并发线程数 (默认 8, 建议 4-16)",
    )
    parser.add_argument(
        "--shard_size", type=int, default=100_000_000,
        help="numpy 格式下每个 shard 的 token 数 (默认 1 亿, 约 400MB)",
    )

    # 合并模式
    parser.add_argument(
        "--merge_only", action="store_true",
        help="跳过 tokenize，直接将 _tmp_tokenize/ 中的临时 npy 文件合并为单个 train.bin",
    )
    parser.add_argument(
        "--merge_input_dir", type=str, default=None,
        help="临时 npy 文件所在目录 (默认为 output_dir/_tmp_tokenize)",
    )
    parser.add_argument(
        "--output_filename", type=str, default="train.bin",
        help="合并输出的文件名 (默认 train.bin)",
    )

    # 验证
    parser.add_argument(
        "--verify", action="store_true",
        help="验证已有的 tokenize 输出",
    )

    args = parser.parse_args()

    # 模式 0: 仅合并临时文件
    if args.merge_only:
        merge_input = args.merge_input_dir or os.path.join(args.output_dir, "_tmp_tokenize")
        merge_tmp_to_single_bin(
            tmp_dir=merge_input,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
        )
        return

    # 模式 1: 验证
    if args.verify:
        verify_output(args.output_dir, args.output_format)
        return

    # 模式 2: 只下载
    if args.download_only:
        download_dataset(args.output_dir, args.subsets, args.use_mirror, args.max_size_gb, args.download_workers)
        return

    # 模式 3: 从本地文件 tokenize
    if args.local_input_dir:
        tokenize_local_files(
            tokenizer_path=args.tokenizer_path,
            input_dir=args.local_input_dir,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            output_format=args.output_format,
            num_workers=args.num_workers,
            shard_size=args.shard_size,
        )
        verify_output(args.output_dir, args.output_format)
        return

    # 模式 4: 下载 + tokenize
    # 如果指定了 max_size_gb，先下载（带大小限制）再本地 tokenize
    # 否则使用流式下载 + tokenize
    if args.max_size_gb is not None:
        logger.info(
            f"指定了 --max_size_gb {args.max_size_gb}, "
            f"将先下载（限制 {args.max_size_gb} GB）再本地 tokenize"
        )
        # 第一步: 下载
        download_dir = download_dataset(
            args.output_dir, args.subsets, args.use_mirror, args.max_size_gb,
            args.download_workers,
        )
        # 第二步: 本地 tokenize
        tokenize_local_files(
            tokenizer_path=args.tokenizer_path,
            input_dir=download_dir,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            output_format=args.output_format,
            num_workers=args.num_workers,
            shard_size=args.shard_size,
        )
    elif args.output_format == "numpy":
        tokenize_streaming_to_numpy(
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            shard_size=args.shard_size,
            subsets=args.subsets,
            use_mirror=args.use_mirror,
        )
    else:
        tokenize_streaming_to_jsonl(
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            subsets=args.subsets,
            use_mirror=args.use_mirror,
        )

    # 自动验证
    verify_output(args.output_dir, args.output_format)


if __name__ == "__main__":
    main()
