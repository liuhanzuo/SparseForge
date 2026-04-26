#!/usr/bin/env python3
"""准备 FineWeb / FineWeb-Edu 数据集，使用 LLaMA-2 tokenizer 进行 tokenize。

使用 **多线程 streaming 模式** 从 HuggingFace 并行下载多个分片数据，
只取所需的 token 数量，避免下载整个数据集（FineWeb-Edu 完整版有 ~1.3T tokens）。

输出:
  - train.bin (uint16, LLaMA-2 vocab ~32k，uint16 足够)
  - val.bin   (uint16)
  - dtype.txt (内容为 "uint16")
  - metadata.json (数据集元信息)

用法:
  # 从 FineWeb-Edu 多线程下载 12B tokens（推荐，8 线程并行）
  python scripts/prepare_fineweb_llama.py \
      --tokenizer NousResearch/Llama-2-7b-hf \
      --dataset_name HuggingFaceFW/fineweb-edu \
      --max_tokens 12_000_000_000 \
      --num_workers 8 \
      --output_dir data/fineweb-edu-llama \
      --num_proc 64

  # 从 FineWeb 多线程下载 12B tokens
  python scripts/prepare_fineweb_llama.py \
      --tokenizer NousResearch/Llama-2-7b-hf \
      --dataset_name HuggingFaceFW/fineweb \
      --max_tokens 12_000_000_000 \
      --num_workers 8 \
      --output_dir data/fineweb-12B-llama \
      --num_proc 64

  # 使用预定义子集（非 streaming，会下载整个子集）
  python scripts/prepare_fineweb_llama.py \
      --tokenizer NousResearch/Llama-2-7b-hf \
      --subset sample-10BT \
      --output_dir data/fineweb-10BT-llama \
      --num_proc 64

注意:
  - LLaMA-2 vocab_size=32000，uint16 (max 65535) 完全够用
  - 每条文本末尾追加 EOS token
  - 默认从收集到的数据中切出 0.5% 作为 val split
  - 指定 --max_tokens 时自动启用 streaming 模式
  - --num_workers 控制并行下载线程数（默认 8）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoTokenizer


# ============================================================================
# 线程安全的 Tokenizer 管理（每个线程独立实例，避免 GIL 竞争）
# ============================================================================
_THREAD_LOCAL = threading.local()


def _get_thread_tokenizer(name_or_path: str):
    """获取当前线程的 tokenizer 实例（线程安全，每线程一个实例）。"""
    if not hasattr(_THREAD_LOCAL, "tokenizer") or _THREAD_LOCAL.tokenizer is None:
        tok = AutoTokenizer.from_pretrained(
            name_or_path, use_fast=True, trust_remote_code=True
        )
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer 没有 eos_token_id，无法追加 EOS")
        _THREAD_LOCAL.tokenizer = tok
    return _THREAD_LOCAL.tokenizer


# 全局单例（用于非多线程场景）
_TOKENIZER = None


def _get_tokenizer(name_or_path: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"[tokenizer] 加载 tokenizer: {name_or_path}")
        tok = AutoTokenizer.from_pretrained(
            name_or_path, use_fast=True, trust_remote_code=True
        )
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer 没有 eos_token_id，无法追加 EOS")
        _TOKENIZER = tok
        print(f"[tokenizer] ✓ 加载完成 (vocab_size={len(tok)}, eos_id={tok.eos_token_id})")
    return _TOKENIZER


def _tokenize_batch(batch, tokenizer_path: str):
    """对一个 batch 的文本进行 tokenize，末尾追加 EOS。"""
    tok = _get_tokenizer(tokenizer_path)
    texts = batch["text"]
    encoded = tok(texts, add_special_tokens=False)
    ids_list = encoded["input_ids"]
    eos = int(tok.eos_token_id)
    ids_list = [ids + [eos] for ids in ids_list]
    lens = [len(ids) for ids in ids_list]
    return {"ids": ids_list, "len": lens}


# ============================================================================
# 共享的原子 token 计数器（多线程安全）
# ============================================================================
class AtomicTokenCounter:
    """线程安全的 token 计数器，用于多个 worker 共享全局进度。"""

    def __init__(self, max_tokens: int):
        self._lock = threading.Lock()
        self._count = 0
        self._max = max_tokens
        self._done = False

    def add(self, n: int) -> bool:
        """增加 n 个 token，返回 True 表示还可以继续，False 表示已达到上限。"""
        with self._lock:
            if self._done:
                return False
            self._count += n
            if self._count >= self._max:
                self._done = True
            return True

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done


# ============================================================================
# 获取数据集的所有 parquet 文件 URL 列表
# ============================================================================
def _get_parquet_urls(dataset_name: str, subset: str | None = None) -> list[str]:
    """
    获取数据集的所有 parquet 文件 URL。

    优先使用 HuggingFace datasets-server REST API（极快，秒级返回），
    如果失败则回退到 huggingface_hub 的 list_repo_files()。

    返回: parquet 文件 URL 列表
    """
    # ---- 方案 1: datasets-server REST API（推荐，极快） ----
    try:
        api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
        print(f"  [parquet] 尝试 datasets-server API: {api_url}")
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        parquet_urls = []
        # API 返回格式: {"parquet_files": [{"dataset": ..., "config": ..., "split": ..., "url": ..., "filename": ...}, ...]}
        for pf in data.get("parquet_files", []):
            # 只取 train split
            if pf.get("split") != "train":
                continue
            # 如果指定了 subset/config，过滤
            if subset and subset != "default":
                if pf.get("config") != subset:
                    continue
            url = pf.get("url", "")
            if url:
                parquet_urls.append(url)

        if parquet_urls:
            parquet_urls.sort()
            print(f"  [parquet] ✓ datasets-server API 返回 {len(parquet_urls)} 个 parquet 文件")
            return parquet_urls
        else:
            print(f"  [parquet] ⚠ datasets-server API 返回 0 个文件，尝试备选方案...")
    except Exception as e:
        print(f"  [parquet] ⚠ datasets-server API 失败 ({e})，尝试备选方案...")

    # ---- 方案 2: 直接构造 URL 模式（适用于已知结构的数据集） ----
    # FineWeb-Edu 的文件结构: data/train-XXXXX-of-YYYYY-*.parquet
    # 先尝试用 HuggingFace Hub API 的 list_repo_tree（只列 data/ 目录，不递归整个仓库）
    try:
        print(f"  [parquet] 尝试 list_repo_files()...")
        api = HfApi()
        all_files = api.list_repo_files(
            repo_id=dataset_name,
            repo_type="dataset",
        )

        parquet_urls = []
        for fname in all_files:
            if not fname.endswith(".parquet"):
                continue
            if subset and subset != "default":
                if not fname.startswith(f"{subset}/") and not fname.startswith(f"data/{subset}/"):
                    continue
            url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{fname}"
            parquet_urls.append(url)

        if not parquet_urls:
            for fname in all_files:
                if fname.endswith(".parquet") and "train" in fname:
                    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{fname}"
                    parquet_urls.append(url)

        parquet_urls.sort()
        print(f"  [parquet] ✓ list_repo_files 返回 {len(parquet_urls)} 个 parquet 文件")
        return parquet_urls
    except Exception as e:
        raise RuntimeError(f"无法获取 parquet 文件列表: {e}")


# ============================================================================
# 单个 worker：streaming 下载并 tokenize 指定的 parquet 文件
# ============================================================================
def _worker_stream_shard(
    worker_id: int,
    data_files: list[str],
    tokenizer_path: str,
    counter: AtomicTokenCounter,
    batch_size: int = 1000,
) -> tuple[list[int], int]:
    """
    单个 worker 线程：streaming 下载指定的 parquet 文件并 tokenize。

    每个 worker 负责不同的 parquet 文件，实现真正的并行下载。

    返回: (token_ids_list, n_docs)
    """
    tok = _get_thread_tokenizer(tokenizer_path)
    eos = int(tok.eos_token_id)

    local_ids: list[int] = []
    n_docs = 0
    text_buffer: list[str] = []

    # 直接加载指定的 parquet 文件（streaming 模式）
    ds_stream = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True,
        columns=["text"],  # 只读取 text 列，避免 schema 不一致
    )

    for example in ds_stream:
        # 全局已达标，提前退出
        if counter.done:
            break

        text = example.get("text", "")
        if not text or not text.strip():
            continue

        text_buffer.append(text)

        if len(text_buffer) >= batch_size:
            encoded = tok(text_buffer, add_special_tokens=False)
            batch_tokens = 0
            for ids in encoded["input_ids"]:
                doc_ids = ids + [eos]
                local_ids.extend(doc_ids)
                batch_tokens += len(doc_ids)
                n_docs += 1

            text_buffer = []

            # 更新全局计数器
            if not counter.add(batch_tokens):
                break

    # 处理剩余 buffer
    if text_buffer and not counter.done:
        encoded = tok(text_buffer, add_special_tokens=False)
        batch_tokens = 0
        for ids in encoded["input_ids"]:
            doc_ids = ids + [eos]
            local_ids.extend(doc_ids)
            batch_tokens += len(doc_ids)
            n_docs += 1
        counter.add(batch_tokens)

    return local_ids, n_docs


# ============================================================================
# 多线程 streaming 下载 + tokenize
# ============================================================================
def _parallel_stream_and_tokenize(
    dataset_name: str,
    subset: str | None,
    max_tokens: int,
    tokenizer_path: str,
    num_workers: int = 8,
    seed: int = 42,
    cache_dir: str | None = None,
    batch_size: int = 1000,
) -> list[int]:
    """
    多线程并行 streaming 下载 + tokenize。

    通过 HuggingFace Hub API 获取数据集的所有 parquet 文件列表，
    然后将文件均匀分配给 num_workers 个线程，每个线程只下载自己负责的文件。
    所有线程共享一个原子计数器，达到 max_tokens 后全部停止。

    返回: 所有 token id 的列表
    """
    print(f"[parallel] 数据集: {dataset_name}" +
          (f" / {subset}" if subset and subset != "default" else ""))
    print(f"[parallel] 目标: {max_tokens:,} tokens")

    # Step 1: 获取所有 parquet 文件 URL
    print(f"[parallel] 正在获取 parquet 文件列表...")
    parquet_urls = _get_parquet_urls(dataset_name, subset)
    print(f"[parallel] 共 {len(parquet_urls)} 个 parquet 文件")

    if not parquet_urls:
        raise RuntimeError(
            f"未找到 parquet 文件！请检查数据集名称 '{dataset_name}' 和 subset '{subset}'"
        )

    # Step 2: 将文件均匀分配给各 worker
    actual_workers = min(num_workers, len(parquet_urls))
    files_per_worker = math.ceil(len(parquet_urls) / actual_workers)
    worker_file_groups = []
    for i in range(actual_workers):
        start = i * files_per_worker
        end = min(start + files_per_worker, len(parquet_urls))
        if start < len(parquet_urls):
            worker_file_groups.append(parquet_urls[start:end])

    print(f"[parallel] 启动 {len(worker_file_groups)} 个下载线程")
    for i, group in enumerate(worker_file_groups):
        print(f"  [worker {i}] 负责 {len(group)} 个文件")

    counter = AtomicTokenCounter(max_tokens)
    t0 = time.time()

    # 进度条（在主线程更新）
    pbar = tqdm(total=max_tokens, unit="tok", desc="多线程下载+tokenize",
                unit_scale=True, smoothing=0.1)

    # 启动多线程
    futures = {}
    with ThreadPoolExecutor(max_workers=len(worker_file_groups)) as executor:
        for i, file_group in enumerate(worker_file_groups):
            future = executor.submit(
                _worker_stream_shard,
                worker_id=i,
                data_files=file_group,
                tokenizer_path=tokenizer_path,
                counter=counter,
                batch_size=batch_size,
            )
            futures[future] = i

        # 定期更新进度条
        last_count = 0
        while not all(f.done() for f in futures):
            time.sleep(0.5)
            current = counter.count
            if current > last_count:
                pbar.update(current - last_count)
                last_count = current

        # 最终更新
        current = counter.count
        if current > last_count:
            pbar.update(current - last_count)

    pbar.close()

    # 收集所有 worker 的结果
    all_ids: list[int] = []
    total_docs = 0
    for future in futures:
        worker_id = futures[future]
        try:
            ids, n_docs = future.result()
            all_ids.extend(ids)
            total_docs += n_docs
            print(f"  [worker {worker_id}] {len(ids):,} tokens, {n_docs:,} 文档")
        except Exception as e:
            print(f"  [worker {worker_id}] ❌ 出错: {e}")

    t1 = time.time()

    # 截断到 max_tokens
    if len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]

    print(f"[parallel] ✓ 完成: {len(all_ids):,} tokens, {total_docs:,} 文档, "
          f"耗时 {t1-t0:.1f}s")
    print(f"[parallel]   速度: {len(all_ids)/(t1-t0):,.0f} tokens/s, "
          f"{total_docs/(t1-t0):,.0f} docs/s")
    print(f"[parallel]   使用 {len(worker_file_groups)} 个并行线程")

    return all_ids


# ============================================================================
# 单线程 streaming（保留作为 fallback）
# ============================================================================
def _stream_and_tokenize(
    dataset_name: str,
    subset: str | None,
    max_tokens: int,
    tokenizer_path: str,
    seed: int = 42,
    cache_dir: str | None = None,
    batch_size: int = 1000,
) -> list[int]:
    """
    单线程 streaming 模式（fallback，当 num_workers=1 时使用）。
    """
    tok = _get_tokenizer(tokenizer_path)
    eos = int(tok.eos_token_id)

    load_kwargs = {
        "path": dataset_name,
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
        "columns": ["text"],  # 只读取 text 列，避免不同 parquet 文件 schema 不一致导致 cast 错误
    }
    if subset and subset != "default":
        load_kwargs["name"] = subset
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    print(f"[stream] 开始 streaming 下载: {dataset_name}" +
          (f" / {subset}" if subset and subset != "default" else ""))
    print(f"[stream] 目标: {max_tokens:,} tokens")

    ds_stream = load_dataset(**load_kwargs)

    all_ids: list[int] = []
    total_tokens = 0
    n_docs = 0
    t0 = time.time()

    text_buffer: list[str] = []

    pbar = tqdm(total=max_tokens, unit="tok", desc="streaming tokenize",
                unit_scale=True, smoothing=0.1)

    for example in ds_stream:
        text = example.get("text") or example.get("content") or example.get("document", "")
        if not text or not text.strip():
            continue

        text_buffer.append(text)

        if len(text_buffer) >= batch_size:
            encoded = tok(text_buffer, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                doc_ids = ids + [eos]
                all_ids.extend(doc_ids)
                total_tokens += len(doc_ids)
                n_docs += 1

            pbar.update(total_tokens - pbar.n)
            text_buffer = []

            if total_tokens >= max_tokens:
                break

    if text_buffer and total_tokens < max_tokens:
        encoded = tok(text_buffer, add_special_tokens=False)
        for ids in encoded["input_ids"]:
            doc_ids = ids + [eos]
            all_ids.extend(doc_ids)
            total_tokens += len(doc_ids)
            n_docs += 1
        pbar.update(total_tokens - pbar.n)

    pbar.close()
    t1 = time.time()

    if len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]

    print(f"[stream] ✓ 完成: {len(all_ids):,} tokens, {n_docs:,} 文档, 耗时 {t1-t0:.1f}s")
    print(f"[stream]   速度: {len(all_ids)/(t1-t0):,.0f} tokens/s, "
          f"{n_docs/(t1-t0):,.0f} docs/s")

    return all_ids


# ============================================================================
# 写入 memmap（从 token id 数组直接写入）
# ============================================================================
def _write_memmap_from_array(token_ids: np.ndarray, dst_path: str, dtype=np.uint16):
    """将 token id 数组写入 numpy memmap 文件。"""
    arr_len = len(token_ids)
    print(f"[write] {os.path.basename(dst_path)}: {arr_len:,} tokens, dtype={dtype.__name__}")

    mmap = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(arr_len,))

    chunk_size = 10_000_000  # 每次写 10M tokens
    for start in tqdm(range(0, arr_len, chunk_size),
                      desc=f"写入 {os.path.basename(dst_path)}",
                      total=(arr_len + chunk_size - 1) // chunk_size):
        end = min(start + chunk_size, arr_len)
        mmap[start:end] = token_ids[start:end]

    mmap.flush()
    file_size_gb = os.path.getsize(dst_path) / (1024**3)
    print(f"[write] ✓ {dst_path} ({file_size_gb:.2f} GB)")


def _write_memmap_from_dataset(tokenized_dataset, dst_path: str, total_batches: int, dtype=np.uint16):
    """将 tokenized HF dataset 写入 numpy memmap 文件（非 streaming 模式用）。"""
    arr_len = int(np.sum(tokenized_dataset["len"], dtype=np.uint64))
    print(f"[write] {os.path.basename(dst_path)}: {arr_len:,} tokens, dtype={dtype.__name__}")

    mmap = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"写入 {os.path.basename(dst_path)}"):
        batch = (
            tokenized_dataset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
        )
        arr_batch = np.concatenate(batch["ids"])
        mmap[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    mmap.flush()
    file_size_gb = os.path.getsize(dst_path) / (1024**3)
    print(f"[write] ✓ {dst_path} ({file_size_gb:.2f} GB)")


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="下载并 tokenize FineWeb/FineWeb-Edu 数据集（LLaMA-2 tokenizer）"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="NousResearch/Llama-2-7b-hf",
        help="HuggingFace tokenizer 名称或路径（默认: NousResearch/Llama-2-7b-hf）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace 数据集名称（默认: HuggingFaceFW/fineweb-edu）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="数据集子集/config 名称。"
             "可选: sample-10BT, sample-100BT, sample-350BT, default(完整版)。"
             "不指定时使用 streaming 模式配合 --max_tokens",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12_000_000_000,
        help="streaming 模式下最多收集的 token 数量（默认: 12B）。"
             "仅在未指定 --subset 时生效",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="streaming 模式下并行下载线程数（默认: 8）。"
             "每个线程独立下载并 tokenize 一个数据分片",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认: data/fineweb-edu-llama）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.005,
        help="验证集比例（默认: 0.005 即 0.5%%）",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="tokenize 并行进程数（非 streaming 模式，默认: 64）",
    )
    parser.add_argument(
        "--total_batches",
        type=int,
        default=1024,
        help="写入 memmap 时的分批数（非 streaming 模式，默认: 1024）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据集 shuffle/split 的随机种子",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成（即使 train.bin/val.bin 已存在）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace datasets 缓存目录（默认: 系统默认）",
    )
    parser.add_argument(
        "--streaming_batch_size",
        type=int,
        default=1000,
        help="streaming 模式下每个线程批量 tokenize 的文档数（默认: 1000）",
    )
    args = parser.parse_args()

    # 判断模式
    use_streaming = args.subset is None

    # 输出目录
    if args.output_dir is None:
        if use_streaming:
            ds_short = args.dataset_name.split("/")[-1]
            args.output_dir = os.path.join("data", f"{ds_short}-llama")
        else:
            subset_tag = args.subset.replace("/", "_")
            args.output_dir = os.path.join("data", f"fineweb-{subset_tag}-llama")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_train = os.path.join(out_dir, "train.bin")
    out_val = os.path.join(out_dir, "val.bin")

    if not args.force and os.path.exists(out_train) and os.path.exists(out_val):
        print(f"[skip] 已存在 {out_train} 和 {out_val}，跳过。使用 --force 强制重新生成。")
        return

    print("=" * 70)
    print("FineWeb 数据集准备（LLaMA-2 Tokenizer）")
    print("=" * 70)
    print(f"  数据集:     {args.dataset_name}")
    if use_streaming:
        print(f"  模式:       多线程 streaming (max_tokens={args.max_tokens:,})")
        print(f"  并行线程:   {args.num_workers}")
    else:
        print(f"  模式:       完整下载 (subset={args.subset})")
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  输出目录:   {out_dir}")
    print(f"  验证集比例: {args.val_ratio}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: 预下载 tokenizer 到本地（避免多线程同时下载）
    # ------------------------------------------------------------------
    print("\n[Step 1] 预下载 tokenizer...")
    local_tok_dir = os.path.join(out_dir, "tokenizer_cache")
    os.makedirs(local_tok_dir, exist_ok=True)

    try:
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True, trust_remote_code=True
        )
        tok.save_pretrained(local_tok_dir)
        tokenizer_path = local_tok_dir
        print(f"  ✓ Tokenizer 已缓存到 {local_tok_dir}")
        print(f"  vocab_size={len(tok)}, eos_token_id={tok.eos_token_id}")
    except Exception as e:
        print(f"  ⚠ 缓存失败 ({e})，使用原始路径: {args.tokenizer}")
        tokenizer_path = args.tokenizer

    # LLaMA-2 vocab_size=32000，uint16 (max 65535) 完全够用
    dtype = np.uint16

    if use_streaming:
        # ==============================================================
        # Streaming 模式：多线程并行下载 + tokenize
        # ==============================================================
        print(f"\n[Step 2] 多线程 Streaming 下载 + tokenize...")

        if args.num_workers > 1:
            all_ids = _parallel_stream_and_tokenize(
                dataset_name=args.dataset_name,
                subset=args.subset,
                max_tokens=args.max_tokens,
                tokenizer_path=tokenizer_path,
                num_workers=args.num_workers,
                seed=args.seed,
                cache_dir=args.cache_dir,
                batch_size=args.streaming_batch_size,
            )
        else:
            all_ids = _stream_and_tokenize(
                dataset_name=args.dataset_name,
                subset=args.subset,
                max_tokens=args.max_tokens,
                tokenizer_path=tokenizer_path,
                seed=args.seed,
                cache_dir=args.cache_dir,
                batch_size=args.streaming_batch_size,
            )

        # 转为 numpy 数组
        print(f"\n[Step 3] 转换为 numpy 数组...")
        all_ids_np = np.array(all_ids, dtype=np.uint32)  # 先用 uint32 避免溢出
        del all_ids  # 释放 list 内存

        # 切分 train / val
        print(f"\n[Step 4] 切分 train/val (val_ratio={args.val_ratio})...")
        total_tokens = len(all_ids_np)
        val_tokens = int(total_tokens * args.val_ratio)
        train_tokens = total_tokens - val_tokens

        train_ids = all_ids_np[:train_tokens].astype(dtype)
        val_ids = all_ids_np[train_tokens:].astype(dtype)
        del all_ids_np

        print(f"  ✓ train: {train_tokens:,} tokens")
        print(f"  ✓ val:   {val_tokens:,} tokens")

        # 写入 memmap
        print(f"\n[Step 5] 写入 memmap 文件...")
        _write_memmap_from_array(train_ids, out_train, dtype=dtype)
        _write_memmap_from_array(val_ids, out_val, dtype=dtype)
        del train_ids, val_ids

        n_train_docs = "N/A (streaming)"
        n_val_docs = "N/A (streaming)"

    else:
        # ==============================================================
        # 非 streaming 模式：下载完整子集后 tokenize
        # ==============================================================
        print(f"\n[Step 2] 下载数据集: {args.dataset_name} / {args.subset}")
        print(f"  并行下载线程数: {args.num_workers}")
        t0 = time.time()

        load_kwargs = {
            "path": args.dataset_name,
            "split": "train",
            "trust_remote_code": True,
            "num_proc": args.num_workers,  # 多线程并行下载多个 parquet 文件
        }
        if args.subset and args.subset != "default":
            load_kwargs["name"] = args.subset
        if args.cache_dir:
            load_kwargs["cache_dir"] = args.cache_dir

        dataset = load_dataset(**load_kwargs)
        t1 = time.time()
        print(f"  ✓ 加载完成: {len(dataset):,} 条文本, 耗时 {t1-t0:.1f}s")

        # 只保留 text 列
        text_col = "text"
        if text_col not in dataset.column_names:
            for candidate in ["content", "document", "raw_content"]:
                if candidate in dataset.column_names:
                    text_col = candidate
                    break
            else:
                print(f"  [ERROR] 找不到文本列！可用列: {dataset.column_names}")
                return

        cols_to_remove = [c for c in dataset.column_names if c != text_col]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)
        if text_col != "text":
            dataset = dataset.rename_column(text_col, "text")

        # 切分 train / val
        print(f"\n[Step 3] 切分 train/val (val_ratio={args.val_ratio})...")
        split = dataset.train_test_split(
            test_size=args.val_ratio, seed=args.seed, shuffle=True
        )
        train_dataset = split["train"]
        val_dataset = split["test"]
        print(f"  ✓ train: {len(train_dataset):,} 条, val: {len(val_dataset):,} 条")

        # Tokenize
        print(f"\n[Step 4] Tokenize (num_proc={args.num_proc})...")
        t2 = time.time()

        train_tokenized = train_dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer_path),
            batched=True,
            remove_columns=["text"],
            desc="tokenize train",
            num_proc=args.num_proc,
        )
        val_tokenized = val_dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer_path),
            batched=True,
            remove_columns=["text"],
            desc="tokenize val",
            num_proc=args.num_proc,
        )

        t3 = time.time()
        train_tokens = int(np.sum(train_tokenized["len"], dtype=np.uint64))
        val_tokens = int(np.sum(val_tokenized["len"], dtype=np.uint64))
        print(f"  ✓ Tokenize 完成, 耗时 {t3-t2:.1f}s")
        print(f"  train: {train_tokens:,} tokens")
        print(f"  val:   {val_tokens:,} tokens")

        # 写入 memmap
        print(f"\n[Step 5] 写入 memmap 文件...")
        _write_memmap_from_dataset(train_tokenized, out_train,
                                   total_batches=args.total_batches, dtype=dtype)
        _write_memmap_from_dataset(val_tokenized, out_val,
                                   total_batches=max(1, args.total_batches // 8), dtype=dtype)

        n_train_docs = len(train_dataset)
        n_val_docs = len(val_dataset)

    # ------------------------------------------------------------------
    # 写入 dtype.txt 和 metadata.json
    # ------------------------------------------------------------------
    dtype_file = os.path.join(out_dir, "dtype.txt")
    with open(dtype_file, "w") as f:
        f.write(dtype.__name__ + "\n")
    print(f"  ✓ dtype.txt → {dtype.__name__}")

    metadata = {
        "dataset": args.dataset_name,
        "subset": args.subset,
        "tokenizer": args.tokenizer,
        "dtype": dtype.__name__,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": n_train_docs,
        "val_docs": n_val_docs,
        "val_ratio": args.val_ratio,
        "max_tokens": args.max_tokens if use_streaming else None,
        "num_workers": args.num_workers if use_streaming else None,
        "streaming": use_streaming,
        "seed": args.seed,
    }
    meta_file = os.path.join(out_dir, "metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ metadata.json 已保存")

    print("\n" + "=" * 70)
    print("✓ 完成!")
    print(f"  train: {out_train} ({train_tokens:,} tokens, "
          f"{os.path.getsize(out_train)/(1024**3):.2f} GB)")
    print(f"  val:   {out_val} ({val_tokens:,} tokens, "
          f"{os.path.getsize(out_val)/(1024**3):.2f} GB)")
    print(f"\n训练时使用:")
    print(f"  --dataset {os.path.basename(out_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
