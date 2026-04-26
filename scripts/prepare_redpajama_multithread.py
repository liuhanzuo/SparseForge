#!/usr/bin/env python3
"""
RedPajama 多线程下载 & 数据准备一体化脚本

支持功能:
  1. 多线程并行下载多个域 (跳过 common_crawl)
  2. 多进程并行 tokenize
  3. 断点续传 (已下载的域自动跳过)
  4. 实时进度显示
  5. 输出 train.bin / val.bin 用于训练

使用方法:
  # 下载高质量域并 tokenize (推荐, 跳过 common_crawl)
  python prepare_redpajama_multithread.py --domains quality --samples_per_domain 500000
  
  # 下载指定域
  python prepare_redpajama_multithread.py --domains arxiv,wikipedia,book --samples_per_domain 300000
  
  # 只下载不处理
  python prepare_redpajama_multithread.py --domains quality --download_only
  
  # 只处理已下载的数据
  python prepare_redpajama_multithread.py --tokenize_only
  
  # 完整流程 (下载 + tokenize)
  python prepare_redpajama_multithread.py --domains quality --samples_per_domain 500000 --tokenizer Qwen/Qwen3-1.7B

域说明:
  quality = arxiv, wikipedia, stackexchange, github (高质量, 跳过 common_crawl 和 c4)
  all = 所有可用域 (包括 common_crawl, 非常大)
  或指定: arxiv,wikipedia,stackexchange,github,c4,common_crawl
  注意: book 域因版权问题已下架，不可用
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count, Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# 代理配置
os.environ.setdefault("http_proxy", "http://your-proxy:port")
os.environ.setdefault("https_proxy", "http://your-proxy:port")


# ============================================================================
# 配置常量
# ============================================================================

# RedPajama 域定义 (注意: book 因版权问题已下架)
ALL_DOMAINS = ["arxiv", "c4", "common_crawl", "github", "stackexchange", "wikipedia"]

# 高质量域 (跳过 common_crawl 和 c4, 因为你已有 c4_qwen)
# book 域因版权问题已被 HuggingFace 下架
QUALITY_DOMAINS = ["arxiv", "wikipedia", "stackexchange", "github"]

# 各域大小估算 (用于进度显示)
DOMAIN_SIZE_ESTIMATE = {
    "arxiv": 2_000_000,       # ~2M samples
    "c4": 350_000_000,        # ~350M samples (很大)
    "common_crawl": 900_000_000,  # 很大，不推荐
    "github": 7_000_000,      # ~7M samples
    "stackexchange": 30_000_000,  # ~30M samples
    "wikipedia": 29_000_000,  # ~29M samples
    # book 域已因版权问题下架
}


@dataclass
class DownloadConfig:
    """下载配置"""
    domains: List[str]
    samples_per_domain: int
    output_dir: Path
    num_download_workers: int = 4
    retry_times: int = 3
    
    
@dataclass
class TokenizeConfig:
    """Tokenize 配置"""
    input_dir: Path
    output_dir: Path
    tokenizer_name: str
    num_workers: int = 32
    batch_size: int = 1000
    val_ratio: float = 0.01


# ============================================================================
# 下载模块
# ============================================================================

def download_single_domain(
    domain: str,
    max_samples: int,
    output_dir: Path,
    progress_dict: Optional[Dict] = None,
    retry_times: int = 3,
) -> Tuple[str, int, str]:
    """
    下载单个域的数据
    
    Args:
        domain: 域名称
        max_samples: 最大样本数
        output_dir: 输出目录
        progress_dict: 进度共享字典 (用于多线程)
        retry_times: 重试次数
        
    Returns:
        (domain, downloaded_count, status)
    """
    from datasets import load_dataset
    
    output_file = output_dir / f"{domain}.jsonl"
    temp_file = output_dir / f"{domain}.jsonl.tmp"
    
    # 检查是否已完成
    if output_file.exists():
        # 统计已有行数
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        if existing_count >= max_samples:
            if progress_dict is not None:
                progress_dict[domain] = existing_count
            return (domain, existing_count, "skipped (already complete)")
    
    # 检查是否有部分下载
    start_count = 0
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            start_count = sum(1 for _ in f)
        print(f"[{domain}] Resuming from {start_count}")
    
    for attempt in range(retry_times):
        try:
            ds = load_dataset(
                "togethercomputer/RedPajama-Data-1T",
                name=domain,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            
            count = start_count
            mode = 'a' if start_count > 0 else 'w'
            
            # 如果需要跳过已下载的
            ds_iter = iter(ds)
            if start_count > 0:
                for _ in range(start_count):
                    try:
                        next(ds_iter)
                    except StopIteration:
                        break
            
            with open(temp_file, mode, encoding='utf-8') as f:
                for sample in ds_iter:
                    if count >= max_samples:
                        break
                    
                    text = sample.get("text", "")
                    if not text:
                        continue
                    
                    record = {"text": text, "meta": sample.get("meta", {})}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    
                    # 更新进度
                    if progress_dict is not None and count % 1000 == 0:
                        progress_dict[domain] = count
            
            # 下载完成，重命名
            temp_file.rename(output_file)
            
            if progress_dict is not None:
                progress_dict[domain] = count
                
            return (domain, count, "success")
            
        except Exception as e:
            print(f"[{domain}] Attempt {attempt + 1}/{retry_times} failed: {e}")
            if attempt < retry_times - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return (domain, start_count, f"failed: {e}")
    
    return (domain, 0, "failed")


def download_all_domains(config: DownloadConfig) -> Dict[str, int]:
    """
    多线程并行下载所有域
    
    Args:
        config: 下载配置
        
    Returns:
        每个域下载的样本数
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 开始多线程下载 RedPajama 数据")
    print("=" * 60)
    print(f"域: {config.domains}")
    print(f"每个域样本数: {config.samples_per_domain:,}")
    print(f"下载线程数: {config.num_download_workers}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 60 + "\n")
    
    # 使用 Manager 创建共享进度字典
    manager = Manager()
    progress_dict = manager.dict()
    
    for domain in config.domains:
        progress_dict[domain] = 0
    
    results = {}
    
    # 创建进度显示线程
    def show_progress():
        """显示实时进度"""
        while True:
            total = sum(progress_dict.values())
            status = " | ".join([f"{d}: {progress_dict[d]:,}" for d in config.domains])
            print(f"\r[进度] Total: {total:,} | {status}", end="", flush=True)
            time.sleep(2)
    
    import threading
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()
    
    # 多线程下载
    with ThreadPoolExecutor(max_workers=config.num_download_workers) as executor:
        futures = {
            executor.submit(
                download_single_domain,
                domain,
                config.samples_per_domain,
                config.output_dir,
                progress_dict,
                config.retry_times,
            ): domain
            for domain in config.domains
        }
        
        for future in as_completed(futures):
            domain = futures[future]
            try:
                domain_name, count, status = future.result()
                results[domain_name] = count
                print(f"\n✓ [{domain_name}] {status} ({count:,} samples)")
            except Exception as e:
                print(f"\n✗ [{domain}] Exception: {e}")
                results[domain] = 0
    
    print("\n")
    return results


# ============================================================================
# Tokenize 模块
# ============================================================================

def load_tokenizer(tokenizer_name: str):
    """加载 tokenizer"""
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def tokenize_chunk(args) -> List[int]:
    """
    Tokenize 一批文本 (用于多进程)
    
    Args:
        args: (texts, tokenizer_name, eos_token_id)
        
    Returns:
        token ids list
    """
    texts, tokenizer_name, eos_token_id = args
    
    from transformers import AutoTokenizer
    
    # 每个进程加载自己的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    all_ids = []
    encoded = tokenizer(texts, add_special_tokens=False)
    
    for ids in encoded["input_ids"]:
        ids_with_eos = ids + [eos_token_id]
        all_ids.extend(ids_with_eos)
    
    return all_ids


def load_all_jsonl_files(input_dir: Path) -> List[str]:
    """
    加载所有 jsonl 文件中的文本
    """
    texts = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")
    
    print(f"Found {len(jsonl_files)} jsonl files:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    for jsonl_file in tqdm(jsonl_files, desc="Loading files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
    
    return texts


def tokenize_parallel(
    texts: List[str],
    tokenizer_name: str,
    num_workers: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.dtype]:
    """
    多进程并行 tokenize
    
    Args:
        texts: 文本列表
        tokenizer_name: tokenizer 名称
        num_workers: 进程数
        batch_size: 每批大小
        
    Returns:
        (token_ids_array, dtype)
    """
    print(f"\n开始多进程 tokenize (workers={num_workers}, batch_size={batch_size})")
    
    # 先加载一次 tokenizer 获取信息
    tokenizer = load_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    
    print(f"Tokenizer vocab_size: {vocab_size}")
    print(f"EOS token id: {eos_token_id}")
    
    # 确定 dtype
    if vocab_size > 65536:
        dtype = np.uint32
        print("⚠ vocab_size > 65536, using uint32")
    else:
        dtype = np.uint16
        print("Using uint16")
    
    # 分批
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i + batch_size]
        chunks.append((chunk_texts, tokenizer_name, eos_token_id))
    
    print(f"Total chunks: {len(chunks)}")
    
    # 多进程处理
    all_ids = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = list(tqdm(
            executor.map(tokenize_chunk, chunks),
            total=len(chunks),
            desc="Tokenizing",
        ))
        
        for chunk_ids in futures:
            all_ids.extend(chunk_ids)
    
    print(f"Total tokens: {len(all_ids):,}")
    return np.array(all_ids, dtype=dtype), dtype


def write_memmap(data: np.ndarray, output_path: Path, dtype) -> None:
    """写入 memmap 文件"""
    print(f"Writing {len(data):,} tokens to {output_path}...")
    
    mmap = np.memmap(str(output_path), dtype=dtype, mode="w+", shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    
    file_size = output_path.stat().st_size
    print(f"✓ Written {file_size / (1024**3):.2f} GB")


def split_train_val(data: np.ndarray, val_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """分割训练集和验证集"""
    val_size = int(len(data) * val_ratio)
    val_size = max(val_size, 10000)
    
    val_data = data[-val_size:]
    train_data = data[:-val_size]
    
    print(f"Train size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")
    
    return train_data, val_data


def tokenize_all(config: TokenizeConfig) -> None:
    """
    执行完整的 tokenize 流程
    """
    print("\n" + "=" * 60)
    print("🔄 开始 Tokenize 流程")
    print("=" * 60)
    print(f"输入目录: {config.input_dir}")
    print(f"输出目录: {config.output_dir}")
    print(f"Tokenizer: {config.tokenizer_name}")
    print(f"进程数: {config.num_workers}")
    print("=" * 60 + "\n")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    train_bin = config.output_dir / "train.bin"
    val_bin = config.output_dir / "val.bin"
    
    if train_bin.exists() and val_bin.exists():
        print(f"⚠ Output files already exist:")
        print(f"  {train_bin}")
        print(f"  {val_bin}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Skipped.")
            return
    
    # 加载数据
    texts = load_all_jsonl_files(config.input_dir)
    print(f"Loaded {len(texts):,} text samples")
    
    if not texts:
        raise ValueError("No text samples found!")
    
    # Tokenize
    all_ids, dtype = tokenize_parallel(
        texts,
        config.tokenizer_name,
        config.num_workers,
        config.batch_size,
    )
    
    # 写入 dtype
    dtype_str = "uint32" if dtype == np.uint32 else "uint16"
    (config.output_dir / "dtype.txt").write_text(dtype_str)
    
    # 分割
    train_data, val_data = split_train_val(all_ids, config.val_ratio)
    
    # 写入
    write_memmap(train_data, train_bin, dtype)
    write_memmap(val_data, val_bin, dtype)
    
    print("\n" + "=" * 60)
    print("✅ Tokenize 完成!")
    print(f"  train.bin: {train_bin}")
    print(f"  val.bin: {val_bin}")
    print(f"  dtype: {dtype_str}")
    print("=" * 60 + "\n")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RedPajama 多线程下载 & 数据准备一体化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 下载参数
    parser.add_argument(
        "--domains",
        type=str,
        default="quality",
        help="要下载的域: quality, all, 或逗号分隔的域名 (default: quality)",
    )
    parser.add_argument(
        "--samples_per_domain",
        type=int,
        default=500000,
        help="每个域下载的样本数 (default: 500000)",
    )
    parser.add_argument(
        "--download_workers",
        type=int,
        default=4,
        help="下载线程数 (default: 4)",
    )
    
    # Tokenize 参数
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace tokenizer 名称 (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--tokenize_workers",
        type=int,
        default=None,
        help="Tokenize 进程数 (default: CPU 核心数 / 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Tokenize 批大小 (default: 1000)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="验证集比例 (default: 0.01)",
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (default: data/redpajama_qwen)",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="原始数据目录 (default: data/redpajama_raw)",
    )
    
    # 流程控制
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="只下载，不 tokenize",
    )
    parser.add_argument(
        "--tokenize_only",
        action="store_true",
        help="只 tokenize 已下载的数据",
    )
    
    args = parser.parse_args()
    
    # 确定目录
    script_dir = Path(__file__).parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else script_dir / "data" / "redpajama_raw"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "data" / "redpajama_qwen"
    
    # 解析域
    if args.domains == "quality":
        domains = QUALITY_DOMAINS.copy()
    elif args.domains == "all":
        domains = ALL_DOMAINS.copy()
    else:
        domains = [d.strip() for d in args.domains.split(",")]
        # 验证域名
        for d in domains:
            if d not in ALL_DOMAINS:
                print(f"⚠ 未知域: {d}")
                print(f"可用域: {ALL_DOMAINS}")
                sys.exit(1)
    
    # tokenize workers
    tokenize_workers = args.tokenize_workers or max(1, cpu_count() // 2)
    
    print("\n" + "=" * 70)
    print("🎯 RedPajama 多线程下载 & 数据准备")
    print("=" * 70)
    print(f"域: {domains}")
    print(f"每个域样本数: {args.samples_per_domain:,}")
    print(f"原始数据目录: {raw_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"下载线程数: {args.download_workers}")
    print(f"Tokenize 进程数: {tokenize_workers}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    # Step 1: 下载
    if not args.tokenize_only:
        download_config = DownloadConfig(
            domains=domains,
            samples_per_domain=args.samples_per_domain,
            output_dir=raw_dir,
            num_download_workers=args.download_workers,
        )
        download_results = download_all_domains(download_config)
        
        print("\n📊 下载统计:")
        total_samples = 0
        for domain, count in download_results.items():
            print(f"  {domain}: {count:,}")
            total_samples += count
        print(f"  Total: {total_samples:,}")
    
    # Step 2: Tokenize
    if not args.download_only:
        tokenize_config = TokenizeConfig(
            input_dir=raw_dir,
            output_dir=output_dir,
            tokenizer_name=args.tokenizer,
            num_workers=tokenize_workers,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
        )
        tokenize_all(tokenize_config)
    
    elapsed = time.time() - start_time
    print(f"\n⏱ 总耗时: {elapsed / 60:.1f} 分钟")
    print("✅ 全部完成!")


if __name__ == "__main__":
    main()
