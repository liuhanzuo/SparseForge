#!/usr/bin/env python3
"""
Prepare mixed domain dataset (C4 + ArXiv + GitHub + StackExchange) for LLaMA training.

This script:
1. Loads C4 data from data/c4_dataset/ (json.gz files 00000-00127)
2. Loads domain-specific data from data/mixed_raw/ (arxiv, github, stackexchange)
3. Tokenizes all data with LLaMA tokenizer
4. **Samples according to specified ratios** (e.g., C4 60%, ArXiv 15%, etc.)
5. Outputs train.bin and val.bin in memmap format

Usage:
    python scripts/prepare_mixed_llama.py
    python scripts/prepare_mixed_llama.py --output_dir data/mixed_llama
    python scripts/prepare_mixed_llama.py --c4_ratio 0.6 --arxiv_ratio 0.15 --github_ratio 0.15 --stackexchange_ratio 0.10
    python scripts/prepare_mixed_llama.py --target_tokens 10B  # 10 billion tokens total
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class DomainConfig:
    """Configuration for a single domain."""
    name: str
    files: List[str]
    text_field: str = "text"
    max_samples: Optional[int] = None
    weight: float = 1.0


# Global tokenizer for multiprocessing
_TOKENIZER = None
_TOKENIZER_PATH = None


def get_tokenizer(tokenizer_path: str):
    """Get or initialize the global tokenizer."""
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


def parse_token_count(s: str) -> int:
    """Parse token count string like '10B', '500M', '1G'."""
    if isinstance(s, int):
        return s
    s = s.strip().upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'G': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def read_jsonl_file(file_path: str, text_field: str = "text", max_samples: Optional[int] = None) -> Generator[str, None, None]:
    """Read texts from a JSONL file."""
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text and len(text.strip()) > 50:  # Filter very short texts
                        yield text
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def read_json_gz_file(file_path: str, text_field: str = "text", max_samples: Optional[int] = None) -> Generator[str, None, None]:
    """Read texts from a gzipped JSON file (C4 format)."""
    count = 0
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text and len(text.strip()) > 50:
                        yield text
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def collect_domain_files(base_dir: str) -> Dict[str, DomainConfig]:
    """Collect all files for each domain."""
    domains = {}
    
    # C4 dataset
    c4_dir = os.path.join(base_dir, "c4_dataset")
    c4_files = sorted([
        os.path.join(c4_dir, f"c4-train.{i:05d}-of-01024.json.gz")
        for i in range(128)
        if os.path.exists(os.path.join(c4_dir, f"c4-train.{i:05d}-of-01024.json.gz"))
    ])
    if c4_files:
        domains["c4"] = DomainConfig(
            name="c4",
            files=c4_files,
            text_field="text"
        )
        print(f"Found {len(c4_files)} C4 files")
    
    # ArXiv
    arxiv_dir = os.path.join(base_dir, "mixed_raw", "arxiv")
    if os.path.isdir(arxiv_dir):
        arxiv_files = sorted([
            os.path.join(arxiv_dir, f)
            for f in os.listdir(arxiv_dir)
            if f.endswith('.jsonl')
        ])
        if arxiv_files:
            domains["arxiv"] = DomainConfig(
                name="arxiv",
                files=arxiv_files,
                text_field="text"
            )
            print(f"Found {len(arxiv_files)} ArXiv files")
    
    # GitHub
    github_dir = os.path.join(base_dir, "mixed_raw", "github")
    if os.path.isdir(github_dir):
        github_files = sorted([
            os.path.join(github_dir, f)
            for f in os.listdir(github_dir)
            if f.endswith('.jsonl')
        ])
        if github_files:
            domains["github"] = DomainConfig(
                name="github",
                files=github_files,
                text_field="text"
            )
            print(f"Found {len(github_files)} GitHub files")
    
    # StackExchange
    se_dir = os.path.join(base_dir, "mixed_raw", "stackexchange")
    if os.path.isdir(se_dir):
        se_files = sorted([
            os.path.join(se_dir, f)
            for f in os.listdir(se_dir)
            if f.endswith('.jsonl')
        ])
        if se_files:
            domains["stackexchange"] = DomainConfig(
                name="stackexchange",
                files=se_files,
                text_field="text"
            )
            print(f"Found {len(se_files)} StackExchange files")
    
    # Wikipedia (if exists)
    wiki_dir = os.path.join(base_dir, "mixed_raw", "wikipedia")
    if os.path.isdir(wiki_dir):
        wiki_files = sorted([
            os.path.join(wiki_dir, f)
            for f in os.listdir(wiki_dir)
            if f.endswith('.jsonl') or f.endswith('.json')
        ])
        if wiki_files:
            domains["wikipedia"] = DomainConfig(
                name="wikipedia",
                files=wiki_files,
                text_field="text"
            )
            print(f"Found {len(wiki_files)} Wikipedia files")
    
    return domains


def tokenize_texts(texts: List[str], tokenizer_path: str) -> Tuple[List[List[int]], List[int]]:
    """Tokenize a batch of texts."""
    tokenizer = get_tokenizer(tokenizer_path)
    encoded = tokenizer(texts, add_special_tokens=False)
    ids_list = encoded["input_ids"]
    eos = int(tokenizer.eos_token_id)
    # Append EOS to each sequence
    ids_list = [ids + [eos] for ids in ids_list]
    lens = [len(ids) for ids in ids_list]
    return ids_list, lens


def process_file_worker(args: Tuple[str, str, str, Optional[int]]) -> Tuple[List[List[int]], List[int], str]:
    """Worker function to process a single file."""
    file_path, text_field, tokenizer_path, max_samples = args
    
    texts = []
    # Determine file type
    if file_path.endswith('.json.gz'):
        for text in read_json_gz_file(file_path, text_field, max_samples):
            texts.append(text)
    else:
        for text in read_jsonl_file(file_path, text_field, max_samples):
            texts.append(text)
    
    if not texts:
        return [], [], file_path
    
    # Tokenize in batches
    all_ids = []
    all_lens = []
    batch_size = 1000
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        ids, lens = tokenize_texts(batch_texts, tokenizer_path)
        all_ids.extend(ids)
        all_lens.extend(lens)
    
    return all_ids, all_lens, file_path


def write_memmap(all_ids: List[List[int]], output_path: str, desc: str = "writing") -> int:
    """Write tokenized data to memmap file."""
    total_tokens = sum(len(ids) for ids in all_ids)
    print(f"\n{desc}: {total_tokens:,} tokens -> {output_path}")
    
    dtype = np.uint16  # LLaMA vocab is 32k, safe for uint16
    mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    
    idx = 0
    for ids in tqdm(all_ids, desc=desc):
        arr = np.array(ids, dtype=dtype)
        mmap[idx:idx + len(arr)] = arr
        idx += len(arr)
    
    mmap.flush()
    return total_tokens


def sample_sequences_by_tokens(
    sequences: List[List[int]], 
    target_tokens: int, 
    allow_repeat: bool = False,
    seed: int = 42
) -> Tuple[List[List[int]], int, bool]:
    """
    Sample sequences to reach approximately target_tokens.
    
    Returns:
        - sampled sequences
        - actual token count
        - whether data was insufficient (had to repeat or truncate)
    """
    random.seed(seed)
    
    total_available = sum(len(s) for s in sequences)
    
    if total_available == 0:
        return [], 0, True
    
    # 如果目标 token 数 <= 可用数据，直接采样（不重复）
    if target_tokens <= total_available:
        random.shuffle(sequences)
        sampled = []
        current_tokens = 0
        for seq in sequences:
            if current_tokens >= target_tokens:
                break
            sampled.append(seq)
            current_tokens += len(seq)
        return sampled, current_tokens, False
    
    # 如果目标 token 数 > 可用数据
    if allow_repeat:
        # 重复采样直到达到目标
        sampled = []
        current_tokens = 0
        while current_tokens < target_tokens:
            random.shuffle(sequences)
            for seq in sequences:
                if current_tokens >= target_tokens:
                    break
                sampled.append(seq)
                current_tokens += len(seq)
        return sampled, current_tokens, True
    else:
        # 使用所有可用数据
        random.shuffle(sequences)
        actual_tokens = sum(len(s) for s in sequences)
        return sequences, actual_tokens, True


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed domain dataset for LLaMA training")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/NousResearch--Llama-2-7b-hf",
        help="Path to LLaMA tokenizer"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mixed_llama",
        help="Output directory for train.bin and val.bin"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.001,
        help="Ratio of data to use for validation (default: 0.1%%)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for processing"
    )
    parser.add_argument(
        "--c4_ratio",
        type=float,
        default=0.60,
        help="Ratio of C4 data in the mix (default: 60%%)"
    )
    parser.add_argument(
        "--arxiv_ratio",
        type=float,
        default=0.15,
        help="Ratio of ArXiv data (default: 15%%)"
    )
    parser.add_argument(
        "--github_ratio",
        type=float,
        default=0.15,
        help="Ratio of GitHub data (default: 15%%)"
    )
    parser.add_argument(
        "--stackexchange_ratio",
        type=float,
        default=0.10,
        help="Ratio of StackExchange data (default: 10%%)"
    )
    parser.add_argument(
        "--target_tokens",
        type=str,
        default=None,
        help="Target total tokens (e.g., '10B', '500M'). If not set, use all available data."
    )
    parser.add_argument(
        "--allow_repeat",
        action="store_true",
        help="Allow repeating data if a domain has insufficient samples to meet ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
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
    print("Mixed Domain Dataset Preparation for LLaMA (with Ratio Sampling)")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Workers: {args.num_workers}")
    print()
    
    # Pre-load tokenizer
    print("[Step 1/6] Loading tokenizer...")
    tokenizer = get_tokenizer(tokenizer_path)
    print()
    
    # Collect domain files
    print("[Step 2/6] Collecting domain files...")
    domains = collect_domain_files(data_dir)
    
    if not domains:
        print("ERROR: No data files found!")
        return
    
    print()
    
    # Set domain ratios (only for available domains)
    requested_ratios = {
        "c4": args.c4_ratio,
        "arxiv": args.arxiv_ratio,
        "github": args.github_ratio,
        "stackexchange": args.stackexchange_ratio,
        "wikipedia": 0.0
    }
    
    # Normalize ratios for available domains only
    available_domains = set(domains.keys())
    total_ratio = sum(requested_ratios.get(d, 0) for d in available_domains)
    
    domain_ratios = {}
    for d in available_domains:
        if total_ratio > 0:
            domain_ratios[d] = requested_ratios.get(d, 0) / total_ratio
        else:
            domain_ratios[d] = 1.0 / len(available_domains)
    
    print("Target domain mixing ratios (normalized for available data):")
    for domain_name in sorted(domains.keys()):
        print(f"  {domain_name}: {domain_ratios.get(domain_name, 0):.1%}")
    print()
    
    # Process all files and collect tokenized data per domain
    print("[Step 3/6] Processing and tokenizing all files...")
    
    domain_data = {}  # domain_name -> list of token id sequences
    
    for domain_name, config in domains.items():
        print(f"\n--- Processing {domain_name.upper()} ({len(config.files)} files) ---")
        
        # Prepare worker arguments
        worker_args = [
            (f, config.text_field, tokenizer_path, None)
            for f in config.files
        ]
        
        domain_ids = []
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_file_worker, arg): arg[0] for arg in worker_args}
            
            with tqdm(total=len(futures), desc=f"Processing {domain_name}") as pbar:
                for future in as_completed(futures):
                    try:
                        ids_list, lens, file_path = future.result()
                        domain_ids.extend(ids_list)
                        pbar.set_postfix_str(f"seqs: {len(domain_ids):,}")
                    except Exception as e:
                        print(f"Error processing file: {e}")
                    pbar.update(1)
        
        domain_data[domain_name] = domain_ids
        domain_tokens = sum(len(ids) for ids in domain_ids)
        print(f"  {domain_name}: {len(domain_ids):,} sequences, {domain_tokens:,} tokens")
    
    # Calculate total available tokens
    total_available_tokens = sum(
        sum(len(ids) for ids in seqs) 
        for seqs in domain_data.values()
    )
    
    print(f"\n[Info] Total available tokens across all domains: {total_available_tokens:,}")
    
    # Determine target tokens
    if args.target_tokens:
        target_tokens = parse_token_count(args.target_tokens)
        print(f"[Info] Target tokens specified: {target_tokens:,}")
    else:
        # 如果没有指定目标，使用所有数据
        target_tokens = total_available_tokens
        print(f"[Info] No target specified, using all available data: {target_tokens:,} tokens")
    
    # Calculate per-domain target tokens based on ratios
    print("\n[Step 4/6] Sampling data according to ratios...")
    
    domain_targets = {}
    for domain_name in domain_data:
        domain_targets[domain_name] = int(target_tokens * domain_ratios.get(domain_name, 0))
    
    print("\nPer-domain token targets:")
    for domain_name in sorted(domain_targets.keys()):
        available = sum(len(ids) for ids in domain_data[domain_name])
        target = domain_targets[domain_name]
        status = "✓" if available >= target else "⚠ insufficient"
        print(f"  {domain_name:15s}: target {target:>12,} tokens, available {available:>12,} tokens {status}")
    
    # Sample from each domain
    all_train_ids = []
    all_val_ids = []
    domain_stats = {}
    actual_ratios = {}
    
    for domain_name, sequences in domain_data.items():
        target = domain_targets[domain_name]
        
        if not sequences:
            print(f"\n⚠ {domain_name}: No data available, skipping")
            continue
        
        # Sample according to target
        sampled, actual_tokens, was_insufficient = sample_sequences_by_tokens(
            sequences, 
            target, 
            allow_repeat=args.allow_repeat,
            seed=args.seed + hash(domain_name) % 10000
        )
        
        if was_insufficient:
            if args.allow_repeat:
                print(f"\n⚠ {domain_name}: Data insufficient, repeated to reach {actual_tokens:,} tokens")
            else:
                print(f"\n⚠ {domain_name}: Data insufficient ({actual_tokens:,} < {target:,}), using all available")
        
        # Split into train/val
        random.shuffle(sampled)
        n_val = max(1, int(len(sampled) * args.val_ratio))
        val_ids = sampled[:n_val]
        train_ids = sampled[n_val:]
        
        all_train_ids.extend(train_ids)
        all_val_ids.extend(val_ids)
        
        train_tokens = sum(len(ids) for ids in train_ids)
        val_tokens = sum(len(ids) for ids in val_ids)
        
        domain_stats[domain_name] = {
            "train_seqs": len(train_ids),
            "train_tokens": train_tokens,
            "val_seqs": len(val_ids),
            "val_tokens": val_tokens,
            "target_tokens": target,
            "was_insufficient": was_insufficient
        }
    
    # Calculate actual ratios
    total_train_tokens = sum(stats['train_tokens'] for stats in domain_stats.values())
    for domain_name, stats in domain_stats.items():
        actual_ratios[domain_name] = stats['train_tokens'] / total_train_tokens if total_train_tokens > 0 else 0
    
    # Final shuffle
    print("\n[Step 5/6] Shuffling combined data...")
    random.shuffle(all_train_ids)
    random.shuffle(all_val_ids)
    
    # Write output files
    print("\n[Step 6/6] Writing output files...")
    train_tokens = write_memmap(all_train_ids, train_path, "Writing train.bin")
    val_tokens = write_memmap(all_val_ids, val_path, "Writing val.bin")
    
    # Save metadata
    metadata = {
        "tokenizer": tokenizer_path,
        "vocab_size": len(tokenizer),
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_sequences": len(all_train_ids),
        "val_sequences": len(all_val_ids),
        "target_tokens": target_tokens,
        "requested_ratios": {k: v for k, v in requested_ratios.items() if k in domains},
        "actual_ratios": actual_ratios,
        "domain_stats": domain_stats,
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save dtype for compatibility
    dtype_path = os.path.join(output_dir, "dtype.txt")
    with open(dtype_path, 'w') as f:
        f.write("uint16")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total train tokens: {train_tokens:,}")
    print(f"Total val tokens:   {val_tokens:,}")
    print(f"Train sequences:    {len(all_train_ids):,}")
    print(f"Val sequences:      {len(all_val_ids):,}")
    print()
    print("Domain breakdown (Target vs Actual):")
    print(f"  {'Domain':<15s} {'Target %':>10s} {'Actual %':>10s} {'Tokens':>15s} {'Status':<12s}")
    print("  " + "-" * 62)
    for domain_name in sorted(domain_stats.keys()):
        stats = domain_stats[domain_name]
        target_pct = domain_ratios.get(domain_name, 0) * 100
        actual_pct = actual_ratios.get(domain_name, 0) * 100
        status = "⚠ LOW" if stats['was_insufficient'] else "✓ OK"
        print(f"  {domain_name:<15s} {target_pct:>9.1f}% {actual_pct:>9.1f}% {stats['train_tokens']:>15,} {status:<12s}")
    print()
    print(f"Output files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {metadata_path}")
    print()
    
    # Warnings
    insufficient_domains = [d for d, s in domain_stats.items() if s['was_insufficient']]
    if insufficient_domains:
        print("⚠ WARNING: The following domains had insufficient data:")
        for d in insufficient_domains:
            stats = domain_stats[d]
            print(f"    - {d}: got {stats['train_tokens']:,} tokens (target was {stats['target_tokens']:,})")
        if not args.allow_repeat:
            print("  Consider using --allow_repeat to repeat data, or adjust ratios.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
