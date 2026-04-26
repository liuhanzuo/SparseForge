#!/usr/bin/env python3
"""
Warmup model files into OS page cache for faster loading on shared filesystems.

Usage:
    python scripts/warmup_model_cache.py models/Qwen--Qwen3-1.7B

This script reads all .safetensors files in the model directory into memory,
which populates the OS page cache. Subsequent model loads (from any process
on the same node) will be much faster as they read from RAM instead of disk.

For multi-node training, run this on each node before starting training,
or run it once on rank 0 and use a barrier to ensure cache is warm.
"""
import os
import sys
import time
from pathlib import Path


def warmup_model_files(model_dir: str, verbose: bool = True) -> float:
    """Read all model files into OS page cache.
    
    Args:
        model_dir: Path to model directory containing .safetensors files
        verbose: Print progress information
        
    Returns:
        Total bytes read
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"ERROR: Model directory does not exist: {model_dir}")
        sys.exit(1)
    
    # Find all model weight files
    patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    weight_files = []
    for pattern in patterns:
        weight_files.extend(model_path.glob(pattern))
    
    if not weight_files:
        print(f"WARNING: No weight files found in {model_dir}")
        return 0
    
    total_bytes = 0
    start_time = time.time()
    
    for i, fpath in enumerate(sorted(weight_files)):
        file_size = fpath.stat().st_size
        file_size_gb = file_size / (1024**3)
        
        if verbose:
            print(f"[{i+1}/{len(weight_files)}] Warming up: {fpath.name} ({file_size_gb:.2f} GB)...", end=" ", flush=True)
        
        file_start = time.time()
        
        # Read file in chunks to populate page cache
        chunk_size = 64 * 1024 * 1024  # 64MB chunks
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
        
        file_time = time.time() - file_start
        if verbose:
            speed = file_size / (1024**3) / file_time if file_time > 0 else 0
            print(f"done in {file_time:.1f}s ({speed:.2f} GB/s)")
        
        total_bytes += file_size
    
    total_time = time.time() - start_time
    total_gb = total_bytes / (1024**3)
    
    if verbose:
        print(f"\n✓ Warmup complete: {total_gb:.2f} GB in {total_time:.1f}s ({total_gb/total_time:.2f} GB/s)")
    
    return total_bytes


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/warmup_model_cache.py <model_dir> [model_dir2 ...]")
        print("Example: python scripts/warmup_model_cache.py models/Qwen--Qwen3-1.7B")
        sys.exit(1)
    
    for model_dir in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Warming up model cache: {model_dir}")
        print(f"{'='*60}")
        warmup_model_files(model_dir)


if __name__ == "__main__":
    main()
