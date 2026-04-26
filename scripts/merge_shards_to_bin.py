#!/usr/bin/env python3
"""
将 shard_XXXX.npy 合并为 train.bin + val.bin (uint32 memmap)。

用法:
    python scripts/merge_shards_to_bin.py \
        --shard_dir data/dolmino-mix-1124-llama3 \
        --output_dir data/dolmino-mix-1124-llama3 \
        --val_ratio 0.001

shard 文件格式: numpy memmap, dtype=uint32, 由 download_and_tokenize_dolmino.py 生成。
输出: train.bin, val.bin, dtype.txt, metadata.json
"""

import argparse
import glob
import json
import os
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="合并 shard npy 文件为 train.bin + val.bin")
    parser.add_argument("--shard_dir", type=str, required=True, help="shard_XXXX.npy 所在目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录 (默认同 shard_dir)")
    parser.add_argument("--val_ratio", type=float, default=0.001, help="验证集比例 (默认 0.1%%)")
    parser.add_argument("--shard_pattern", type=str, default="shard_*.npy", help="shard 文件匹配模式")
    parser.add_argument("--force", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    output_dir = args.output_dir or args.shard_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    if not args.force and os.path.exists(train_path):
        print(f"❌ {train_path} 已存在，使用 --force 覆盖")
        return

    # 1. 发现所有 shard 文件
    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, args.shard_pattern)))
    if not shard_files:
        print(f"❌ 在 {args.shard_dir} 中未找到匹配 {args.shard_pattern} 的文件")
        return

    print(f"找到 {len(shard_files)} 个 shard 文件")

    # 2. 扫描每个 shard 的大小 (raw uint32 binary, 无 numpy header)
    t0 = time.time()
    shard_sizes = []
    for f in shard_files:
        file_bytes = os.path.getsize(f)
        n_tokens = file_bytes // 4  # uint32 = 4 bytes
        if file_bytes % 4 != 0:
            print(f"  ⚠️ 文件大小不是 4 的倍数，跳过: {f} ({file_bytes} bytes)")
            continue
        shard_sizes.append((f, n_tokens))

    total_tokens = sum(s for _, s in shard_sizes)
    print(f"总 token 数: {total_tokens:,} ({total_tokens * 4 / 1024**3:.2f} GB)")

    # 3. 计算 train/val 分割点
    val_tokens = max(1000, int(total_tokens * args.val_ratio))
    train_tokens = total_tokens - val_tokens
    print(f"Train: {train_tokens:,} tokens ({train_tokens * 4 / 1024**3:.2f} GB)")
    print(f"Val:   {val_tokens:,} tokens ({val_tokens * 4 / 1024**3:.2f} GB)")

    # 4. 创建 train.bin 和 val.bin
    print(f"\n写入 {train_path} ...")
    train_mm = np.memmap(train_path, dtype=np.uint32, mode='w+', shape=(train_tokens,))
    print(f"写入 {val_path} ...")
    val_mm = np.memmap(val_path, dtype=np.uint32, mode='w+', shape=(val_tokens,))

    # 5. 顺序读取 shard，写入 train 和 val
    global_offset = 0  # 在整个数据集中的偏移
    train_offset = 0
    val_offset = 0

    for i, (fpath, n_tokens) in enumerate(shard_sizes):
        arr = np.memmap(fpath, dtype=np.uint32, mode='r', shape=(n_tokens,))

        # 这个 shard 中有多少属于 train，多少属于 val
        shard_start = global_offset
        shard_end = global_offset + n_tokens

        # train 部分: [shard_start, min(shard_end, train_tokens))
        if shard_start < train_tokens:
            train_chunk_end = min(shard_end, train_tokens)
            chunk_size = train_chunk_end - shard_start
            train_mm[train_offset: train_offset + chunk_size] = arr[:chunk_size]
            train_offset += chunk_size

        # val 部分: [max(shard_start, train_tokens), shard_end)
        if shard_end > train_tokens:
            val_chunk_start = max(0, train_tokens - shard_start)
            chunk_size = n_tokens - val_chunk_start
            val_mm[val_offset: val_offset + chunk_size] = arr[val_chunk_start:]
            val_offset += chunk_size

        global_offset = shard_end
        del arr

        if (i + 1) % 100 == 0 or (i + 1) == len(shard_sizes):
            pct = global_offset / total_tokens * 100
            elapsed = time.time() - t0
            speed = global_offset / max(elapsed, 0.01)
            print(
                f"  [{i+1}/{len(shard_sizes)}] {pct:.1f}%, "
                f"{global_offset:,}/{total_tokens:,} tokens, "
                f"{speed:,.0f} tok/s, {elapsed:.1f}s"
            )

    train_mm.flush()
    val_mm.flush()
    del train_mm, val_mm

    t1 = time.time()

    # 6. 写 dtype.txt
    dtype_path = os.path.join(output_dir, "dtype.txt")
    with open(dtype_path, "w") as f:
        f.write("uint32")

    # 7. 写 metadata.json
    metadata = {
        "description": "Merged from shard npy files",
        "dtype": "uint32",
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "num_shards": len(shard_files),
        "val_ratio": args.val_ratio,
        "elapsed_seconds": t1 - t0,
    }

    # 尝试读取已有的 metadata.json 补充信息
    existing_meta = os.path.join(args.shard_dir, "metadata.json")
    if os.path.exists(existing_meta):
        try:
            with open(existing_meta) as f:
                old_meta = json.load(f)
            for key in ["tokenizer", "vocab_size", "bos_token_id", "eos_token_id", "max_seq_len", "dataset"]:
                if key in old_meta:
                    metadata[key] = old_meta[key]
        except Exception:
            pass

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ 合并完成!")
    print(f"  Train: {train_path} ({train_tokens:,} tokens, {train_tokens * 4 / 1024**3:.2f} GB)")
    print(f"  Val:   {val_path} ({val_tokens:,} tokens, {val_tokens * 4 / 1024**3:.2f} GB)")
    print(f"  dtype: {dtype_path}")
    print(f"  meta:  {meta_path}")
    print(f"  耗时:  {t1 - t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
