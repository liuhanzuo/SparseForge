#!/usr/bin/env python3
"""
Download Qwen3-1.7B model to local directory.
Usage: python scripts/download_qwen_model.py
"""
from huggingface_hub import snapshot_download
import os

def main():
    model_name = "Qwen/Qwen3-1.7B"
    # 保存到 models/Qwen--Qwen3-1.7B，和 facebook--opt-2.7b 格式一致
    save_dir = "/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models/Qwen--Qwen3-1.7B"
    
    print(f"Downloading {model_name} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 清理可能存在的锁文件
    cache_dir = os.path.join(save_dir, ".cache", "huggingface", "download")
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".lock"):
                lock_path = os.path.join(cache_dir, f)
                print(f"Removing stale lock file: {lock_path}")
                os.remove(lock_path)
    
    # 使用 snapshot_download 下载完整模型
    local_dir = snapshot_download(
        repo_id=model_name,
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
        resume_download=True,           # 支持断点续传
    )
    
    print(f"\n✓ Model downloaded to: {local_dir}")
    print("\nFiles saved:")
    for f in sorted(os.listdir(save_dir)):
        if f.startswith("."):  # 跳过隐藏文件/目录
            continue
        fpath = os.path.join(save_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024*1024*1024)
            if size > 0.001:
                print(f"  {f}: {size:.2f} GB")
            else:
                print(f"  {f}")
    
    # 验证关键文件存在
    required_files = [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = []
    for rf in required_files:
        if not os.path.exists(os.path.join(save_dir, rf)):
            missing.append(rf)
    
    if missing:
        print(f"\n⚠️ WARNING: Missing required files: {missing}")
        return 1
    else:
        print(f"\n✓ All required files present!")
        print(f"\nYou can now use the model with:")
        print(f'  STUDENT_MODEL="models/Qwen--Qwen3-1.7B"')
        print(f'  TEACHER_MODEL="models/Qwen--Qwen3-1.7B"')
        return 0

if __name__ == "__main__":
    exit(main())
