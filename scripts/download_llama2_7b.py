#!/usr/bin/env python3
"""
下载 LLaMA2-7B 模型到本地目录
使用 HuggingFace 镜像站点
"""

import os
import sys

# 使用 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 移除代理（镜像站不需要）
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(key, None)

from huggingface_hub import snapshot_download

MODEL_ID = "NousResearch/Llama-2-7b-hf"
LOCAL_DIR = "/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models/NousResearch--Llama-2-7b-hf"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
print(f"Using mirror: {os.environ.get('HF_ENDPOINT', 'default')}")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Model downloaded to {LOCAL_DIR}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("\n尝试备选方案...")
    
    # 备选：使用 modelscope
    try:
        from modelscope import snapshot_download as ms_download
        ms_download('AI-ModelScope/Llama-2-7b-hf', cache_dir=LOCAL_DIR)
        print(f"✓ Model downloaded via ModelScope to {LOCAL_DIR}")
    except Exception as e2:
        print(f"✗ ModelScope also failed: {e2}")
        sys.exit(1)
