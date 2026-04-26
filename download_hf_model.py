#!/usr/bin/env python3
"""
HuggingFace 模型下载脚本
将模型下载到共享存储，确保所有节点都能访问缓存
"""

import os
import argparse
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def download_model(model_name, cache_dir, token=None, download_tokenizer=True):
    """
    下载 HuggingFace 模型到指定目录
    
    Args:
        model_name: HuggingFace 模型名称 (例如: facebook/opt-2.7b)
        cache_dir: 缓存目录路径
        token: HuggingFace 访问令牌 (可选，用于私有模型)
        download_tokenizer: 是否下载 tokenizer
    """
    print(f"开始下载模型: {model_name}")
    print(f"缓存目录: {cache_dir}")
    
    # 确保缓存目录存在
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用 snapshot_download 下载模型文件
        # 这会下载所有必要的文件，包括权重、配置等
        print("正在下载模型文件...")
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=token,
            local_dir=cache_dir / model_name.replace("/", "--"),  # 替换斜杠避免路径问题
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"模型已下载到: {model_path}")
        
        # 验证下载的文件
        print("\n验证下载的文件...")
        model_files = list(Path(model_path).glob("*"))
        for f in model_files:
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # 尝试加载配置以验证模型完整性
        print("\n验证模型配置...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"  模型类型: {config.model_type}")
        print(f"  隐藏层大小: {config.hidden_size}")
        print(f"  层数: {config.num_hidden_layers}")
        print(f"  词表大小: {config.vocab_size}")
        
        # 尝试加载 tokenizer
        if download_tokenizer:
            print("\n验证 tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"  Tokenizer 类型: {tokenizer.__class__.__name__}")
                print(f"  词表大小: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"  警告: 无法加载 tokenizer - {e}")
        
        # 尝试加载模型权重（仅验证，不加载到 GPU）
        print("\n验证模型权重...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            print(f"  模型加载成功!")
            # 清理模型释放内存
            del model
        except Exception as e:
            print(f"  警告: 无法加载模型权重 - {e}")
            print(f"  这可能是正常的，如果后续训练时使用正确的 dtype 和 device_map")
        
        print("\n" + "="*80)
        print("✓ 模型下载并验证完成!")
        print("="*80)
        print(f"\n模型路径: {model_path}")
        print(f"\n在训练脚本中，设置以下环境变量以确保使用本地缓存:")
        print(f"  export HF_HOME={cache_dir}")
        print(f"  export TRANSFORMERS_CACHE={cache_dir}")
        print(f"  export HF_DATASETS_CACHE={cache_dir}")
        print(f"\n或者在 train_universal.sh 中添加:")
        print(f"  export HF_HOME={cache_dir}")
        print(f"  export TRANSFORMERS_CACHE={cache_dir}")
        
        return model_path
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}", file=sys.stderr)
        print("\n常见解决方案:")
        print("  1. 如果遇到 429 错误（请求过多），请稍后重试或使用 HF token")
        print("  2. 如果模型需要访问权限，请提供 --token 参数")
        print("  3. 检查网络连接和磁盘空间")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 模型到本地缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下载 OPT-2.7B 模型
  python download_hf_model.py facebook/opt-2.7b

  # 指定缓存目录
  python download_hf_model.py facebook/opt-2.7b --cache-dir /apdcephfs/pig_data/Adaptive-Sparse-Trainer/models

  # 使用 HF token（用于私有模型）
  python download_hf_model.py meta-llama/Llama-2-7b-hf --token YOUR_TOKEN

  # 只下载模型权重，不下载 tokenizer
  python download_hf_model.py facebook/opt-2.7b --no-tokenizer
        """
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace 模型名称 (例如: facebook/opt-2.7b)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models",
        help="模型缓存目录 (默认: /apdcephfs/pig_data/Adaptive-Sparse-Trainer/models)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace 访问令牌 (可选，用于私有模型)"
    )
    
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="不下载 tokenizer"
    )
    
    args = parser.parse_args()
    
    # 从环境变量读取 token（如果未通过命令行提供）
    if args.token is None:
        args.token = os.environ.get("HF_TOKEN", None)
    
    download_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        token=args.token,
        download_tokenizer=not args.no_tokenizer
    )


if __name__ == "__main__":
    main()
