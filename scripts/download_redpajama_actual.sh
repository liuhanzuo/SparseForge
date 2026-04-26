#!/bin/bash
# ============================================================================
# RedPajama 实际数据下载脚本
# ============================================================================
# 
# RedPajama-Data-1T 在 HuggingFace 上只提供 URL 索引，
# 实际数据需要通过 streaming 方式下载或从 URL 下载。
#
# 本脚本支持两种方式：
#   1. streaming: 使用 datasets 库流式下载（推荐）
#   2. urls: 从 URL 文件下载原始数据
#
# 用法:
#   bash download_redpajama_actual.sh [sample|domain] [num_samples]
#
# 示例:
#   bash download_redpajama_actual.sh sample 100000    # 下载 10万条样本
#   bash download_redpajama_actual.sh c4 500000        # 下载 C4 域 50万条
#   bash download_redpajama_actual.sh arxiv 100000     # 下载 ArXiv 域 10万条
#   bash download_redpajama_actual.sh all 1000000      # 每个域下载 100万条
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
DEFAULT_DATA_DIR="${REPO_DIR}/data/redpajama_raw"
DATA_DIR="${RED_PAJAMA_DATA_DIR:-$DEFAULT_DATA_DIR}"

# 代理配置
export http_proxy="${http_proxy:-http://your-proxy:port}"
export https_proxy="${https_proxy:-http://your-proxy:port}"

MODE="${1:-sample}"
NUM_SAMPLES="${2:-100000}"  # 默认下载 10万条

echo "=============================================="
echo "RedPajama 实际数据下载"
echo "=============================================="
echo "模式: $MODE"
echo "样本数: $NUM_SAMPLES"
echo "输出目录: $DATA_DIR"
echo "=============================================="

mkdir -p "$DATA_DIR"

# Python 下载脚本
download_streaming() {
    local subset="${1:-}"
    local num="${2:-100000}"
    local output_dir="$DATA_DIR"
    
    python3 << EOF
import os
import json
from tqdm import tqdm

# 设置代理
os.environ.setdefault("http_proxy", "http://your-proxy:port")
os.environ.setdefault("https_proxy", "http://your-proxy:port")

from datasets import load_dataset

subset = "${subset}" if "${subset}" else None
num_samples = ${num}
output_dir = "${output_dir}"

print(f"[INFO] 下载 RedPajama-Data-1T")
print(f"[INFO] Subset: {subset or 'all'}")
print(f"[INFO] 样本数: {num_samples}")

# 可用的子集
SUBSETS = [
    "arxiv",
    "book", 
    "c4",
    "common_crawl",
    "github",
    "stackexchange",
    "wikipedia"
]

def download_subset(subset_name, max_samples):
    """下载单个子集"""
    print(f"\n>>> 下载子集: {subset_name} (max {max_samples} samples)")
    
    try:
        # 使用 streaming 模式
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T",
            name=subset_name,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        output_file = os.path.join(output_dir, f"{subset_name}.jsonl")
        count = 0
        
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in tqdm(ds, total=max_samples, desc=subset_name):
                if count >= max_samples:
                    break
                # 写入 jsonl
                json.dump({"text": sample.get("text", ""), "meta": sample.get("meta", {})}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
        
        print(f"✓ {subset_name}: 下载 {count} 条到 {output_file}")
        return count
        
    except Exception as e:
        print(f"✗ {subset_name} 下载失败: {e}")
        return 0

# 下载
if subset and subset in SUBSETS:
    # 下载单个子集
    download_subset(subset, num_samples)
elif subset == "all" or not subset:
    # 下载所有子集
    total = 0
    for s in SUBSETS:
        total += download_subset(s, num_samples)
    print(f"\n[DONE] 总计下载 {total} 条样本")
else:
    print(f"未知子集: {subset}")
    print(f"可用子集: {SUBSETS}")

EOF
}

# Sample 版本（更简单）
download_sample() {
    local num="${1:-100000}"
    
    python3 << EOF
import os
import json
from tqdm import tqdm

os.environ.setdefault("http_proxy", "http://your-proxy:port")
os.environ.setdefault("https_proxy", "http://your-proxy:port")

from datasets import load_dataset

num_samples = ${num}
output_dir = "${DATA_DIR}"
output_file = os.path.join(output_dir, "redpajama_sample.jsonl")

print(f"[INFO] 下载 RedPajama-Data-1T-Sample")
print(f"[INFO] 样本数: {num_samples}")

# Sample 版本可以直接 load
ds = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    streaming=True,
    trust_remote_code=True
)

count = 0
with open(output_file, "w", encoding="utf-8") as f:
    for sample in tqdm(ds, total=num_samples, desc="downloading"):
        if count >= num_samples:
            break
        json.dump({"text": sample.get("text", ""), "meta": sample.get("meta", {})}, f, ensure_ascii=False)
        f.write("\n")
        count += 1

print(f"\n✓ 下载完成: {count} 条样本")
print(f"✓ 保存到: {output_file}")

# 显示文件大小
import subprocess
result = subprocess.run(["du", "-h", output_file], capture_output=True, text=True)
print(result.stdout)
EOF
}

# 主逻辑
case "$MODE" in
    sample|Sample)
        download_sample "$NUM_SAMPLES"
        ;;
    arxiv|book|c4|common_crawl|github|stackexchange|wikipedia)
        download_streaming "$MODE" "$NUM_SAMPLES"
        ;;
    all|full)
        download_streaming "all" "$NUM_SAMPLES"
        ;;
    help|--help|-h)
        echo ""
        echo "用法: bash $(basename $0) [mode] [num_samples]"
        echo ""
        echo "模式:"
        echo "  sample      - 下载 Sample 版本 (默认)"
        echo "  all         - 下载所有域"
        echo ""
        echo "单个域:"
        echo "  arxiv       - ArXiv 论文"
        echo "  book        - 书籍"
        echo "  c4          - C4 数据集"
        echo "  common_crawl- CommonCrawl"
        echo "  github      - GitHub 代码"
        echo "  stackexchange - StackExchange"
        echo "  wikipedia   - Wikipedia"
        echo ""
        echo "示例:"
        echo "  bash $(basename $0) sample 100000    # 下载10万条Sample"
        echo "  bash $(basename $0) c4 500000        # 下载50万条C4"
        echo "  bash $(basename $0) all 100000       # 每个域10万条"
        echo ""
        ;;
    *)
        echo "未知模式: $MODE"
        echo "运行 'bash $(basename $0) --help' 查看帮助"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "下载完成!"
echo "数据目录: $DATA_DIR"
ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "(无 jsonl 文件)"
echo "=============================================="
