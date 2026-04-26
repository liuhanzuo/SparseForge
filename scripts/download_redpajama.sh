#!/bin/bash
# ============================================================================
# RedPajama 数据集下载脚本
# ============================================================================
# 用法:
#   bash download_redpajama.sh [sample|full|domain]
#
# 示例:
#   bash download_redpajama.sh sample      # 下载 Sample 版本 (~1GB)
#   bash download_redpajama.sh full        # 下载完整版 (~1.2TB)
#   bash download_redpajama.sh arxiv       # 只下载 arxiv 域
#   bash download_redpajama.sh c4          # 只下载 c4 域
#
# 完整版包含以下域 (总计 ~1.2TB):
#   - common_crawl: ~878GB
#   - c4: ~175GB  
#   - github: ~59GB
#   - wikipedia: ~24GB
#   - books: ~26GB
#   - arxiv: ~28GB
#   - stackexchange: ~20GB
#
# 环境变量:
#   RED_PAJAMA_DATA_DIR: 下载目录 (默认: ./data/redpajama_raw)
#   HF_TOKEN: HuggingFace token (如果需要认证)
# ============================================================================

set -euo pipefail

# 配置
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
DEFAULT_DATA_DIR="${REPO_DIR}/data/redpajama_raw"
DATA_DIR="${RED_PAJAMA_DATA_DIR:-$DEFAULT_DATA_DIR}"

# 代理配置 (如果需要)
export http_proxy="${http_proxy:-http://your-proxy:port}"
export https_proxy="${https_proxy:-http://your-proxy:port}"

MODE="${1:-sample}"

echo "=============================================="
echo "RedPajama 数据集下载"
echo "=============================================="
echo "模式: $MODE"
echo "下载目录: $DATA_DIR"
echo "=============================================="

mkdir -p "$DATA_DIR"

# 检查 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface_hub..."
    pip install -U huggingface_hub
fi

# 登录检查 (RedPajama 可能需要认证)
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "使用提供的 HF_TOKEN 登录..."
    huggingface-cli login --token "$HF_TOKEN"
elif [[ ! -f ~/.huggingface/token ]] && [[ ! -f ~/.cache/huggingface/token ]]; then
    echo "⚠ 警告: 未检测到 HuggingFace token"
    echo "如果下载失败，请运行: huggingface-cli login"
    echo "或设置环境变量: export HF_TOKEN=your_token"
fi

download_sample() {
    echo ""
    echo ">>> 下载 RedPajama-Data-1T-Sample (~1GB)..."
    echo ""
    
    # 方法1: 使用 huggingface-cli (推荐)
    huggingface-cli download togethercomputer/RedPajama-Data-1T-Sample \
        --repo-type dataset \
        --local-dir "$DATA_DIR/sample" \
        --local-dir-use-symlinks False
    
    echo ""
    echo "✓ Sample 下载完成: $DATA_DIR/sample"
}

download_full() {
    echo ""
    echo ">>> 下载 RedPajama-Data-1T (~1.2TB)..."
    echo ">>> 这可能需要很长时间..."
    echo ""
    
    huggingface-cli download togethercomputer/RedPajama-Data-1T \
        --repo-type dataset \
        --local-dir "$DATA_DIR/full" \
        --local-dir-use-symlinks False
    
    echo ""
    echo "✓ 完整数据集下载完成: $DATA_DIR/full"
}

download_domain() {
    local domain="$1"
    echo ""
    echo ">>> 下载域: $domain..."
    echo ""
    
    # 域名到路径的映射
    case "$domain" in
        arxiv)
            pattern="arxiv*"
            ;;
        book|books)
            pattern="book*"
            ;;
        c4)
            pattern="c4*"
            ;;
        cc|common_crawl|commoncrawl)
            pattern="common_crawl*"
            ;;
        github)
            pattern="github*"
            ;;
        stackexchange|stack)
            pattern="stackexchange*"
            ;;
        wiki|wikipedia)
            pattern="wikipedia*"
            ;;
        *)
            echo "未知域: $domain"
            echo "可用的域: arxiv, book, c4, common_crawl, github, stackexchange, wikipedia"
            exit 1
            ;;
    esac
    
    huggingface-cli download togethercomputer/RedPajama-Data-1T \
        --repo-type dataset \
        --include "$pattern" \
        --local-dir "$DATA_DIR/$domain" \
        --local-dir-use-symlinks False
    
    echo ""
    echo "✓ 域 $domain 下载完成: $DATA_DIR/$domain"
}

# Python 下载方法 (备选)
download_with_python() {
    local dataset_name="${1:-togethercomputer/RedPajama-Data-1T-Sample}"
    
    echo "使用 Python datasets 库下载..."
    
    python3 << EOF
import os
os.environ["RED_PAJAMA_DATA_DIR"] = "$DATA_DIR"

from datasets import load_dataset

print(f"下载 {dataset_name}...")
ds = load_dataset("$dataset_name", trust_remote_code=True)

print(f"数据集信息:")
print(ds)

# 保存到磁盘
ds.save_to_disk("$DATA_DIR/hf_cache")
print(f"✓ 保存到: $DATA_DIR/hf_cache")
EOF
}

# 主逻辑
case "$MODE" in
    sample|Sample|SAMPLE)
        download_sample
        ;;
    full|Full|FULL|all)
        download_full
        ;;
    python|py)
        download_with_python "togethercomputer/RedPajama-Data-1T-Sample"
        ;;
    python-full)
        download_with_python "togethercomputer/RedPajama-Data-1T"
        ;;
    arxiv|book|books|c4|cc|common_crawl|commoncrawl|github|stackexchange|stack|wiki|wikipedia)
        download_domain "$MODE"
        ;;
    help|--help|-h)
        echo ""
        echo "用法: bash $(basename $0) [sample|full|<domain>]"
        echo ""
        echo "模式:"
        echo "  sample      - 下载 Sample 版本 (~1GB)"
        echo "  full        - 下载完整版 (~1.2TB)"
        echo "  python      - 使用 Python datasets 下载 Sample"
        echo "  python-full - 使用 Python datasets 下载完整版"
        echo ""
        echo "单独下载某个域:"
        echo "  arxiv       - ArXiv 论文 (~28GB)"
        echo "  book        - 书籍 (~26GB)"
        echo "  c4          - C4 数据集 (~175GB)"
        echo "  common_crawl- CommonCrawl (~878GB)"
        echo "  github      - GitHub 代码 (~59GB)"
        echo "  stackexchange - StackExchange (~20GB)"
        echo "  wikipedia   - Wikipedia (~24GB)"
        echo ""
        echo "环境变量:"
        echo "  RED_PAJAMA_DATA_DIR - 下载目录"
        echo "  HF_TOKEN - HuggingFace token"
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
echo "=============================================="
