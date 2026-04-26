#!/bin/bash
# ============================================================================
# RedPajama 多线程下载 & 数据准备 - Shell 包装脚本
# ============================================================================
#
# 用法:
#   bash run_prepare_redpajama.sh [mode] [options]
#
# 模式:
#   quality   - 下载高质量域 (arxiv, wikipedia, book, stackexchange, github)
#   full      - 完整流程 (下载 + tokenize)
#   download  - 只下载
#   tokenize  - 只 tokenize
#
# 示例:
#   bash run_prepare_redpajama.sh quality            # 高质量域，50万/域
#   bash run_prepare_redpajama.sh quality 1000000    # 高质量域，100万/域
#   bash run_prepare_redpajama.sh download           # 只下载
#   bash run_prepare_redpajama.sh tokenize           # 只 tokenize
#
# 后台运行:
#   nohup bash run_prepare_redpajama.sh quality > prepare.log 2>&1 &
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# 代理设置
export http_proxy="${http_proxy:-http://your-proxy:port}"
export https_proxy="${https_proxy:-http://your-proxy:port}"

# 默认配置
MODE="${1:-quality}"
SAMPLES_PER_DOMAIN="${2:-500000}"
TOKENIZER="${3:-Qwen/Qwen3-1.7B}"

# 目录配置
RAW_DIR="${PROJECT_ROOT}/data/redpajama_raw"
OUTPUT_DIR="${PROJECT_ROOT}/data/redpajama_qwen"

# 线程/进程数配置
DOWNLOAD_WORKERS=4          # 下载线程数 (4个域并行)
TOKENIZE_WORKERS=32         # tokenize 进程数

echo "=============================================="
echo "🚀 RedPajama 多线程下载 & 数据准备"
echo "=============================================="
echo "模式: $MODE"
echo "每域样本数: $SAMPLES_PER_DOMAIN"
echo "Tokenizer: $TOKENIZER"
echo "原始数据目录: $RAW_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "下载线程数: $DOWNLOAD_WORKERS"
echo "Tokenize 进程数: $TOKENIZE_WORKERS"
echo "=============================================="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# 检查依赖
python3 -c "import transformers; import datasets; import tqdm" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install transformers datasets tqdm -q
}

case "$MODE" in
    quality|full)
        # 完整流程: 下载高质量域 + tokenize
        python3 "${SCRIPT_DIR}/prepare_redpajama_multithread.py" \
            --domains quality \
            --samples_per_domain "$SAMPLES_PER_DOMAIN" \
            --tokenizer "$TOKENIZER" \
            --download_workers "$DOWNLOAD_WORKERS" \
            --tokenize_workers "$TOKENIZE_WORKERS" \
            --raw_dir "$RAW_DIR" \
            --output_dir "$OUTPUT_DIR"
        ;;
        
    download)
        # 只下载
        python3 "${SCRIPT_DIR}/prepare_redpajama_multithread.py" \
            --domains quality \
            --samples_per_domain "$SAMPLES_PER_DOMAIN" \
            --download_workers "$DOWNLOAD_WORKERS" \
            --raw_dir "$RAW_DIR" \
            --download_only
        ;;
        
    tokenize)
        # 只 tokenize
        python3 "${SCRIPT_DIR}/prepare_redpajama_multithread.py" \
            --tokenizer "$TOKENIZER" \
            --tokenize_workers "$TOKENIZE_WORKERS" \
            --raw_dir "$RAW_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --tokenize_only
        ;;
        
    custom)
        # 自定义域 (需要额外参数)
        DOMAINS="${4:-arxiv,wikipedia}"
        python3 "${SCRIPT_DIR}/prepare_redpajama_multithread.py" \
            --domains "$DOMAINS" \
            --samples_per_domain "$SAMPLES_PER_DOMAIN" \
            --tokenizer "$TOKENIZER" \
            --download_workers "$DOWNLOAD_WORKERS" \
            --tokenize_workers "$TOKENIZE_WORKERS" \
            --raw_dir "$RAW_DIR" \
            --output_dir "$OUTPUT_DIR"
        ;;
        
    help|--help|-h)
        echo ""
        echo "用法: bash $(basename $0) [mode] [samples_per_domain] [tokenizer]"
        echo ""
        echo "模式:"
        echo "  quality   - 下载高质量域 + tokenize (默认)"
        echo "  full      - 同 quality"
        echo "  download  - 只下载"
        echo "  tokenize  - 只 tokenize 已下载的数据"
        echo "  custom    - 自定义域 (第4个参数指定域)"
        echo ""
        echo "高质量域包括:"
        echo "  arxiv, wikipedia, stackexchange, github"
        echo "  (跳过 common_crawl 和 c4, book 因版权问题已下架)"
        echo ""
        echo "示例:"
        echo "  bash $(basename $0) quality 500000           # 50万/域"
        echo "  bash $(basename $0) quality 1000000          # 100万/域"  
        echo "  bash $(basename $0) download 300000          # 只下载 30万/域"
        echo "  bash $(basename $0) tokenize                 # 只 tokenize"
        echo "  bash $(basename $0) custom 500000 Qwen/Qwen3-1.7B arxiv,book"
        echo ""
        echo "后台运行:"
        echo "  nohup bash $(basename $0) quality > prepare.log 2>&1 &"
        echo "  tail -f prepare.log"
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
echo "✅ 完成!"
echo "=============================================="
