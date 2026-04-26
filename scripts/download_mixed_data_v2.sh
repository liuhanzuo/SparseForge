#!/bin/bash
#############################################################################
# 混合域数据下载 & 准备脚本 V2
# 
# 使用直接 URL 下载，比 HuggingFace streaming 更稳定
#
# 推荐配比 (补充已有的C4数据):
#   - C4:           已有 (~22GB)  - 用户已下载1/8 C4
#   - ArXiv:        40% (~28GB)   - 学术/推理能力 (全部101个文件)
#   - GitHub:       30% (~20GB)   - 代码理解 (35个文件)
#   - StackExchange: 10% (~5GB)   - QA问答能力 (单文件)
#   - Wikipedia:    20% (~24GB)   - 事实性知识 (单文件，可选)
#
# 使用方法:
#   bash scripts/download_mixed_data_v2.sh          # 默认模式：不含C4 (~55GB)
#   bash scripts/download_mixed_data_v2.sh with-c4  # 包含C4 (~100GB)
#   bash scripts/download_mixed_data_v2.sh small    # 小规模测试 (~15GB)
#   bash scripts/download_mixed_data_v2.sh large    # 大规模含Wiki (~80GB)
#   bash scripts/download_mixed_data_v2.sh tokenize # 只tokenize已下载数据
#############################################################################

set -e

# ============== 配置区 ==============

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
URL_DIR="${PROJECT_DIR}/urls"

MODE="${1:-default}"
TOKENIZER="Qwen/Qwen3-1.7B"
DOWNLOAD_WORKERS=8
TOKENIZE_WORKERS=$(nproc)

# 数据目录
RAW_DIR="${PROJECT_DIR}/data/mixed_raw"
OUTPUT_DIR="${PROJECT_DIR}/data/mixed_qwen"

# ============== 配比配置 ==============
# C4: 1024个文件，每个约170MB
# ArXiv: 101个文件，每个约280MB
# GitHub: 99个文件，每个约600MB
# Wikipedia: 1个文件，约24GB
# StackExchange: 1个文件，约20GB

case "$MODE" in
    "small"|"test")
        echo "📦 模式: 小规模测试 (~15GB, 不含C4)"
        C4_FILES=0            # 跳过，用户已有
        ARXIV_FILES=30        # ~8GB
        GITHUB_FILES=10       # ~6GB
        WIKI_ENABLED=false    # 跳过 (太大)
        SE_ENABLED=false      # 跳过
        ;;
    "medium")
        echo "📦 模式: 中等规模 (~35GB, 不含C4)"
        C4_FILES=0            # 跳过，用户已有
        ARXIV_FILES=60        # ~17GB
        GITHUB_FILES=25       # ~15GB
        WIKI_ENABLED=false    
        SE_ENABLED=true       # ~5GB (单文件)
        ;;
    "large")
        echo "📦 模式: 大规模 (~80GB, 不含C4, 含Wiki)"
        C4_FILES=0            # 跳过，用户已有
        ARXIV_FILES=101       # ~28GB (全部)
        GITHUB_FILES=50       # ~30GB
        WIKI_ENABLED=true     # ~24GB
        SE_ENABLED=true       # ~5GB
        ;;
    "with-c4")
        echo "📦 模式: 包含C4 (~100GB)"
        C4_FILES=250          # ~42GB
        ARXIV_FILES=70        # ~20GB
        GITHUB_FILES=25       # ~15GB
        WIKI_ENABLED=false    
        SE_ENABLED=true       # ~5GB
        ;;
    "default")
        echo "📦 模式: 默认配比 (~55GB, 不含C4 - 用户已有1/8 C4)"
        C4_FILES=0            # 跳过，用户已有约22GB C4
        ARXIV_FILES=101       # ~28GB (全部101个文件)
        GITHUB_FILES=35       # ~21GB
        WIKI_ENABLED=false    # 跳过(太大单文件24GB)
        SE_ENABLED=true       # ~5GB
        ;;
    "tokenize")
        echo "📦 模式: 只tokenize已下载数据"
        TOKENIZE_ONLY=1
        ;;
    *)
        echo "❌ 未知模式: $MODE"
echo "可用模式: default, small, medium, large, with-c4, tokenize"
        exit 1
        ;;
esac

# ============== 显示配置 ==============

echo ""
echo "=============================================="
echo "🚀 混合域数据下载 & 准备 (V2 - URL直接下载)"
echo "=============================================="
echo "Tokenizer: $TOKENIZER"
echo "原始数据目录: $RAW_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "下载线程数: $DOWNLOAD_WORKERS"
echo "Tokenize 进程数: $TOKENIZE_WORKERS"

if [ -z "$TOKENIZE_ONLY" ]; then
    echo ""
    echo "📊 域配比:"
    echo "  C4:           ${C4_FILES:-0} 文件 (~$((${C4_FILES:-0} * 170 / 1024))GB)"
    echo "  ArXiv:        ${ARXIV_FILES:-0} 文件 (~$((${ARXIV_FILES:-0} * 280 / 1024))GB)"
    echo "  GitHub:       ${GITHUB_FILES:-0} 文件 (~$((${GITHUB_FILES:-0} * 600 / 1024))GB)"
    echo "  Wikipedia:    ${WIKI_ENABLED:-false}"
    echo "  StackExchange: ${SE_ENABLED:-false}"
fi
echo "=============================================="
echo ""

# ============== 创建目录 ==============

mkdir -p "$RAW_DIR"/{c4,arxiv,github,wikipedia,stackexchange}
mkdir -p "$OUTPUT_DIR"

# ============== 下载函数 ==============

download_domain_by_urls() {
    local domain=$1
    local url_file="$URL_DIR/${domain}.txt"
    local output_dir="$RAW_DIR/$domain"
    local max_files=$2
    
    if [ ! -f "$url_file" ]; then
        echo "⚠ URL文件不存在: $url_file"
        return 1
    fi
    
    # 获取URL列表
    local urls=($(head -n "$max_files" "$url_file"))
    local total=${#urls[@]}
    
    if [ "$total" -eq 0 ]; then
        echo "⚠ 没有URL: $domain"
        return 0
    fi
    
    echo ""
    echo "============================================================"
    echo "📥 下载 $domain ($total 个文件)"
    echo "============================================================"
    
    # 统计已下载
    local downloaded=0
    for url in "${urls[@]}"; do
        local filename=$(basename "$url")
        if [ -f "$output_dir/$filename" ]; then
            ((downloaded++)) || true
        fi
    done
    
    if [ "$downloaded" -eq "$total" ]; then
        echo "✓ $domain 已全部下载 ($total 个文件)"
        return 0
    fi
    
    echo "已下载: $downloaded / $total"
    echo ""
    
    # 创建临时URL文件（只包含未下载的）
    local temp_url_file=$(mktemp)
    for url in "${urls[@]}"; do
        local filename=$(basename "$url")
        if [ ! -f "$output_dir/$filename" ]; then
            echo "$url" >> "$temp_url_file"
        fi
    done
    
    # 使用 xargs + wget 并行下载
    echo "开始下载 (并行度: $DOWNLOAD_WORKERS)..."
    cat "$temp_url_file" | xargs -P "$DOWNLOAD_WORKERS" -I {} bash -c '
        url="{}"
        filename=$(basename "$url")
        output_dir="'"$output_dir"'"
        output_file="$output_dir/$filename"
        
        # 重试3次
        for i in 1 2 3; do
            if wget -q --show-progress -O "$output_file.tmp" "$url" 2>/dev/null; then
                mv "$output_file.tmp" "$output_file"
                echo "✓ $filename"
                exit 0
            fi
            echo "⚠ $filename 重试 $i/3..."
            sleep 2
        done
        echo "✗ $filename 失败"
        rm -f "$output_file.tmp"
    '
    
    rm -f "$temp_url_file"
    
    # 统计结果
    local final_count=$(ls -1 "$output_dir"/*.jsonl 2>/dev/null | wc -l)
    local total_size=$(du -sh "$output_dir" 2>/dev/null | cut -f1)
    echo ""
    echo "✓ $domain 完成: $final_count 文件, $total_size"
}

download_single_file() {
    local domain=$1
    local url_file="$URL_DIR/${domain}.txt"
    local output_dir="$RAW_DIR/$domain"
    
    if [ ! -f "$url_file" ]; then
        echo "⚠ URL文件不存在: $url_file"
        return 1
    fi
    
    local url=$(head -n 1 "$url_file")
    local filename=$(basename "$url")
    local output_file="$output_dir/$filename"
    
    if [ -f "$output_file" ]; then
        local size=$(du -h "$output_file" | cut -f1)
        echo "✓ $domain 已存在 ($size)"
        return 0
    fi
    
    echo ""
    echo "============================================================"
    echo "📥 下载 $domain (单文件)"
    echo "URL: $url"
    echo "============================================================"
    
    wget --show-progress -O "$output_file.tmp" "$url"
    mv "$output_file.tmp" "$output_file"
    
    local size=$(du -h "$output_file" | cut -f1)
    echo "✓ $domain 完成 ($size)"
}

# ============== 执行下载 ==============

if [ -z "$TOKENIZE_ONLY" ]; then
    echo ""
    echo "🔄 开始下载..."
    
    # 下载 C4
    if [ "${C4_FILES:-0}" -gt 0 ]; then
        download_domain_by_urls "c4" "$C4_FILES"
    fi
    
    # 下载 ArXiv
    if [ "${ARXIV_FILES:-0}" -gt 0 ]; then
        download_domain_by_urls "arxiv" "$ARXIV_FILES"
    fi
    
    # 下载 GitHub
    if [ "${GITHUB_FILES:-0}" -gt 0 ]; then
        download_domain_by_urls "github" "$GITHUB_FILES"
    fi
    
    # 下载 Wikipedia (单文件)
    if [ "${WIKI_ENABLED:-false}" = "true" ]; then
        download_single_file "wikipedia"
    fi
    
    # 下载 StackExchange (单文件)
    if [ "${SE_ENABLED:-false}" = "true" ]; then
        download_single_file "stackexchange"
    fi
    
    echo ""
    echo "=============================================="
    echo "✅ 下载完成!"
    echo ""
    echo "📊 数据统计:"
    for domain in c4 arxiv github wikipedia stackexchange; do
        if [ -d "$RAW_DIR/$domain" ]; then
            count=$(ls -1 "$RAW_DIR/$domain"/*.jsonl 2>/dev/null | wc -l)
            if [ "$count" -gt 0 ]; then
                size=$(du -sh "$RAW_DIR/$domain" 2>/dev/null | cut -f1)
                echo "  $domain: $count 文件, $size"
            fi
        fi
    done
    total_size=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1)
    echo "  --------------------------------"
    echo "  总计: $total_size"
    echo "=============================================="
fi

# ============== Tokenize ==============

echo ""
read -p "是否继续执行 Tokenize? [Y/n]: " response
response=${response:-Y}
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "跳过 Tokenize. 稍后可运行:"
    echo "  bash scripts/download_mixed_data_v2.sh tokenize"
    exit 0
fi

echo ""
echo "🔄 开始 Tokenize..."
echo ""

export RAW_DIR OUTPUT_DIR TOKENIZER TOKENIZE_WORKERS

python3 << 'PYTHON_SCRIPT'
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from transformers import AutoTokenizer

# 配置
RAW_DIR = Path(os.environ.get('RAW_DIR', 'data/mixed_raw'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', 'data/mixed_qwen'))
TOKENIZER_NAME = os.environ.get('TOKENIZER', 'Qwen/Qwen3-1.7B')
NUM_WORKERS = int(os.environ.get('TOKENIZE_WORKERS', '32'))
BATCH_SIZE = 1000
VAL_RATIO = 0.005  # 0.5% 验证集

print(f"输入目录: {RAW_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"Tokenizer: {TOKENIZER_NAME}")
print(f"进程数: {NUM_WORKERS}")
print()

# 检查输出
train_bin = OUTPUT_DIR / "train.bin"
val_bin = OUTPUT_DIR / "val.bin"

if train_bin.exists():
    print(f"⚠ 输出文件已存在: {train_bin}")
    response = input("覆盖? [y/N]: ")
    if response.lower() != 'y':
        print("跳过.")
        sys.exit(0)

# 加载 tokenizer
print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

eos_token_id = tokenizer.eos_token_id
vocab_size = len(tokenizer)

print(f"Vocab size: {vocab_size}")
print(f"EOS token id: {eos_token_id}")

# 确定 dtype
dtype = np.uint32 if vocab_size > 65536 else np.uint16
print(f"使用 dtype: {dtype}")

# 查找所有 jsonl 文件
print()
print("扫描数据文件...")
jsonl_files = []
for domain in ['c4', 'arxiv', 'github', 'wikipedia', 'stackexchange']:
    domain_dir = RAW_DIR / domain
    if domain_dir.exists():
        files = sorted(domain_dir.glob("*.jsonl"))
        if files:
            print(f"  {domain}: {len(files)} 文件")
            jsonl_files.extend(files)

if not jsonl_files:
    print(f"❌ 未找到数据文件")
    sys.exit(1)

print(f"总计: {len(jsonl_files)} 文件")

# Tokenize 函数
def tokenize_file(args):
    filepath, tokenizer_name, eos_id = args
    from transformers import AutoTokenizer
    
    tok = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    all_ids = []
    count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if text and len(text) > 50:  # 过滤太短的文本
                        texts.append(text)
                        count += 1
                        
                        # 批量处理
                        if len(texts) >= 500:
                            encoded = tok(texts, add_special_tokens=False)
                            for ids in encoded["input_ids"]:
                                all_ids.extend(ids + [eos_id])
                            texts = []
                except:
                    continue
            
            # 处理剩余
            if texts:
                encoded = tok(texts, add_special_tokens=False)
                for ids in encoded["input_ids"]:
                    all_ids.extend(ids + [eos_id])
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []
    
    return all_ids

# 多进程 tokenize
print()
print(f"开始 Tokenize ({NUM_WORKERS} workers)...")

args_list = [(f, TOKENIZER_NAME, eos_token_id) for f in jsonl_files]

all_ids = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = list(tqdm(
        executor.map(tokenize_file, args_list),
        total=len(args_list),
        desc="Tokenizing",
    ))
    
    for chunk_ids in results:
        all_ids.extend(chunk_ids)

print(f"总 tokens: {len(all_ids):,}")

if len(all_ids) == 0:
    print("❌ 没有数据!")
    sys.exit(1)

# 转换为数组
data = np.array(all_ids, dtype=dtype)

# 分割
val_size = max(int(len(data) * VAL_RATIO), 10000)
val_data = data[-val_size:]
train_data = data[:-val_size]

print(f"训练集: {len(train_data):,} tokens ({len(train_data) * (4 if dtype == np.uint32 else 2) / 1e9:.2f} GB)")
print(f"验证集: {len(val_data):,} tokens")

# 写入
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print(f"写入 {train_bin}...")
train_mmap = np.memmap(str(train_bin), dtype=dtype, mode="w+", shape=(len(train_data),))
train_mmap[:] = train_data
train_mmap.flush()
del train_mmap

print(f"写入 {val_bin}...")
val_mmap = np.memmap(str(val_bin), dtype=dtype, mode="w+", shape=(len(val_data),))
val_mmap[:] = val_data
val_mmap.flush()
del val_mmap

# dtype 文件
dtype_str = "uint32" if dtype == np.uint32 else "uint16"
(OUTPUT_DIR / "dtype.txt").write_text(dtype_str)

print()
print("=" * 60)
print("✅ 完成!")
print(f"  train.bin: {train_bin} ({train_bin.stat().st_size / 1e9:.2f} GB)")
print(f"  val.bin: {val_bin}")
print(f"  dtype: {dtype_str}")
print(f"  总 tokens: {len(all_ids):,}")
print("=" * 60)
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "🎉 全部完成!"
echo ""
echo "输出文件:"
ls -lh "$OUTPUT_DIR"/*.bin 2>/dev/null || echo "  (无)"
echo ""
echo "下一步训练命令:"
echo "  torchrun --nproc_per_node=8 train_sparse.py \\"
echo "    --data_dir $OUTPUT_DIR \\"
echo "    --model_path models/Qwen3-1.7B \\"
echo "    --output_dir outputs/sparse_qwen3"
echo "=============================================="
