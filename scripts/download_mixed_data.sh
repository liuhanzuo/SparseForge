#!/bin/bash
#############################################################################
# 混合域数据下载 & 准备脚本
# 
# 推荐配比 (约100GB):
#   - C4:           50% (~50GB, ~13B tokens)   - 通用语言能力基础
#   - ArXiv:        20% (~20GB, ~5B tokens)    - 学术/推理能力
#   - Wikipedia:    15% (~15GB, ~4B tokens)    - 事实性知识
#   - GitHub:       10% (~10GB, ~2.5B tokens)  - 代码理解
#   - StackExchange: 5% (~5GB, ~1.3B tokens)   - QA问答能力
#
# 总计: ~100GB, ~26B tokens
#
# 使用方法:
#   bash scripts/download_mixed_data.sh          # 使用默认配比
#   bash scripts/download_mixed_data.sh small    # 小规模测试 (~20GB)
#   bash scripts/download_mixed_data.sh large    # 大规模 (~150GB)
#   bash scripts/download_mixed_data.sh tokenize # 只tokenize已下载数据
#############################################################################

set -e

# ============== 配置区 ==============

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认配置
MODE="${1:-default}"
TOKENIZER="Qwen/Qwen3-1.7B"
DOWNLOAD_WORKERS=4
TOKENIZE_WORKERS=$(nproc)

# 数据目录
RAW_DIR="${PROJECT_DIR}/data/mixed_raw"
OUTPUT_DIR="${PROJECT_DIR}/data/mixed_qwen"

# ============== 配比配置 ==============
# 格式: 域名:样本数
# 根据测试: 每100万样本约 2-4GB (取决于域)

case "$MODE" in
    "small"|"test")
        # 小规模测试 (~20GB)
        echo "📦 模式: 小规模测试 (~20GB)"
        C4_SAMPLES=2500000        # ~10GB
        ARXIV_SAMPLES=500000      # ~4GB  
        WIKIPEDIA_SAMPLES=400000  # ~3GB
        GITHUB_SAMPLES=300000     # ~2GB
        STACKEXCHANGE_SAMPLES=200000  # ~1GB
        ;;
    "large")
        # 大规模 (~150GB)
        echo "📦 模式: 大规模 (~150GB)"
        C4_SAMPLES=12500000       # ~75GB
        ARXIV_SAMPLES=2500000     # ~30GB
        WIKIPEDIA_SAMPLES=1500000 # ~22GB
        GITHUB_SAMPLES=1000000    # ~15GB
        STACKEXCHANGE_SAMPLES=500000  # ~8GB
        ;;
    "default"|"medium")
        # 默认配比 (~100GB)
        echo "📦 模式: 默认配比 (~100GB)"
        C4_SAMPLES=8000000        # ~50GB (50%)
        ARXIV_SAMPLES=1500000     # ~20GB (20%)
        WIKIPEDIA_SAMPLES=1000000 # ~15GB (15%)
        GITHUB_SAMPLES=600000     # ~10GB (10%)
        STACKEXCHANGE_SAMPLES=300000  # ~5GB (5%)
        ;;
    "tokenize")
        # 只tokenize
        echo "📦 模式: 只tokenize已下载数据"
        TOKENIZE_ONLY=1
        ;;
    "no-c4")
        # 不包含C4 (~50GB)
        echo "📦 模式: 不包含C4 (~50GB)"
        C4_SAMPLES=0
        ARXIV_SAMPLES=2000000     # ~25GB
        WIKIPEDIA_SAMPLES=1200000 # ~18GB
        GITHUB_SAMPLES=500000     # ~8GB
        STACKEXCHANGE_SAMPLES=300000  # ~5GB
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "可用模式: default, small, large, no-c4, tokenize"
        exit 1
        ;;
esac

# ============== 显示配置 ==============

echo ""
echo "=============================================="
echo "🚀 混合域数据下载 & 准备"
echo "=============================================="
echo "Tokenizer: $TOKENIZER"
echo "原始数据目录: $RAW_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "下载线程数: $DOWNLOAD_WORKERS"
echo "Tokenize 进程数: $TOKENIZE_WORKERS"
echo ""

if [ -z "$TOKENIZE_ONLY" ]; then
    echo "📊 域配比:"
    echo "  C4:           ${C4_SAMPLES:-0} 样本"
    echo "  ArXiv:        ${ARXIV_SAMPLES:-0} 样本"
    echo "  Wikipedia:    ${WIKIPEDIA_SAMPLES:-0} 样本"
    echo "  GitHub:       ${GITHUB_SAMPLES:-0} 样本"
    echo "  StackExchange: ${STACKEXCHANGE_SAMPLES:-0} 样本"
    echo ""
    
    TOTAL_SAMPLES=$((${C4_SAMPLES:-0} + ${ARXIV_SAMPLES:-0} + ${WIKIPEDIA_SAMPLES:-0} + ${GITHUB_SAMPLES:-0} + ${STACKEXCHANGE_SAMPLES:-0}))
    echo "总样本数: $TOTAL_SAMPLES"
    echo "=============================================="
    echo ""
fi

# ============== 创建目录 ==============

mkdir -p "$RAW_DIR"
mkdir -p "$OUTPUT_DIR"

# ============== 下载函数 ==============

download_domain() {
    local domain=$1
    local samples=$2
    local output_file="$RAW_DIR/${domain}.jsonl"
    
    if [ "$samples" -eq 0 ]; then
        echo "⏭ 跳过 $domain (样本数为0)"
        return 0
    fi
    
    # 检查是否已下载完成
    if [ -f "$output_file" ]; then
        local existing_count=$(wc -l < "$output_file")
        if [ "$existing_count" -ge "$samples" ]; then
            echo "✓ $domain 已存在 ($existing_count 样本), 跳过"
            return 0
        fi
    fi
    
    echo ""
    echo "📥 下载 $domain ($samples 样本)..."
    echo ""
    
    python3 -c "
import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 代理
os.environ.setdefault('http_proxy', 'http://your-proxy:port')
os.environ.setdefault('https_proxy', 'http://your-proxy:port')

domain = '$domain'
max_samples = $samples
output_file = '$output_file'
temp_file = output_file + '.tmp'

# 检查断点续传
start_count = 0
if os.path.exists(temp_file):
    with open(temp_file, 'r') as f:
        start_count = sum(1 for _ in f)
    print(f'[{domain}] 从 {start_count} 续传')

try:
    ds = load_dataset(
        'togethercomputer/RedPajama-Data-1T',
        name=domain,
        split='train',
        streaming=True,
        trust_remote_code=True,
    )
    
    count = start_count
    mode = 'a' if start_count > 0 else 'w'
    
    ds_iter = iter(ds)
    # 跳过已下载的
    if start_count > 0:
        for _ in range(start_count):
            try:
                next(ds_iter)
            except StopIteration:
                break
    
    with open(temp_file, mode, encoding='utf-8') as f:
        pbar = tqdm(total=max_samples, initial=start_count, desc=f'[{domain}]')
        for sample in ds_iter:
            if count >= max_samples:
                break
            
            text = sample.get('text', '')
            if not text:
                continue
            
            record = {'text': text, 'meta': sample.get('meta', {})}
            f.write(json.dumps(record, ensure_ascii=False) + '\\n')
            count += 1
            pbar.update(1)
        pbar.close()
    
    # 下载完成，重命名
    os.rename(temp_file, output_file)
    print(f'✓ [{domain}] 完成 ({count} 样本)')
    
except Exception as e:
    print(f'✗ [{domain}] 失败: {e}')
    exit(1)
"
}

# ============== 执行下载 ==============

if [ -z "$TOKENIZE_ONLY" ]; then
    echo ""
    echo "🔄 开始下载..."
    echo ""
    
    # 按域下载 (可以并行，但为了稳定性使用串行)
    if [ "${C4_SAMPLES:-0}" -gt 0 ]; then
        download_domain "c4" "$C4_SAMPLES"
    fi
    
    if [ "${ARXIV_SAMPLES:-0}" -gt 0 ]; then
        download_domain "arxiv" "$ARXIV_SAMPLES"
    fi
    
    if [ "${WIKIPEDIA_SAMPLES:-0}" -gt 0 ]; then
        download_domain "wikipedia" "$WIKIPEDIA_SAMPLES"
    fi
    
    if [ "${GITHUB_SAMPLES:-0}" -gt 0 ]; then
        download_domain "github" "$GITHUB_SAMPLES"
    fi
    
    if [ "${STACKEXCHANGE_SAMPLES:-0}" -gt 0 ]; then
        download_domain "stackexchange" "$STACKEXCHANGE_SAMPLES"
    fi
    
    echo ""
    echo "✅ 所有域下载完成!"
    echo ""
    
    # 统计下载结果
    echo "📊 下载统计:"
    for f in "$RAW_DIR"/*.jsonl; do
        if [ -f "$f" ]; then
            domain=$(basename "$f" .jsonl)
            count=$(wc -l < "$f")
            size=$(du -h "$f" | cut -f1)
            echo "  $domain: $count 样本, $size"
        fi
    done
    echo ""
fi

# ============== Tokenize ==============

echo ""
echo "🔄 开始 Tokenize..."
echo ""

python3 << 'EOF'
import json
import os
import sys
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
VAL_RATIO = 0.01

print(f"输入目录: {RAW_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"Tokenizer: {TOKENIZER_NAME}")
print(f"进程数: {NUM_WORKERS}")
print()

# 检查是否已存在
train_bin = OUTPUT_DIR / "train.bin"
val_bin = OUTPUT_DIR / "val.bin"

if train_bin.exists() and val_bin.exists():
    print(f"⚠ 输出文件已存在:")
    print(f"  {train_bin}")
    print(f"  {val_bin}")
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
if vocab_size > 65536:
    dtype = np.uint32
    print("⚠ vocab_size > 65536, 使用 uint32")
else:
    dtype = np.uint16
    print("使用 uint16")

# 加载所有文本
print()
print("加载数据文件...")
texts = []
jsonl_files = sorted(RAW_DIR.glob("*.jsonl"))

if not jsonl_files:
    print(f"❌ 未找到 .jsonl 文件: {RAW_DIR}")
    sys.exit(1)

print(f"找到 {len(jsonl_files)} 个文件:")
for f in jsonl_files:
    print(f"  - {f.name}")

for jsonl_file in tqdm(jsonl_files, desc="加载文件"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

print(f"总样本数: {len(texts):,}")

# Tokenize 函数
def tokenize_batch(args):
    batch_texts, tokenizer_name, eos_id = args
    from transformers import AutoTokenizer
    
    tok = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    all_ids = []
    encoded = tok(batch_texts, add_special_tokens=False)
    
    for ids in encoded["input_ids"]:
        ids_with_eos = ids + [eos_id]
        all_ids.extend(ids_with_eos)
    
    return all_ids

# 分批
print()
print(f"开始 tokenize (workers={NUM_WORKERS}, batch_size={BATCH_SIZE})...")

chunks = []
for i in range(0, len(texts), BATCH_SIZE):
    chunk_texts = texts[i:i + BATCH_SIZE]
    chunks.append((chunk_texts, TOKENIZER_NAME, eos_token_id))

print(f"总批次: {len(chunks)}")

# 多进程处理
all_ids = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = list(tqdm(
        executor.map(tokenize_batch, chunks),
        total=len(chunks),
        desc="Tokenizing",
    ))
    
    for chunk_ids in results:
        all_ids.extend(chunk_ids)

print(f"总 tokens: {len(all_ids):,}")

# 转换为数组
data = np.array(all_ids, dtype=dtype)

# 分割训练/验证集
val_size = max(int(len(data) * VAL_RATIO), 10000)
val_data = data[-val_size:]
train_data = data[:-val_size]

print(f"训练集: {len(train_data):,} tokens")
print(f"验证集: {len(val_data):,} tokens")

# 写入文件
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print(f"写入 {train_bin}...")
train_mmap = np.memmap(str(train_bin), dtype=dtype, mode="w+", shape=(len(train_data),))
train_mmap[:] = train_data
train_mmap.flush()
print(f"✓ {train_bin.stat().st_size / (1024**3):.2f} GB")

print(f"写入 {val_bin}...")
val_mmap = np.memmap(str(val_bin), dtype=dtype, mode="w+", shape=(len(val_data),))
val_mmap[:] = val_data
val_mmap.flush()
print(f"✓ {val_bin.stat().st_size / (1024**3):.2f} GB")

# 写入 dtype
dtype_str = "uint32" if dtype == np.uint32 else "uint16"
(OUTPUT_DIR / "dtype.txt").write_text(dtype_str)

print()
print("=" * 50)
print("✅ Tokenize 完成!")
print(f"  train.bin: {train_bin}")
print(f"  val.bin: {val_bin}")
print(f"  dtype: {dtype_str}")
print(f"  总 tokens: {len(all_ids):,}")
print("=" * 50)
EOF

echo ""
echo "=============================================="
echo "🎉 全部完成!"
echo ""
echo "输出文件:"
echo "  $OUTPUT_DIR/train.bin"
echo "  $OUTPUT_DIR/val.bin"
echo ""
echo "下一步: 使用以下命令开始训练:"
echo "  bash scripts/run_sparse_train.sh --data_dir $OUTPUT_DIR"
echo "=============================================="
