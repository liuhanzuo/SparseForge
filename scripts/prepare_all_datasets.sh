#!/usr/bin/env bash
#===============================================================================
# prepare_all_datasets.sh
# 为所有新模型类型准备 tokenized 数据
#
# 用法:
#   bash scripts/prepare_all_datasets.sh [qwen|mistral|deepseek|all]
#
# 示例:
#   bash scripts/prepare_all_datasets.sh qwen      # 只准备 Qwen 数据
#   bash scripts/prepare_all_datasets.sh all       # 准备所有新模型数据
#===============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."
DATA_DIR="${PROJECT_ROOT}/data"

# 代理设置
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 模型定义
declare -A MODELS=(
    ["qwen"]="Qwen/Qwen3-1.7B"
    ["mistral"]="mistralai/Mistral-7B-v0.3"
    ["deepseek"]="deepseek-ai/deepseek-moe-16b-base"
)

prepare_dataset() {
    local model_type=$1
    local tokenizer=${MODELS[$model_type]}
    local output_dir="${DATA_DIR}/c4_${model_type}"
    
    # 特殊处理 deepseek
    if [[ "$model_type" == "deepseek" ]]; then
        output_dir="${DATA_DIR}/c4_deepseek_moe"
    fi
    
    echo ""
    echo "=============================================="
    echo "Preparing dataset for: ${model_type}"
    echo "Tokenizer: ${tokenizer}"
    echo "Output: ${output_dir}"
    echo "=============================================="
    
    mkdir -p "${output_dir}"
    
    # 复制 prepare.py 到目标目录（如果不存在）
    if [[ ! -f "${output_dir}/prepare.py" ]]; then
        cp "${DATA_DIR}/c4_qwen/prepare.py" "${output_dir}/prepare.py"
    fi
    
    # 检查是否已存在数据
    if [[ -f "${output_dir}/train.bin" && -f "${output_dir}/val.bin" ]]; then
        echo "✓ Dataset already exists at ${output_dir}"
        echo "  Use --force flag in prepare.py to regenerate"
        return 0
    fi
    
    # 运行数据准备
    python "${output_dir}/prepare.py" \
        --tokenizer "${tokenizer}" \
        --output_dir "${output_dir}" \
        --c4_dataset_dir "${DATA_DIR}/c4_dataset" \
        --train_shards 128 \
        --val_shards 1 \
        --num_proc 64 \
        --max_workers 8
    
    echo "✓ Dataset prepared: ${output_dir}"
}

# 主逻辑
TARGET=${1:-"all"}

case "$TARGET" in
    qwen|mistral|deepseek)
        prepare_dataset "$TARGET"
        ;;
    all)
        for model_type in qwen mistral deepseek; do
            prepare_dataset "$model_type"
        done
        ;;
    *)
        echo "Usage: $0 [qwen|mistral|deepseek|all]"
        echo ""
        echo "Models:"
        echo "  qwen     - Qwen/Qwen3-1.7B"
        echo "  mistral  - mistralai/Mistral-7B-v0.3"
        echo "  deepseek - deepseek-ai/deepseek-moe-16b-base"
        echo "  all      - Prepare all datasets"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✓ All requested datasets prepared!"
echo "=============================================="
