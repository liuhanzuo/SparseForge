#!/usr/bin/env bash
###############################################################################
# prepare_mixed_c4_based.sh
#
# 以 C4 为基准准备混合数据集
#
# 逻辑：
#   1. 首先处理 C4 数据，计算 C4 的 token 总数作为基准
#   2. 根据配比计算其他域需要的 token 数
#   3. 如果某域数据不足，复制补全
#
# 例如：
#   - C4 占 60%，假设 C4 有 10B tokens
#   - 则总数据集应该是 10B / 0.6 = 16.67B tokens
#   - ArXiv 占 15%，需要 16.67B * 0.15 = 2.5B tokens
#   - GitHub 占 15%，需要 16.67B * 0.15 = 2.5B tokens  
#   - SE 占 10%，需要 16.67B * 0.10 = 1.67B tokens
#   - 如果某域不足，复制补全
#
# 用法：
#   # 默认配比
#   bash scripts/prepare_mixed_c4_based.sh
#
#   # 自定义配比
#   C4_RATIO=0.5 ARXIV_RATIO=0.2 bash scripts/prepare_mixed_c4_based.sh
#
#   # 强制重新生成
#   bash scripts/prepare_mixed_c4_based.sh --force
###############################################################################

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")

# Tokenizer 路径
TOKENIZER="${TOKENIZER:-models/NousResearch--Llama-2-7b-hf}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-data/mixed_c4_based}"

# 并行 workers
NUM_WORKERS="${NUM_WORKERS:-48}"

# 配比（必须加起来 = 1.0）
C4_RATIO="${C4_RATIO:-0.60}"         # 60% - C4 (基准)
ARXIV_RATIO="${ARXIV_RATIO:-0.15}"   # 15% - ArXiv
GITHUB_RATIO="${GITHUB_RATIO:-0.15}" # 15% - GitHub
SE_RATIO="${SE_RATIO:-0.10}"         # 10% - StackExchange

# 随机种子
SEED="${SEED:-42}"

# 解析参数
EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

echo "=============================================================="
echo "以 C4 为基准的混合数据集准备"
echo "=============================================================="
echo "Project: ${PROJECT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo ""
echo "配比:"
echo "  C4 (基准):     ${C4_RATIO} (60%)"
echo "  ArXiv:         ${ARXIV_RATIO} (15%)"
echo "  GitHub:        ${GITHUB_RATIO} (15%)"
echo "  StackExchange: ${SE_RATIO} (10%)"
echo ""
echo "策略: 以 C4 的 token 数为基准，其他域按比例取数据，不足复制补全"
echo "=============================================================="
echo ""

CMD=(
    python "${SCRIPT_DIR}/prepare_mixed_c4_based.py"
    --tokenizer "${TOKENIZER}"
    --output_dir "${OUTPUT_DIR}"
    --num_workers "${NUM_WORKERS}"
    --c4_ratio "${C4_RATIO}"
    --arxiv_ratio "${ARXIV_RATIO}"
    --github_ratio "${GITHUB_RATIO}"
    --se_ratio "${SE_RATIO}"
    --seed "${SEED}"
)

CMD+=("${EXTRA_ARGS[@]}")

cd "${PROJECT_DIR}"
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=============================================================="
echo "Done!"
echo "Output: ${OUTPUT_DIR}/train.bin, ${OUTPUT_DIR}/val.bin"
echo "=============================================================="
