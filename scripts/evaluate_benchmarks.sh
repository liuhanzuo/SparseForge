#!/usr/bin/env bash
# =============================================================================
# 评测脚本：使用 lm-evaluation-harness 测试论文中的所有 benchmark
# 
# 论文测试的 Benchmark：
#   - BoolQ: 阅读理解（是/否问题）
#   - RTE: 文本蕴含识别
#   - HellaSwag: 常识推理（故事结尾选择）
#   - WinoGrande: 常识推理（代词消歧）
#   - ARC-e: AI2 推理挑战（简单）
#   - ARC-c: AI2 推理挑战（困难）
#   - OBQA (OpenBookQA): 开放式常识问答
#
# 使用方法：
#   bash scripts/evaluate_benchmarks.sh <model_path> [output_dir] [batch_size]
#
# 示例：
#   bash scripts/evaluate_benchmarks.sh out_llama/checkpoint_1000
#   bash scripts/evaluate_benchmarks.sh out_llama/checkpoint_1000 results/eval 8
# =============================================================================

set -euo pipefail

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [output_dir] [batch_size]"
    echo ""
    echo "Arguments:"
    echo "  model_path   - Path to the model checkpoint or HuggingFace model name"
    echo "  output_dir   - (Optional) Directory to save results (default: results/benchmarks)"
    echo "  batch_size   - (Optional) Batch size for evaluation (default: 4)"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="${2:-results/benchmarks}"
BATCH_SIZE="${3:-4}"

# 获取模型名称（用于输出目录）
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"

mkdir -p "$RESULT_DIR"

echo "=============================================="
echo "LLM Benchmark Evaluation"
echo "=============================================="
echo "Model Path: $MODEL_PATH"
echo "Output Dir: $RESULT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# 检查 lm-eval 是否安装
if ! command -v lm_eval &> /dev/null; then
    echo "[WARNING] lm-evaluation-harness not found. Installing..."
    pip install lm-eval
fi

# 定义论文中使用的 benchmark 任务
# 注意：lm-evaluation-harness 中的任务名称
TASKS="boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"

echo ""
echo "[INFO] Running evaluation on tasks: $TASKS"
echo ""

# 运行评测
# --model hf: 使用 HuggingFace 模型
# --model_args: 指定模型路径和精度
# --tasks: 指定要评测的任务
# --batch_size: 批次大小
# --output_path: 输出结果路径
# --log_samples: 记录样本级别的结果

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=float16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RESULT_DIR" \
    --log_samples \
    2>&1 | tee "${RESULT_DIR}/eval.log"

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: $RESULT_DIR"
echo "=============================================="

# 解析并显示结果摘要
echo ""
echo "=== Results Summary ==="
python3 << EOF
import json
import os
import glob

result_dir = "$RESULT_DIR"
result_files = glob.glob(os.path.join(result_dir, "*.json"))

if not result_files:
    print("No result files found.")
    exit(0)

# 找到最新的结果文件
latest_file = max(result_files, key=os.path.getctime)

with open(latest_file, 'r') as f:
    data = json.load(f)

results = data.get('results', {})

# 任务名称映射（lm-eval 名称 -> 论文名称）
task_mapping = {
    'boolq': 'BoolQ',
    'rte': 'RTE',
    'hellaswag': 'HellaSwag',
    'winogrande': 'WinoGrande',
    'arc_easy': 'ARC-e',
    'arc_challenge': 'ARC-c',
    'openbookqa': 'OBQA',
}

print(f"{'Task':<15} {'Metric':<15} {'Score':>10}")
print("-" * 42)

# HellaSwag 和 ARC-Challenge 使用 acc_norm（长度归一化准确率），与 CAST 等论文一致
ACC_NORM_TASKS = {'hellaswag', 'arc_challenge'}

scores = []
for task_key, task_name in task_mapping.items():
    if task_key in results:
        task_result = results[task_key]
        # 根据任务类型选择合适的 metric
        if task_key in ACC_NORM_TASKS:
            acc = task_result.get('acc_norm,none',
                  task_result.get('acc_norm',
                  task_result.get('acc,none',
                  task_result.get('acc', 0))))
            metric_label = 'Acc (norm)'
        else:
            acc = task_result.get('acc,none',
                  task_result.get('acc',
                  task_result.get('acc_norm,none',
                  task_result.get('acc_norm', 0))))
            metric_label = 'Accuracy'
        if isinstance(acc, (int, float)):
            score = acc * 100  # 转换为百分比
            scores.append(score)
            print(f"{task_name:<15} {metric_label:<15} {score:>10.2f}")

if scores:
    mean_score = sum(scores) / len(scores)
    print("-" * 42)
    print(f"{'Mean':<15} {'':<15} {mean_score:>10.2f}")
EOF

echo ""
echo "Full results available in: ${RESULT_DIR}"
