"""
Benchmark 评测脚本

评测论文中提到的所有 benchmark：
- BoolQ: 阅读理解（是/否问题）
- RTE: 文本蕴含识别  
- HellaSwag: 常识推理（故事结尾选择）
- WinoGrande: 常识推理（代词消歧）
- ARC-e: AI2 推理挑战（简单）
- ARC-c: AI2 推理挑战（困难）
- OBQA (OpenBookQA): 开放式常识问答

使用方法：
    python evaluate_benchmarks.py --model_path <path_to_model> [options]

依赖安装：
    pip install lm-eval
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import torch


def check_lm_eval_installed():
    """检查 lm-evaluation-harness 是否安装"""
    try:
        import lm_eval
        return True
    except ImportError:
        print("[ERROR] lm-evaluation-harness not installed.")
        print("Please install it with: pip install lm-eval")
        return False


def run_evaluation(
    model_path: str,
    tasks: List[str],
    batch_size: int = 4,
    num_fewshot: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "results/benchmarks",
    limit: Optional[int] = None,
) -> Dict:
    """
    运行 benchmark 评测
    
    Args:
        model_path: 模型路径或 HuggingFace 模型名
        tasks: 要评测的任务列表
        batch_size: 批次大小
        num_fewshot: few-shot 样本数（None 表示使用默认值）
        device: 设备（cuda/cpu）
        output_dir: 输出目录
        limit: 限制每个任务的样本数（用于快速测试）
    
    Returns:
        评测结果字典
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    # 创建输出目录
    model_name = os.path.basename(model_path.rstrip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    print("=" * 60)
    print("LLM Benchmark Evaluation")
    print("=" * 60)
    print(f"Model Path: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"Output Dir: {result_dir}")
    if limit:
        print(f"Sample Limit: {limit}")
    print("=" * 60)
    
    # 加载模型
    print("\n[INFO] Loading model...")
    model = HFLM(
        pretrained=model_path,
        dtype=torch.float16,
        device=device,
        trust_remote_code=True,
    )
    
    # 运行评测
    print(f"\n[INFO] Running evaluation on {len(tasks)} tasks...")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=True,
    )
    
    # 保存完整结果
    result_file = os.path.join(result_dir, "results.json")
    with open(result_file, 'w') as f:
        # 移除不可序列化的对象
        save_results = {
            'config': results.get('config', {}),
            'results': results.get('results', {}),
            'model_path': model_path,
            'timestamp': timestamp,
        }
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n[INFO] Results saved to: {result_file}")
    
    return results


def print_results_table(results: Dict):
    """打印结果表格"""
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

    # 需要使用 acc_norm 的任务（社区标准，与 CAST 论文一致）
    # HellaSwag: 使用长度归一化的 acc_norm 是标准做法
    # ARC-Challenge: 使用 acc_norm 是 lm-eval 默认推荐
    ACC_NORM_TASKS = {'hellaswag', 'arc_challenge'}
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Task':<15} {'Metric':<20} {'Score':>10}")
    print("-" * 47)
    
    task_results = results.get('results', {})
    scores = []
    
    for task_key, task_name in task_mapping.items():
        if task_key in task_results:
            task_result = task_results[task_key]
            # 根据任务类型选择合适的 metric
            # HellaSwag/ARC-c 使用 acc_norm（长度归一化），其他任务使用 acc
            acc = None
            if task_key in ACC_NORM_TASKS:
                # 优先使用 acc_norm
                for key in ['acc_norm,none', 'acc_norm', 'acc,none', 'acc']:
                    if key in task_result:
                        acc = task_result[key]
                        break
            else:
                # 其他任务优先使用 acc
                for key in ['acc,none', 'acc', 'acc_norm,none', 'acc_norm']:
                    if key in task_result:
                        acc = task_result[key]
                        break
            
            if acc is not None:
                score = acc * 100  # 转换为百分比
                scores.append(score)
                metric_name = 'Acc (norm)' if (task_key in ACC_NORM_TASKS) else 'Accuracy'
                print(f"{task_name:<15} {metric_name:<20} {score:>10.2f}")
    
    if scores:
        mean_score = sum(scores) / len(scores)
        print("-" * 47)
        print(f"{'Mean':<15} {'':<20} {mean_score:>10.2f}")
    
    print("=" * 60)
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on standard benchmarks (BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or HuggingFace model name"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
        help="Comma-separated list of tasks to evaluate (default: all paper benchmarks)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save results (default: results/benchmarks)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per task for quick testing"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_lm_eval_installed():
        sys.exit(1)
    
    # 解析任务列表
    tasks = [t.strip() for t in args.tasks.split(',')]
    
    # 运行评测
    results = run_evaluation(
        model_path=args.model_path,
        tasks=tasks,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        device=args.device,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    
    # 打印结果表格
    print_results_table(results)


if __name__ == "__main__":
    main()
