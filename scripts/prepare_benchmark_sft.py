#!/usr/bin/env python3
"""
针对 lm_eval zero-shot 评测的 QA 格式学习数据集制作脚本。

核心思路：
  lm_eval 的 zero-shot 评测基于 log-likelihood 选择最佳答案。
  稀疏化后模型的问答格式能力退化，需要通过 SFT 恢复。
  为避免使用 benchmark 自身的 train split 导致过拟合嫌疑，
  我们使用与 lm_eval benchmark 完全不重合的 QA 数据集，
  让模型学会 "Question-Answer" 格式的 pattern。

使用的 QA 数据集 (与 lm_eval benchmark 零重合):
  - CommonsenseQA   (tau/commonsense_qa)       — 常识推理选择题
  - SocialIQA       (allenai/social_i_qa)      — 社交常识 QA
  - CosmosQA        (cosmos_qa)                — 阅读理解 QA
  - SciQ            (allenai/sciq)             — 科学问答
  - RACE            (ehovy/race)               — 阅读理解选择题
  - QASC            (allenai/qasc)             — 科学常识组合 QA
  - DREAM           (dataset-org/dream)        — 对话理解选择题

输出: train.bin + val.bin (uint16 memmap)，兼容 Adaptive-Sparse-Trainer。

用法:
    python scripts/prepare_benchmark_sft.py \
        --output_dir data/qa_format_sft_llama \
        --tokenizer_path models/Llama--Llama2-7b \
        --repeat 3

    # 混合 dolmino 预训练数据 (1:1):
    python scripts/prepare_benchmark_sft.py \
        --output_dir data/qa_dolmino_mix_llama \
        --tokenizer_path models/Llama--Llama2-7b \
        --repeat 3 \
        --mix_data_dir data/dolmino-mix-1124-raw \
        --mix_ratio 1.0

    # 然后训练时指定:
    #   --dataset qa_format_sft_llama
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark_sft")


# ======================================================================
# 每个 QA 数据集的格式化函数
# 将 (question, correct_answer) 转换为自然语言 NTP 文本
# 这些数据集与 lm_eval benchmark 完全不重合
# ======================================================================

def format_commonsenseqa(example: dict) -> str:
    """CommonsenseQA: 常识推理选择题 (5选1)"""
    question = example["question"]
    choices = example["choices"]
    answer_key = example["answerKey"]  # "A", "B", "C", "D", "E"

    labels = choices["label"]
    texts = choices["text"]

    correct_text = ""
    for i, lbl in enumerate(labels):
        if lbl == answer_key:
            correct_text = texts[i]
            break

    return f"Question: {question}\nAnswer: {correct_text}"


def format_social_iqa(example: dict) -> str:
    """SocialIQA: 社交常识 QA (3选1)"""
    context = example["context"]
    question = example["question"]
    label = int(example["label"]) - 1  # 1-indexed -> 0-indexed

    answers = [example["answerA"], example["answerB"], example["answerC"]]
    correct = answers[label] if 0 <= label < 3 else answers[0]

    return f"{context}\nQuestion: {question}\nAnswer: {correct}"


def format_cosmosqa(example: dict) -> str:
    """CosmosQA: 阅读理解 QA (4选1)"""
    context = example["context"]
    question = example["question"]
    label = int(example["label"])  # 0, 1, 2, 3

    answers = [
        example["answer0"],
        example["answer1"],
        example["answer2"],
        example["answer3"],
    ]
    correct = answers[label] if 0 <= label < 4 else answers[0]

    return f"{context}\nQuestion: {question}\nAnswer: {correct}"


def format_sciq(example: dict) -> str:
    """SciQ: 科学问答 (有 support passage)"""
    support = example.get("support", "") or ""
    question = example["question"]
    correct_answer = example["correct_answer"]

    if support.strip():
        return f"{support.strip()}\nQuestion: {question}\nAnswer: {correct_answer}"
    else:
        return f"Question: {question}\nAnswer: {correct_answer}"


def format_race(example: dict) -> str:
    """RACE: 阅读理解选择题 (4选1)"""
    article = example["article"]
    question = example["question"]
    options = example["options"]  # list of 4 strings
    answer = example["answer"]   # "A", "B", "C", "D"

    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    idx = answer_map.get(answer, 0)
    correct = options[idx] if idx < len(options) else options[0]

    return f"{article}\nQuestion: {question}\nAnswer: {correct}"


def format_qasc(example: dict) -> str:
    """QASC: 科学常识组合 QA (8选1)"""
    question = example["question"]
    choices = example["choices"]
    answer_key = example["answerKey"]  # "A"~"H"

    labels = choices["label"]
    texts = choices["text"]

    correct_text = ""
    for i, lbl in enumerate(labels):
        if lbl == answer_key:
            correct_text = texts[i]
            break

    # QASC 有 fact1 和 fact2 作为支撑事实
    fact1 = example.get("fact1", "") or ""
    fact2 = example.get("fact2", "") or ""
    facts = ""
    if fact1.strip() and fact2.strip():
        facts = f"{fact1.strip()} {fact2.strip()}\n"
    elif fact1.strip():
        facts = f"{fact1.strip()}\n"

    return f"{facts}Question: {question}\nAnswer: {correct_text}"


def format_dream(example: dict) -> str:
    """DREAM: 对话理解选择题 (3选1)"""
    dialogue = example["dialogue"]  # list of strings
    question = example["question"]
    choice = example["choice"]      # list of 3 strings
    answer = example["answer"]      # correct answer text

    # 将对话列表拼接为文本
    dialogue_text = "\n".join(dialogue)

    return f"{dialogue_text}\nQuestion: {question}\nAnswer: {answer}"


# ======================================================================
# 数据集加载
# ======================================================================

BENCHMARK_CONFIGS = {
    "commonsenseqa": {
        "hf_path": "tau/commonsense_qa",
        "hf_name": None,
        "split": "train",
        "formatter": format_commonsenseqa,
        "description": "CommonsenseQA (常识推理选择题, ~9.7k)",
    },
    "social_iqa": {
        "hf_path": "allenai/social_i_qa",
        "hf_name": None,
        "split": "train",
        "formatter": format_social_iqa,
        "description": "SocialIQA (社交常识 QA, ~33.4k)",
    },
    "cosmosqa": {
        "hf_path": "cosmos_qa",
        "hf_name": None,
        "split": "train",
        "formatter": format_cosmosqa,
        "description": "CosmosQA (阅读理解 QA, ~25.3k)",
    },
    "sciq": {
        "hf_path": "allenai/sciq",
        "hf_name": None,
        "split": "train",
        "formatter": format_sciq,
        "description": "SciQ (科学问答, ~11.7k)",
    },
    "race_middle": {
        "hf_path": "ehovy/race",
        "hf_name": "middle",
        "split": "train",
        "formatter": format_race,
        "description": "RACE-Middle (阅读理解选择题-初中, ~25.4k)",
    },
    "race_high": {
        "hf_path": "ehovy/race",
        "hf_name": "high",
        "split": "train",
        "formatter": format_race,
        "description": "RACE-High (阅读理解选择题-高中, ~62.4k)",
    },
    "qasc": {
        "hf_path": "allenai/qasc",
        "hf_name": None,
        "split": "train",
        "formatter": format_qasc,
        "description": "QASC (科学常识组合 QA, ~8.1k)",
    },
    "dream": {
        "hf_path": "dataset-org/dream",
        "hf_name": None,
        "split": "train",
        "formatter": format_dream,
        "description": "DREAM (对话理解选择题, ~6.1k)",
    },
}


def load_benchmark_texts(
    benchmarks: List[str],
    cache_dir: str,
    repeat: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    """
    加载指定 benchmark 的训练集，转换为 NTP 文本列表。

    Args:
        benchmarks: 要加载的 benchmark 名称列表
        cache_dir: HuggingFace datasets 缓存目录
        repeat: 数据重复次数（小数据集可以多重复几次）

    Returns:
        texts: 所有格式化后的文本列表
        stats: 每个 benchmark 的样本数统计
    """
    from datasets import load_dataset

    all_texts = []
    stats = {}

    for bench_name in benchmarks:
        if bench_name not in BENCHMARK_CONFIGS:
            logger.warning(f"未知 benchmark: {bench_name}，跳过")
            continue

        cfg = BENCHMARK_CONFIGS[bench_name]
        logger.info(f"  加载 {bench_name}: {cfg['description']}")

        try:
            kwargs = {
                "path": cfg["hf_path"],
                "split": cfg["split"],
                "cache_dir": cache_dir,
                "trust_remote_code": True,
            }
            if cfg["hf_name"] is not None:
                kwargs["name"] = cfg["hf_name"]

            dataset = load_dataset(**kwargs)
        except Exception as e:
            logger.error(f"  加载 {bench_name} 失败: {e}")
            continue

        formatter = cfg["formatter"]
        bench_texts = []

        for example in dataset:
            try:
                text = formatter(example)
                if text and text.strip():
                    bench_texts.append(text.strip())
            except Exception as e:
                # 跳过格式化失败的样本
                continue

        n_original = len(bench_texts)

        # 重复数据
        if repeat > 1:
            bench_texts = bench_texts * repeat

        all_texts.extend(bench_texts)
        stats[bench_name] = {
            "original": n_original,
            "after_repeat": len(bench_texts),
        }
        logger.info(f"    {bench_name}: {n_original} 条 × {repeat} = {n_original * repeat} 条")

    return all_texts, stats


# ======================================================================
# Tokenize + 写入 memmap
# ======================================================================

def tokenize_and_write(
    texts: List[str],
    tokenizer,
    output_dir: str,
    val_ratio: float = 0.005,
    seed: int = 42,
):
    """
    将文本列表 tokenize 并写入 train.bin / val.bin (uint16 memmap)。

    每条文本 tokenize 后追加 EOS token，所有 token 拼接成一个长序列。
    训练时框架会随机截取 block_size 长度的片段。
    """
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    # 确认 uint16 够用 (LLaMA-2 vocab_size = 32000)
    if vocab_size > 65535:
        dtype = np.uint32
        dtype_name = "uint32"
        logger.warning(f"词表大小 {vocab_size} > 65535，使用 uint32")
    else:
        dtype = np.uint16
        dtype_name = "uint16"
        logger.info(f"词表大小 {vocab_size}，使用 {dtype_name}")

    # 打乱文本
    rng = random.Random(seed)
    rng.shuffle(texts)

    # 分割 train / val
    n_val = max(1, int(len(texts) * val_ratio))
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    logger.info(f"  Train: {len(train_texts)} 条, Val: {len(val_texts)} 条")

    def tokenize_to_ids(text_list: List[str], desc: str) -> np.ndarray:
        """批量 tokenize 并拼接为 numpy 数组。"""
        all_ids = []
        BATCH_SIZE = 512

        for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc=desc):
            batch = text_list[i:i + BATCH_SIZE]
            encoded = tokenizer(batch, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                if len(ids) == 0:
                    continue
                # 截断过长的文本 (benchmark 数据一般不会太长)
                if len(ids) > 2048:
                    ids = ids[:2048]
                ids.append(eos_id)
                all_ids.extend(ids)

        return np.array(all_ids, dtype=dtype)

    logger.info("  Tokenizing train set...")
    train_ids = tokenize_to_ids(train_texts, "Tokenize train")
    logger.info(f"  Train: {len(train_ids):,} tokens ({len(train_ids) * dtype().itemsize / 1024**2:.1f} MB)")

    logger.info("  Tokenizing val set...")
    val_ids = tokenize_to_ids(val_texts, "Tokenize val")
    logger.info(f"  Val: {len(val_ids):,} tokens ({len(val_ids) * dtype().itemsize / 1024**2:.1f} MB)")

    # 写入 memmap
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    logger.info(f"  写入 {train_path} ...")
    train_mm = np.memmap(train_path, dtype=dtype, mode="w+", shape=train_ids.shape)
    train_mm[:] = train_ids
    train_mm.flush()
    del train_mm

    logger.info(f"  写入 {val_path} ...")
    val_mm = np.memmap(val_path, dtype=dtype, mode="w+", shape=val_ids.shape)
    val_mm[:] = val_ids
    val_mm.flush()
    del val_mm

    # 写 dtype.txt (如果是 uint32 的话需要标记)
    if dtype_name == "uint32":
        dtype_path = os.path.join(output_dir, "dtype.txt")
        with open(dtype_path, "w") as f:
            f.write("uint32")
        logger.info(f"  写入 {dtype_path}")
    else:
        # uint16 不需要 dtype.txt，框架默认就是 uint16
        # 但为了明确，也写一个
        dtype_path = os.path.join(output_dir, "dtype.txt")
        with open(dtype_path, "w") as f:
            f.write("uint16")
        logger.info(f"  写入 {dtype_path}")

    return {
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "dtype": dtype_name,
    }


# ======================================================================
# 数据混合: 将 QA tokens 与预训练数据 (如 dolmino) 按比例混合
# ======================================================================

def mix_with_pretrain_data(
    qa_train_ids: np.ndarray,
    mix_data_dir: str,
    mix_ratio: float = 1.0,
    block_size: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """
    将 QA 训练 tokens 与预训练数据按比例混合。

    混合策略:
      1. 从预训练数据 (train.bin) 中随机采样 len(qa_train_ids) * mix_ratio 个 tokens
      2. 将 QA 和预训练数据按 block_size 切块
      3. 随机交错排列这些块

    Args:
        qa_train_ids: QA 数据的 token ids (numpy array)
        mix_data_dir: 预训练数据目录 (包含 train.bin, 可选 dtype.txt)
        mix_ratio: 预训练数据与 QA 数据的比例 (1.0 = 1:1)
        block_size: 交错混合时的块大小
        seed: 随机种子

    Returns:
        mixed_ids: 混合后的 token ids (numpy array, dtype=uint32)
    """
    rng = np.random.RandomState(seed)

    # ---- 读取预训练数据 ----
    pretrain_bin = os.path.join(mix_data_dir, "train.bin")
    if not os.path.exists(pretrain_bin):
        raise FileNotFoundError(f"预训练数据不存在: {pretrain_bin}")

    # 检测 dtype
    dtype_file = os.path.join(mix_data_dir, "dtype.txt")
    if os.path.exists(dtype_file):
        with open(dtype_file, "r") as f:
            pretrain_dtype_name = f.read().strip()
    else:
        # 尝试从 metadata.json 读取
        meta_file = os.path.join(mix_data_dir, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            pretrain_dtype_name = meta.get("dtype", "uint16")
        else:
            pretrain_dtype_name = "uint16"

    pretrain_dtype = np.uint32 if pretrain_dtype_name == "uint32" else np.uint16
    logger.info(f"  预训练数据 dtype: {pretrain_dtype_name}")

    # 获取预训练数据总 token 数
    file_size = os.path.getsize(pretrain_bin)
    pretrain_total_tokens = file_size // np.dtype(pretrain_dtype).itemsize
    logger.info(f"  预训练数据总 tokens: {pretrain_total_tokens:,}")

    # 计算需要采样的 token 数
    n_qa = len(qa_train_ids)
    n_pretrain_needed = int(n_qa * mix_ratio)
    logger.info(f"  QA tokens: {n_qa:,}, 需要采样预训练 tokens: {n_pretrain_needed:,} (ratio={mix_ratio})")

    if n_pretrain_needed > pretrain_total_tokens:
        logger.warning(
            f"  预训练数据不足! 需要 {n_pretrain_needed:,} 但只有 {pretrain_total_tokens:,}，"
            f"将使用全部预训练数据"
        )
        n_pretrain_needed = pretrain_total_tokens

    # ---- 从预训练数据中随机采样连续片段 ----
    pretrain_mm = np.memmap(pretrain_bin, dtype=pretrain_dtype, mode="r")

    # 采样策略: 随机选择多个连续片段，每个片段长度为 block_size
    # 这样保持了预训练数据的局部连贯性
    n_blocks_needed = (n_pretrain_needed + block_size - 1) // block_size
    max_start = pretrain_total_tokens - block_size

    if max_start <= 0:
        # 预训练数据太短，直接全部使用
        pretrain_sampled = np.array(pretrain_mm[:n_pretrain_needed], dtype=np.uint32)
    else:
        # 随机选择起始位置
        starts = rng.randint(0, max_start, size=n_blocks_needed)
        blocks = []
        collected = 0
        for s in starts:
            remaining = n_pretrain_needed - collected
            take = min(block_size, remaining)
            blocks.append(np.array(pretrain_mm[s:s + take], dtype=np.uint32))
            collected += take
            if collected >= n_pretrain_needed:
                break
        pretrain_sampled = np.concatenate(blocks)[:n_pretrain_needed]

    del pretrain_mm
    logger.info(f"  采样完成: {len(pretrain_sampled):,} tokens")

    # ---- 将 QA 数据也转为 uint32 ----
    qa_uint32 = qa_train_ids.astype(np.uint32)

    # ---- 按 block_size 切块并交错混合 ----
    def split_into_blocks(arr, bs):
        """将数组切成 block_size 大小的块"""
        n_full = len(arr) // bs
        remainder = len(arr) % bs
        result = []
        for i in range(n_full):
            result.append(arr[i * bs:(i + 1) * bs])
        if remainder > 0:
            result.append(arr[n_full * bs:])
        return result

    qa_blocks = split_into_blocks(qa_uint32, block_size)
    pretrain_blocks = split_into_blocks(pretrain_sampled, block_size)

    logger.info(f"  QA 块数: {len(qa_blocks)}, 预训练块数: {len(pretrain_blocks)}")

    # 合并所有块并随机打乱
    all_blocks = qa_blocks + pretrain_blocks
    rng.shuffle(all_blocks)

    mixed_ids = np.concatenate(all_blocks)
    logger.info(f"  混合后总 tokens: {len(mixed_ids):,}")

    return mixed_ids


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="制作 QA 格式学习数据集 (final finetune 用, 与 lm_eval benchmark 零重合)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录 (会生成 train.bin, val.bin, dtype.txt, metadata.json)",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="models/Llama--Llama2-7b",
        help="Tokenizer 路径 (HuggingFace ID 或本地路径，默认 LLaMA-2-7B)",
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="要包含的 QA 数据集列表 (默认全部)。"
             "可选: commonsenseqa social_iqa cosmosqa sciq race_middle race_high qasc dream",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="数据重复次数 (默认 3，因为 benchmark 训练集较小，约10万条)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.005,
        help="验证集比例 (默认 0.5%%)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace datasets 缓存目录 (默认 data/hf_datasets)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已有输出文件",
    )
    parser.add_argument(
        "--mix_data_dir", type=str, default=None,
        help="预训练数据目录 (包含 train.bin)，用于与 QA 数据混合。"
             "例如: data/dolmino-mix-1124-raw",
    )
    parser.add_argument(
        "--mix_ratio", type=float, default=1.0,
        help="预训练数据与 QA 数据的混合比例 (默认 1.0 即 1:1)。"
             "例如 2.0 表示预训练数据量是 QA 的 2 倍",
    )
    parser.add_argument(
        "--mix_block_size", type=int, default=2048,
        help="混合时的块大小 (默认 2048)，与训练的 block_size 一致",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 默认 QA 数据集列表 (全部)
    if args.benchmarks is None:
        args.benchmarks = [
            "commonsenseqa", "social_iqa", "cosmosqa", "sciq",
            "race_middle", "race_high", "qasc", "dream",
        ]

    # 默认缓存目录
    if args.cache_dir is None:
        # 尝试使用项目内的 hf_datasets 缓存
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.cache_dir = os.path.join(project_dir, "data", "hf_datasets")

    # 检查输出
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.bin")
    if not args.force and os.path.exists(train_path):
        logger.error(f"输出文件已存在: {train_path}，使用 --force 覆盖")
        return

    # ---- 开始 ----
    logger.info("=" * 70)
    logger.info("QA 格式学习数据集制作 (与 lm_eval benchmark 零重合)")
    logger.info("=" * 70)
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"重复次数: {args.repeat}")
    logger.info(f"缓存目录: {args.cache_dir}")
    if args.mix_data_dir:
        logger.info(f"混合预训练数据: {args.mix_data_dir} (ratio={args.mix_ratio})")
    logger.info("")

    # Step 1: 加载 QA 数据
    logger.info("[Step 1/3] 加载 QA 格式数据集...")
    t0 = time.time()
    texts, stats = load_benchmark_texts(
        benchmarks=args.benchmarks,
        cache_dir=args.cache_dir,
        repeat=args.repeat,
    )
    t1 = time.time()
    logger.info(f"  共 {len(texts):,} 条文本，耗时 {t1 - t0:.1f}s")

    if not texts:
        logger.error("没有加载到任何数据！请检查网络连接和 benchmark 名称。")
        return

    # Step 2: 加载 tokenizer
    logger.info(f"\n[Step 2/3] 加载 tokenizer: {args.tokenizer_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_token_id}")

    # Step 3: Tokenize + 写入
    logger.info(f"\n[Step 3/3] Tokenize + 写入 memmap...")
    result = tokenize_and_write(
        texts=texts,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Step 4 (可选): 混合预训练数据
    if args.mix_data_dir:
        logger.info(f"\n[Step 4/4] 混合预训练数据: {args.mix_data_dir}")

        # 读取刚写入的 QA train.bin
        qa_train_path = os.path.join(args.output_dir, "train.bin")
        qa_dtype_name = result["dtype"]
        qa_dtype = np.uint32 if qa_dtype_name == "uint32" else np.uint16
        qa_train_ids = np.memmap(qa_train_path, dtype=qa_dtype, mode="r")
        qa_train_ids = np.array(qa_train_ids)  # 读入内存

        # 解析 mix_data_dir 路径 (支持相对路径)
        mix_data_dir = args.mix_data_dir
        if not os.path.isabs(mix_data_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            mix_data_dir = os.path.join(project_dir, mix_data_dir)

        # 执行混合
        mixed_ids = mix_with_pretrain_data(
            qa_train_ids=qa_train_ids,
            mix_data_dir=mix_data_dir,
            mix_ratio=args.mix_ratio,
            block_size=args.mix_block_size,
            seed=args.seed,
        )

        # 覆盖写入混合后的 train.bin (统一为 uint32)
        mixed_dtype = np.uint32
        mixed_dtype_name = "uint32"

        logger.info(f"  覆盖写入 {qa_train_path} (dtype={mixed_dtype_name})...")
        mixed_mm = np.memmap(qa_train_path, dtype=mixed_dtype, mode="w+", shape=mixed_ids.shape)
        mixed_mm[:] = mixed_ids
        mixed_mm.flush()
        del mixed_mm

        # 更新 dtype.txt
        dtype_path = os.path.join(args.output_dir, "dtype.txt")
        with open(dtype_path, "w") as f:
            f.write(mixed_dtype_name)
        logger.info(f"  更新 {dtype_path} -> {mixed_dtype_name}")

        # 同时把 val.bin 也转为 uint32 (保持一致)
        val_path = os.path.join(args.output_dir, "val.bin")
        if os.path.exists(val_path):
            val_ids = np.memmap(val_path, dtype=qa_dtype, mode="r")
            val_ids_u32 = np.array(val_ids, dtype=np.uint32)
            del val_ids
            val_mm = np.memmap(val_path, dtype=np.uint32, mode="w+", shape=val_ids_u32.shape)
            val_mm[:] = val_ids_u32
            val_mm.flush()
            del val_mm
            logger.info(f"  val.bin 也已转为 {mixed_dtype_name}")

        # 更新 result
        result["train_tokens"] = len(mixed_ids)
        result["dtype"] = mixed_dtype_name
        result["mix_info"] = {
            "mix_data_dir": args.mix_data_dir,
            "mix_ratio": args.mix_ratio,
            "mix_block_size": args.mix_block_size,
            "qa_tokens": len(qa_train_ids),
            "pretrain_tokens_sampled": len(mixed_ids) - len(qa_train_ids),
        }

    # 保存元数据
    metadata = {
        "description": "QA format learning dataset for final finetune (no benchmark overlap)",
        "tokenizer": args.tokenizer_path,
        "dtype": result["dtype"],
        "train_tokens": result["train_tokens"],
        "val_tokens": result["val_tokens"],
        "benchmarks": args.benchmarks,
        "repeat": args.repeat,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "per_benchmark_stats": stats,
    }
    if args.mix_data_dir and "mix_info" in result:
        metadata["mix_info"] = result["mix_info"]

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    logger.info("\n" + "=" * 70)
    logger.info("完成!")
    logger.info("=" * 70)
    logger.info(f"Train tokens: {result['train_tokens']:,}")
    logger.info(f"Val tokens:   {result['val_tokens']:,}")
    logger.info(f"dtype:        {result['dtype']}")
    logger.info("")
    logger.info("各 benchmark 统计:")
    logger.info(f"  {'Benchmark':<18s} {'原始':>8s} {'重复后':>8s}")
    logger.info("  " + "-" * 40)
    for name, s in stats.items():
        logger.info(f"  {name:<18s} {s['original']:>8,} {s['after_repeat']:>8,}")
    logger.info("")
    logger.info(f"输出文件:")
    logger.info(f"  {os.path.join(args.output_dir, 'train.bin')}")
    logger.info(f"  {os.path.join(args.output_dir, 'val.bin')}")
    logger.info(f"  {meta_path}")
    logger.info("")
    logger.info("使用方式:")
    dataset_name = os.path.basename(args.output_dir)
    logger.info(f"  # 在 train_llama.sh 中修改 --dataset {dataset_name}")
    logger.info(f"  # 或者命令行追加: bash train_llama.sh -- --dataset {dataset_name}")


if __name__ == "__main__":
    main()
