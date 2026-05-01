#!/usr/bin/env python3
"""
Synthetic Calibration Data Generator
=====================================

用模型自身续写 C4 的短 prefix，计算 perplexity 并过滤掉高 PPL 样本，
生成更能代表模型"舒适分布"的校准集，从而提升剪枝效果。

用法:
    python prepare_synthetic_calibration.py \
        --model_path models/Llama--Llama2-7b \
        --model_type llama \
        --num_samples 256 \
        --gen_len 512 \
        --prefix_min 1 --prefix_max 4 \
        --ppl_filter_percent 0.15 \
        --temperature 0.8 --top_p 0.95 \
        --seqlen 1024 \
        --cache_dir data/synthetic_calibration \
        --seed 42

生成产物:
    <cache_dir>/synthetic_calibration_<fingerprint>.pkl   -- 与原 calibration_dataset.pkl 格式一致
    <cache_dir>/synthetic_calibration_<fingerprint>.meta.json -- 记录生成参数（可复现）
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# 工具函数
# ============================================================================

def compute_fingerprint(args: argparse.Namespace) -> str:
    """根据生成参数计算一个短 hash，用于缓存文件名去重。"""
    key_fields = {
        "model_path": args.model_path,
        "num_samples": args.num_samples,
        "gen_len": args.gen_len,
        "prefix_min": args.prefix_min,
        "prefix_max": args.prefix_max,
        "ppl_filter_percent": args.ppl_filter_percent,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seqlen": args.seqlen,
        "seed": args.seed,
    }
    h = hashlib.md5(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:10]
    return h


def load_c4_seed_texts(data_dir: str, max_files: int = 17) -> list:
    """从本地 C4 json.gz 中加载原始文本。"""
    from datasets import load_dataset

    json_paths = [
        os.path.join(data_dir, f"c4-train.{i:05d}-of-01024.json.gz")
        for i in range(max_files)
    ]
    json_paths = [p for p in json_paths if os.path.exists(p)]
    if not json_paths:
        raise FileNotFoundError(
            f"在 {data_dir} 中找不到 C4 训练文件 (c4-train.XXXXX-of-01024.json.gz)"
        )

    cache_dir = os.path.join(data_dir, "cache", "train")
    print(f"[Synthetic] 正在从 {len(json_paths)} 个 C4 文件加载种子文本...")
    dataset = load_dataset(
        "json",
        data_files={"train": json_paths},
        split="train",
        cache_dir=cache_dir,
    )
    return dataset


def load_tokenizer(model_path: str, model_type: str):
    """根据 model_type 加载合适的 tokenizer。"""
    model_type_lower = model_type.lower()
    if model_type_lower in ("llama", "qwen"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            local_files_only=os.path.isdir(model_path),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        # OPT / GPT-2: 使用 tiktoken 或 HF tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
                local_files_only=os.path.isdir(model_path),
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            import tiktoken
            return tiktoken.get_encoding("gpt2")


def tokenizer_encode(tokenizer, text: str) -> List[int]:
    """统一 tokenizer encode 接口（兼容 HF tokenizer 和 tiktoken）。"""
    if hasattr(tokenizer, "encode_ordinary"):
        # tiktoken
        return tokenizer.encode_ordinary(text)
    else:
        # HF tokenizer
        return tokenizer.encode(text, add_special_tokens=False)


def tokenizer_decode(tokenizer, token_ids: List[int]) -> str:
    """统一 tokenizer decode 接口。"""
    if hasattr(tokenizer, "decode_batch"):
        # tiktoken
        return tokenizer.decode(token_ids)
    else:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def load_hf_model(model_path: str, device: str = "cuda", dtype=torch.float16):
    """加载 HuggingFace CausalLM 模型用于生成。"""
    from transformers import AutoModelForCausalLM

    print(f"[Synthetic] 加载模型: {model_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=os.path.isdir(model_path),
    )
    model = model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Synthetic] 模型参数量: {num_params / 1e6:.1f}M")
    return model


# ============================================================================
# 核心逻辑：生成 + PPL 过滤
# ============================================================================

@torch.no_grad()
def generate_synthetic_samples(
    model,
    tokenizer,
    seed_texts: list,
    num_samples: int,
    gen_len: int,
    prefix_min: int,
    prefix_max: int,
    temperature: float,
    top_p: float,
    seqlen: int,
    seed: int,
    device: str = "cuda",
) -> List[Tuple[torch.Tensor, float]]:
    """
    从 seed_texts 中抽 prefix，用 model 续写，返回 (token_tensor, avg_nll) 列表。

    每条样本的最终长度为 seqlen（prefix + 续写部分截断/pad 到 seqlen）。
    """
    random.seed(seed)
    torch.manual_seed(seed)

    is_hf_tokenizer = hasattr(tokenizer, "encode")

    results = []
    attempts = 0
    max_attempts = num_samples * 5  # 防止死循环

    print(f"[Synthetic] 开始生成 {num_samples} 条样本 "
          f"(prefix={prefix_min}~{prefix_max} tokens, gen_len={gen_len}, "
          f"temp={temperature}, top_p={top_p})...")

    while len(results) < num_samples and attempts < max_attempts:
        attempts += 1

        # 1) 随机选一条 C4 文本
        idx = random.randint(0, len(seed_texts) - 1)
        text = seed_texts[idx]["text"]
        tokens = tokenizer_encode(tokenizer, text)
        if len(tokens) < prefix_max + 10:
            continue  # 文本太短，跳过

        # 2) 截取短 prefix
        prefix_len = random.randint(prefix_min, prefix_max)
        # 从随机位置截取 prefix（不总是从头开始，增加多样性）
        start_pos = random.randint(0, max(0, len(tokens) - prefix_len - 1))
        prefix_ids = tokens[start_pos : start_pos + prefix_len]
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        # 3) 用模型续写
        try:
            if is_hf_tokenizer:
                outputs = model.generate(
                    prefix_tensor,
                    max_new_tokens=gen_len,
                    temperature=max(temperature, 1e-6),
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
                )
                generated_ids = outputs[0].tolist()
            else:
                # tiktoken 模式：手动 autoregressive sampling
                generated_ids = prefix_ids[:]
                input_ids = prefix_tensor
                for _ in range(gen_len):
                    logits = model(input_ids).logits[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                        probs = F.softmax(logits, dim=-1)
                        # top-p filtering
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumsum - sorted_probs > top_p
                        sorted_probs[mask] = 0
                        sorted_probs /= sorted_probs.sum()
                        next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)]
                    else:
                        next_token = logits.argmax(dim=-1)
                    generated_ids.append(next_token.item())
                    input_ids = next_token.unsqueeze(0).unsqueeze(0)
        except Exception as e:
            print(f"[Synthetic] 生成失败 (attempt {attempts}): {e}")
            continue

        # 4) 截断/pad 到 seqlen
        if len(generated_ids) > seqlen:
            generated_ids = generated_ids[:seqlen]
        elif len(generated_ids) < seqlen:
            # 太短则跳过（不做 pad，因为 pruning 需要完整序列）
            continue

        gen_tensor = torch.tensor(generated_ids, dtype=torch.long, device=device)

        # 5) 计算 avg NLL（即 log-perplexity）
        input_for_loss = gen_tensor.unsqueeze(0)  # [1, seqlen]
        outputs = model(input_for_loss, labels=input_for_loss)
        avg_nll = outputs.loss.item()  # cross-entropy loss = avg NLL

        results.append((gen_tensor.cpu().unsqueeze(0), avg_nll))  # shape [1, seqlen]

        if len(results) % 50 == 0:
            print(f"  已生成 {len(results)}/{num_samples} 条, "
                  f"avg_nll: {sum(r[1] for r in results) / len(results):.4f}")

    if len(results) < num_samples:
        print(f"[Synthetic] 警告: 只生成了 {len(results)}/{num_samples} 条有效样本")

    return results


def filter_by_ppl(
    samples: List[Tuple[torch.Tensor, float]],
    filter_percent: float,
) -> List[torch.Tensor]:
    """
    过滤掉 avg_nll 最高的 filter_percent 比例的样本。

    返回过滤后的 token tensor 列表（与原 calibration_dataset.pkl 格式一致）。
    """
    if filter_percent <= 0 or len(samples) == 0:
        return [s[0] for s in samples]

    # 按 avg_nll 排序
    sorted_samples = sorted(samples, key=lambda x: x[1])
    n_keep = max(1, int(len(sorted_samples) * (1.0 - filter_percent)))
    kept = sorted_samples[:n_keep]
    removed = sorted_samples[n_keep:]

    print(f"[Synthetic] PPL 过滤: 保留 {n_keep}/{len(samples)} 条样本 "
          f"(过滤比例={filter_percent:.1%})")
    if kept:
        print(f"  保留样本 avg_nll 范围: [{kept[0][1]:.4f}, {kept[-1][1]:.4f}]")
    if removed:
        print(f"  过滤样本 avg_nll 范围: [{removed[0][1]:.4f}, {removed[-1][1]:.4f}]")

    return [s[0] for s in kept]


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 Synthetic Calibration Data (模型续写 + PPL 过滤)"
    )
    # 模型相关
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace 模型路径 (local or hub)")
    parser.add_argument("--model_type", type=str, default="llama",
                        choices=["llama", "opt", "gpt2", "qwen"],
                        help="模型类型")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="模型推理精度")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")

    # 生成参数
    parser.add_argument("--num_samples", type=int, default=256,
                        help="生成的候选样本数（过滤前）")
    parser.add_argument("--gen_len", type=int, default=512,
                        help="每条样本的续写长度 (tokens)")
    parser.add_argument("--prefix_min", type=int, default=1,
                        help="从 C4 原文截取的 prefix 最短 token 数")
    parser.add_argument("--prefix_max", type=int, default=4,
                        help="从 C4 原文截取的 prefix 最长 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) 采样阈值")
    parser.add_argument("--seqlen", type=int, default=1024,
                        help="最终每条样本的序列长度（与原 calibration 保持一致）")

    # PPL 过滤
    parser.add_argument("--ppl_filter_percent", type=float, default=0.15,
                        help="过滤掉 PPL 最高的比例 (0.0~1.0, 默认 0.15 即 15%%)")

    # 数据源
    parser.add_argument("--c4_data_dir", type=str, default=None,
                        help="C4 数据目录 (默认: <project>/data/c4_dataset)")

    # 缓存
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="输出目录 (默认: <project>/data/synthetic_calibration)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成（忽略已有缓存）")

    args = parser.parse_args()

    # 设置默认路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.c4_data_dir is None:
        args.c4_data_dir = os.path.join(base_dir, "data", "c4_dataset")
    if args.cache_dir is None:
        args.cache_dir = os.path.join(base_dir, "data", "synthetic_calibration")

    # 计算 fingerprint
    fingerprint = compute_fingerprint(args)
    pkl_path = os.path.join(args.cache_dir, f"synthetic_calibration_{fingerprint}.pkl")
    meta_path = os.path.join(args.cache_dir, f"synthetic_calibration_{fingerprint}.meta.json")

    # 检查缓存
    if not args.force and os.path.exists(pkl_path) and os.path.exists(meta_path):
        print(f"[Synthetic] 缓存已存在，跳过生成: {pkl_path}")
        print(f"  参数: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"  样本数: {meta.get('final_num_samples', '?')}")
        return pkl_path

    os.makedirs(args.cache_dir, exist_ok=True)

    # 解析 dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 1) 加载种子文本
    seed_texts = load_c4_seed_texts(args.c4_data_dir)
    print(f"[Synthetic] 加载了 {len(seed_texts)} 条 C4 种子文本")

    # 2) 加载 tokenizer + 模型
    tokenizer = load_tokenizer(args.model_path, args.model_type)
    model = load_hf_model(args.model_path, device=args.device, dtype=torch_dtype)

    # 3) 生成样本 + 计算 NLL
    t_start = time.time()
    raw_samples = generate_synthetic_samples(
        model=model,
        tokenizer=tokenizer,
        seed_texts=seed_texts,
        num_samples=args.num_samples,
        gen_len=args.gen_len,
        prefix_min=args.prefix_min,
        prefix_max=args.prefix_max,
        temperature=args.temperature,
        top_p=args.top_p,
        seqlen=args.seqlen,
        seed=args.seed,
        device=args.device,
    )
    t_gen = time.time() - t_start
    print(f"[Synthetic] 生成耗时: {t_gen:.1f}s")

    # 4) PPL 过滤
    filtered_tensors = filter_by_ppl(raw_samples, args.ppl_filter_percent)

    # 5) 保存 pkl（格式与 prepare_calibration_data.py 一致: list of [1, seqlen] tensors）
    print(f"[Synthetic] 保存到 {pkl_path} ({len(filtered_tensors)} 条样本)")
    with open(pkl_path, "wb") as f:
        pickle.dump(filtered_tensors, f)

    # 6) 保存 meta
    meta = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "dtype": args.dtype,
        "num_samples_before_filter": len(raw_samples),
        "final_num_samples": len(filtered_tensors),
        "gen_len": args.gen_len,
        "prefix_min": args.prefix_min,
        "prefix_max": args.prefix_max,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seqlen": args.seqlen,
        "ppl_filter_percent": args.ppl_filter_percent,
        "seed": args.seed,
        "generation_time_sec": round(t_gen, 1),
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if raw_samples:
        nlls = [s[1] for s in raw_samples]
        meta["nll_stats"] = {
            "mean": round(sum(nlls) / len(nlls), 4),
            "min": round(min(nlls), 4),
            "max": round(max(nlls), 4),
        }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Synthetic] Meta 保存到 {meta_path}")

    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()

    print(f"[Synthetic] 完成！最终校准集: {len(filtered_tensors)} 条, 序列长度={args.seqlen}")
    return pkl_path


if __name__ == "__main__":
    main()
