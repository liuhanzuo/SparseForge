# Structured Pruning for LLM: FFN Channel & Attention Head

本模块实现了基于 **SlimLLM** importance 计算方法结合 **HASAST** mask 推进机制的结构化剪枝，支持：

1. **FFN Channel Pruning**: 减少 MLP 中间维度 (intermediate_size)
2. **Attention Head Pruning**: 减少注意力头数 (num_attention_heads)，支持 GQA

---

## 核心思想

### FFN Channel Pruning

不同于 weight-level 剪枝，Channel-level 剪枝直接减少 FFN 的中间维度：

```
原始 FFN (SwiGLU):
  up_proj:   [hidden_size, intermediate_size]
  gate_proj: [hidden_size, intermediate_size]  
  down_proj: [intermediate_size, hidden_size]

剪枝后 (保留 75% channels):
  up_proj:   [hidden_size, intermediate_size * 0.75]
  gate_proj: [hidden_size, intermediate_size * 0.75]
  down_proj: [intermediate_size * 0.75, hidden_size]
```

### Attention Head Pruning (GQA-aware)

对于 GQA (Grouped Query Attention) 模型，mask 对象是 **query heads**：

```
原始 Attention:
  q_proj: [hidden_size, num_heads * head_dim]
  k_proj: [hidden_size, num_kv_heads * head_dim]  (GQA: num_kv_heads < num_heads)
  v_proj: [hidden_size, num_kv_heads * head_dim]
  o_proj: [num_heads * head_dim, hidden_size]

剪枝后 (保留 75% query heads):
  q_proj: [hidden_size, num_heads * 0.75 * head_dim]
  k_proj: [hidden_size, num_kv_kept * head_dim]  (KV heads 自动裁剪)
  v_proj: [hidden_size, num_kv_kept * head_dim]
  o_proj: [num_heads * 0.75 * head_dim, hidden_size]
```

**优势**：导出后模型**真正变小**，推理速度**真正提升**，无需特殊稀疏硬件支持。

---

## 模块结构

```
channel_pruning/
├── __init__.py                # 包初始化
├── config.py                  # 配置定义
├── channel_groups.py          # MLP channel group 提取
├── channel_score.py           # FFN channel importance 计算
├── channel_mask.py            # FFN mask 状态管理
├── attention_groups.py        # Attention head group 提取 (GQA-aware)
├── attention_score.py         # Head importance 计算 (GQA 分摊)
├── attention_mask.py          # Head mask 状态管理
├── export_pruned.py           # 导出真实缩维的模型
├── train_channel_pruning.py   # FFN 剪枝训练脚本
└── README.md                  # 本文档
```

---

## 快速开始

### 1. 准备数据

确保你已经有 tokenized 的训练数据：
```bash
ls data/c4_qwen/
# train.bin  val.bin
```

### 2. 运行训练

**单 GPU - FFN only：**
```bash
python -m channel_pruning.train_channel_pruning \
    --model_name Qwen/Qwen3-1.7B \
    --ffn_keep_ratio 0.75 \
    --dataset c4_qwen \
    --max_iters 10000
```

**多 GPU (DDP)：**
```bash
torchrun --standalone --nproc_per_node=8 \
    -m channel_pruning.train_channel_pruning \
    --model_name Qwen/Qwen3-1.7B \
    --ffn_keep_ratio 0.75 \
    --ddp
```

**使用启动脚本：**
```bash
bash scripts/train_channel_pruning.sh \
    --model Qwen/Qwen3-1.7B \
    --keep_ratio 0.75 \
    --mode ddp \
    --wandb
```

---

## 关键参数

### FFN Pruning

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ffn_keep_ratio` | 保留的 FFN channel 比例 | 0.75 |
| `--score_alpha/beta/gamma` | up/gate/down 分数权重 | 1.0 |

### Attention Pruning

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prune_attention` | 是否剪枝 attention | False |
| `--attention_keep_ratio` | 保留的 query head 比例 | 0.75 |
| `--attn_score_alpha/beta/gamma/delta` | q/k/v/o 分数权重 | 1.0 |

### Mask Dynamics

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mask_update_period` | Mask 更新周期 (步数) | 50 |
| `--mask_lr` | Mask EMA 学习率 | 0.1 |
| `--sparsity_warmup_steps` | 稀疏度预热步数 | 500 |
| `--hardening_start_step` | 开始硬化的步数 | 2000 |
| `--hardening_duration` | 硬化持续步数 | 5000 |

---

## 技术细节

### FFN Channel Importance 计算

```python
# Per-weight importance (Hessian-OBD)
s_w = (H + eps) * W^2

# Aggregate to channel j:
s_up[j]   = sum(s_w for w in up_proj[j, :])
s_gate[j] = sum(s_w for w in gate_proj[j, :])
s_down[j] = sum(s_w for w in down_proj[:, j])

# Channel score
channel_score[j] = α * s_up[j] + β * s_gate[j] + γ * s_down[j]
```

### Attention Head Importance 计算 (GQA-aware)

```python
# Per-weight importance (same as FFN)
s_w = (H + eps) * W^2

# Aggregate to head h:
# Q and O use query head slices: I_h = [h*d, (h+1)*d)
s_q[h] = sum(s_w for w in q_proj[I_h, :])
s_o[h] = sum(s_w for w in o_proj[:, I_h])

# K and V use KV head slices with GQA distribution:
# KV head index: h_kv = h // group_size
# I_kv = [h_kv*d, (h_kv+1)*d)
s_k[h] = (1/g) * sum(s_w for w in k_proj[I_kv, :])  # distributed
s_v[h] = (1/g) * sum(s_w for w in v_proj[I_kv, :])  # distributed

# Head score
head_score[h] = α * s_q[h] + β * s_k[h] + γ * s_v[h] + δ * s_o[h]
```

**GQA 分摊原理**：每个 KV head 被 `group_size` 个 query heads 共享，因此 KV 分数被平均分摊到这些 query heads，避免重复计算带来的偏置。

### Mask 更新机制

沿用 HASAST 的 soft-to-hard 机制：

1. **Soft Gating**: `G[j] = sigmoid((score[j] - τ) / T)`
2. **EMA Update**: `mask = (1-lr) * mask + lr * G`
3. **Temperature Decay**: `T = T * decay`
4. **Hardening**: 逐步从 soft mask 过渡到 binary mask

### KV Head 自动裁剪

对于 GQA 模型，KV head 的裁剪基于以下规则：
- 一个 KV head 被保留，当且仅当它对应的任一 query head 被保留
- 当一个 KV head 的所有 query heads 都被剪掉时，该 KV head 也被剪掉

```python
kv_mask[kv_idx] = max(query_mask[h] for h in query_heads_for_kv[kv_idx])
```

---

## 模型导出

训练结束后，将 masked model 转换为真正缩维的模型：

### FFN Export
```python
keep_indices = topk(channel_mask, k=keep_k)
new_up_proj.weight = up_proj.weight[keep_indices, :]
new_gate_proj.weight = gate_proj.weight[keep_indices, :]
new_down_proj.weight = down_proj.weight[:, keep_indices]
```

### Attention Export
```python
# Query head indices -> weight indices
q_keep_rows = [h * head_dim : (h+1) * head_dim for h in kept_heads]
kv_keep_rows = [h * head_dim : (h+1) * head_dim for h in kept_kv_heads]

new_q_proj.weight = q_proj.weight[q_keep_rows, :]
new_k_proj.weight = k_proj.weight[kv_keep_rows, :]
new_v_proj.weight = v_proj.weight[kv_keep_rows, :]
new_o_proj.weight = o_proj.weight[:, q_keep_rows]
```

---

## Qwen3-1.7B 参数分布参考

| 组件 | 每层参数 | 28层总量 | 占比 |
|------|----------|----------|------|
| Attention | ~9.5M | ~266M | ~15.8% |
| MLP | ~50.3M | ~1.41B | ~84.2% |

> **结论**：对 Qwen3，MLP 剪枝效果更显著；Attention 剪枝适合追求更激进的压缩。

---

## 推荐配置

### Qwen3-1.7B → ~1.3B (FFN 75%)
```bash
python -m channel_pruning.train_channel_pruning \
    --model_name Qwen/Qwen3-1.7B \
    --ffn_keep_ratio 0.75 \
    --max_iters 10000
```

### LLaMA-2-7B → ~5.3B (FFN 70%)
```bash
python -m channel_pruning.train_channel_pruning \
    --model_name NousResearch/Llama-2-7b-hf \
    --model_type llama \
    --ffn_keep_ratio 0.70 \
    --max_iters 20000
```

### 激进压缩 (FFN 75% + Attention 75%)
```bash
python -m channel_pruning.train_channel_pruning \
    --model_name Qwen/Qwen3-1.7B \
    --ffn_keep_ratio 0.75 \
    --prune_attention \
    --attention_keep_ratio 0.75 \
    --max_iters 15000
```

---

## 输出文件

```
outputs/channel_pruning/
├── config.json               # 训练配置
├── ckpt_*.pt                 # 训练检查点
├── pruned_model/             # 导出的剪枝模型
│   ├── model.safetensors
│   ├── config.json
│   ├── pruning_info.json           # FFN 剪枝信息
│   └── attention_pruning_info.json # Attention 剪枝信息
└── logs/                     # 训练日志
```

---

## API 使用示例

### 单独使用 Attention Head Score 计算

```python
from channel_pruning import (
    AttentionScoreComputer,
    AttentionMaskState,
    get_attention_config,
    ChannelPruningConfig
)

# 配置
config = ChannelPruningConfig(
    model_name="Qwen/Qwen3-1.7B",
    model_type="qwen",
    attention_keep_ratio=0.75,
)

# 计算 head importance
score_computer = AttentionScoreComputer(model, config)
layer_scores = score_computer.compute_all_scores()

for ls in layer_scores:
    print(f"Layer {ls.layer_idx}: head scores = {ls.scores}")
    print(f"  Q contribution: {ls.q_scores}")
    print(f"  K contribution (GQA distributed): {ls.k_scores}")
    print(f"  V contribution (GQA distributed): {ls.v_scores}")
    print(f"  O contribution: {ls.o_scores}")
```

### GQA 信息查看

```python
from channel_pruning import get_attention_config, AttentionHeadGroupManager

attn_config = get_attention_config(model, "qwen")
print(f"num_heads: {attn_config.num_heads}")
print(f"num_kv_heads: {attn_config.num_kv_heads}")
print(f"group_size: {attn_config.group_size}")
print(f"is_gqa: {attn_config.is_gqa}")

# 查看 query head -> KV head 映射
for h in range(attn_config.num_heads):
    kv_h = attn_config.get_kv_head_for_query(h)
    print(f"Query head {h} -> KV head {kv_h}")
```

---

## 与其他方法对比

| 特性 | HASAST (Weight) | Channel Pruning | Head Pruning |
|------|-----------------|-----------------|--------------|
| 剪枝粒度 | 单个权重 / 2:4 | FFN channel | Attention head |
| 导出后大小 | 不变 (sparse) | 真正变小 | 真正变小 |
| 推理加速 | 需要稀疏硬件 | 任意硬件 | 任意硬件 |
| 精度损失 | 较小 | 中等 | 中等 |
| GQA 支持 | N/A | N/A | ✅ 完整支持 |

---

## TODO

- [x] Attention head pruning with GQA support
- [x] KV head automatic pruning
- [ ] Per-layer adaptive keep ratio (based on layer importance)
- [ ] Calibration-based Hessian estimation
- [ ] Integration with lm-evaluation-harness
- [ ] Unified FFN + Attention training script
