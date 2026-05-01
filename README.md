# SparseForge

> **Official code for _"SparseForge: Efficient LLM Semi-Structured Pruning via Hessian-Guided Soft-Mask Tempering"_**
> Paper: [arXiv (coming soon)]() &nbsp;|&nbsp; Venue: NeurIPS 2026 (under review)

**SparseForge** is a post-training framework for semi-structured LLM sparsification that
optimizes the sparsity mask directly — rather than scaling up retraining tokens — to
achieve efficient sparse recovery. It combines **Hessian-aware importance estimation**
with **progressive tempering** of soft masks into hardware-executable 2:4 patterns.

**Key result:** On LLaMA-2-7B under 2:4 sparsity, SparseForge reaches **57.27%** average
zero-shot accuracy with only **5B** retraining tokens — surpassing the dense model
(56.43%) and approaching a 40B-token prior SOTA (57.52%) with ~8× fewer tokens.

### Highlights

- **Soft-mask optimization** — treats the sparsity mask as a continuous, optimizable variable; avoids premature hard decisions that plague one-shot pruning.
- **Hessian-guided scoring** — uses stochastic Hutchinson estimation to capture deletion sensitivity under grouped (2:4) competition.
- **Progressive tempering** — gradually shapes soft masks into deployable binary patterns, closing the soft-to-hard gap without abrupt accuracy drops.
- **SLoRB** (Sparse Low-Rank Bypass) — lightweight correction modules for residual capacity.
- **Multi-architecture** — supports LLaMA / OPT / GPT-2 / Qwen / Mistral / DeepSeek-MoE / Hunyuan.
- **Block-sparse-16** Triton kernels in addition to the classic 2:4 N:M pattern.
- Optional **FSDP + DeepSpeed** sharding for 7B+ scale training.

---

## Repository Layout

```
SparseForge/
├── sparseforge/             # Modular Python package (clean API)
│   ├── __init__.py
│   ├── cli.py              # Argparse builders
│   ├── data_pipeline.py    # AsyncDataPrefetcher / get_batch
│   ├── distributed.py      # Multi-node barrier / debug logging
│   ├── optim_utils.py      # Optimizer construction, LR schedule
│   ├── eval_utils.py       # estimate_loss
│   ├── state.py            # TrainState dataclass
│   ├── checkpoint.py       # Resume / save logic
│   ├── training_loop.py    # Main training loop interface
│   └── model_builders/     # Per-architecture model builders
│       ├── llama.py
│       └── universal.py
│
├── legacy/                  # Monolithic single-file entry points (see below)
│   ├── main_llama.py       # LLaMA-2-7B training (all-in-one)
│   ├── main_universal.py   # Multi-architecture training (all-in-one)
│   ├── eval_wiki_ppl.py    # Standalone evaluation script
│   ├── sparse_modeling.py  # SparseLinear / Distill_Model
│   ├── utils.py            # Mask ops, Hessian, penalties, calibration
│   ├── model_factory.py    # Auto-dispatch model creation
│   ├── model_*.py          # Per-architecture model adapters
│   ├── adamw.py            # Mask-aware AdamW optimizer
│   └── triton_block_sparse.py  # Triton block-sparse kernels
│
├── channel_pruning/         # Structured channel-pruning utilities
├── data/                    # Dataset download & preparation scripts
├── configs/                 # DeepSpeed config + hostfile template
├── scripts/                 # Cluster launcher + data preparation helpers
├── assets/                  # Figures
│
├── train_llama.sh           # Launcher → legacy/main_llama.py
├── train_universal.sh       # Launcher → legacy/main_universal.py
├── eval_wiki_ppl.sh         # Launcher → legacy/eval_wiki_ppl.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<YOUR_ORG>/SparseForge.git
cd SparseForge

# 1. PyTorch (adjust the CUDA version to your system)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Remaining dependencies
pip install -r requirements.txt
```

Tested with PyTorch 2.1+, CUDA 12.1, and 8×H800 / 8×A100 GPUs.

---

## Alternative: Legacy Monolithic Scripts

The `legacy/` directory contains the **original all-in-one training scripts** that were
used to produce all paper results. Each script embeds the full training loop, model
construction, mask scheduling, and evaluation inline — no external package dependencies
beyond PyTorch and HuggingFace.

This is the **battle-tested, numerically verified** code path. If you just want to
reproduce the paper numbers without touching the modular `sparseforge/` package, use
the shell launchers directly — they already point to `legacy/`:

```bash
bash train_llama.sh 1 0          # single-node LLaMA training
bash train_universal.sh 1 0      # single-node universal training
bash eval_wiki_ppl.sh 1 0        # single-node evaluation
```

See [`legacy/README.md`](legacy/README.md) for details.

---

## Data Preparation

### C4 (training corpus)

```bash
bash data/download_c4.sh                 # raw shards
# Per-tokenizer pre-tokenisation (one of the following, depending on the model):
python data/prepare_instruct.py          # instruction-style preprocessing
# or see scripts/prepare_mixed_c4_based.py for per-tokenizer binarization
```

Each tokenizer produces an isolated binary directory (e.g. `data/c4_llama/`,
`data/c4_qwen/`, ...) which is referenced by `--dataset c4_${MODEL_TYPE}`.

### WikiText-2 (evaluation)

```bash
python data/download_wikitext.py
```

### Pre-trained models

```bash
python data/download_hf_model.py --repo NousResearch/Llama-2-7b-hf --out models/Llama--Llama2-7b
```

---

## Training

### Launch Scripts Overview

| Script | Depends on `cluster_launcher.sh` | Mode | Description |
| --- | --- | --- | --- |
| `train_llama.sh` | Yes | Multi-node (Controller) or single-node | LLaMA-2-7B sparse training |
| `train_universal.sh` | Yes | Multi-node (Controller) or single-node | Universal multi-model sparse training |
| `eval_wiki_ppl.sh` | Yes | Remote node evaluation | WikiText-2 PPL + lm_eval benchmarks |
| `scripts/train_channel_pruning.sh` | No | Single-node DDP only | Structured channel pruning |
| `scripts/cluster_launcher.sh` | — | Library (sourced) | Node pool management, SSH orchestration |

**Controller mode** (recommended): `bash train_llama.sh <NNODES> <IDX1> ... <IDXN>` — selects
nodes from the pool defined in `cluster_launcher.sh` and auto-launches via SSH.

**Legacy mode**: `bash train_llama.sh <MASTER_IP> <NODE_RANK> <NNODES>` — run manually on
each node.

### LLaMA-2-7B (block-sparse-16, Hutchinson Hessian)

```bash
bash train_llama.sh
```

Single-node (default) uses `deepspeed --num_gpus 8`. For multi-node:

```bash
# Edit configs/hosts.txt with your <MASTER_IP> / <WORKER_IP>.
NNODES=2 NODE_RANK=0 MASTER_ADDR=<MASTER_IP> bash train_llama.sh
```

Use `USE_FSDP_FULLY_SHARDED=1 bash train_llama.sh` to train with PyTorch FSDP instead.

### Universal trainer (OPT / Qwen / Mistral / DeepSeek-MoE / Hunyuan / GPT-2)

Open `train_universal.sh` and uncomment the desired model block, e.g.

```bash
STUDENT_MODEL="models/Qwen--Qwen3-1.7b"
TEACHER_MODEL="models/Qwen--Qwen3-1.7b"
MODEL_TYPE="qwen"
MASK_TYPE="block_sparse16"    # or "unstructured" / "structured" (2:4)
```

Then:

```bash
bash train_universal.sh
```

### Key arguments (shared by both entry points)

| Argument | Meaning |
| --- | --- |
| `--mask_type` | `unstructured` / `structured` (2:4) / `block_sparse16` |
| `--hard_mask_type` | Pattern enforced after the hardening phase |
| `--mask_metric` | `hessian_ratio` / `hessian_obd` / `magnitude` / `wanda` |
| `--sparsity_ratio` | Target sparsity (e.g. `0.5`) |
| `--enable_hutchinson` | Use stochastic Hutchinson Hessian for mask scoring |
| `--mask_update_period_before/after` | Mask refresh period across the hardening switch |
| `--mask_hardening_start/duration` | Iterations for continuous→hard mask transition |
| `--SLoRB`, `--SLoRB_k`, `--SLoRB_init_type` | Sparse Low-Rank Bypass module |
| `--distill_model`, `--hardness_task/kldiv/squarehead` | Distillation loss weights |
| `--use_fsdp`, `--fsdp_mode` | `hybrid_shard` / `full_shard` / `none` |

See `legacy/main_universal.py --help` for the full list.

---

## Evaluation

```bash
# Evaluate a trained checkpoint on WikiText-2 and zero-shot benchmarks.
CKPT_PATH=outputs/.../model.pt \
MODEL_PATH=models/Qwen--Qwen3-1.7b \
bash eval_wiki_ppl.sh

# Inspect checkpoint sparsity
python legacy/check_sparsity.py --ckpt outputs/.../model.pt
```

`eval_wiki_ppl.sh` will optionally run `lm_eval` on `boolq, rte, hellaswag, winogrande,
arc_easy, arc_challenge, openbookqa` with `RUN_LM_EVAL=true`.

---

## Reproducing the Paper

| Table | Entry point | Config |
| --- | --- | --- |
| LLaMA-2-7B 2:4 / block16 | `legacy/main_llama.py` via `train_llama.sh` | Defaults in `train_llama.sh` |
| Qwen / Mistral / OPT / DeepSeek-MoE | `legacy/main_universal.py` via `train_universal.sh` | Uncomment the relevant `STUDENT_MODEL` block |
| Zero-shot benchmarks | `eval_wiki_ppl.sh` with `RUN_LM_EVAL=true` | — |

Default hyper-parameters in the provided `.sh` scripts match the paper setup.

---

## Results at a Glance

### Cross-Model Summary (2:4 sparsity, mean zero-shot accuracy %)

| Model | Dense | SparseForge | Δ |
| --- | --- | --- | --- |
| GPT2-Medium | 40.97 | 40.31 | -0.66 |
| GPT2-Large | 42.76 | 42.10 | -0.66 |
| GPT2-XL | 45.49 | 44.34 | -1.15 |
| OPT-2.7B | 47.76 | 46.67 | -1.09 |
| Qwen3-1.7B | 56.51 | 53.33 | -3.18 |
| Qwen3-8B | 65.73 | 63.31 | -2.42 |
| Qwen3-14B | 68.36 | 65.44 | -2.93 |
| DeepSeek-MoE-16B | 59.54 | 58.57 | -0.97 |

### LLaMA-2-7B Comparison (2:4 sparsity)

| Method | Tokens | Mean Acc. | Wiki PPL |
| --- | --- | --- | --- |
| Dense | 2T | 56.43% | 5.12 |
| Wanda | 0 | 45.98% | 11.29 |
| SparseGPT | 0 | 47.16% | 10.42 |
| MaskLLM | 2B | 52.09% | 6.72 |
| CAST | 7.5B | 55.91% | 5.58 |
| **SparseForge** | **1.25B** | **55.96%** | 6.24 |
| **SparseForge** | **5B** | **57.27%** | 6.09 |
| CAST† (40B tokens) | 40B | 57.52% | 5.21 |

---

## Citation

If you find SparseForge useful, please cite:

```bibtex
@article{liu2026sparseforge,
  title   = {SparseForge: Efficient LLM Semi-Structured Pruning via Hessian-Guided Soft-Mask Tempering},
  author  = {Liu, Hanzuo and Lin, Chaofan and Sun, Weixuan and Wang, Yulong and Rayying and Key and Gao, Mingyu},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## Acknowledgements

SparseForge builds upon [nanoGPT](https://github.com/karpathy/nanoGPT),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
HuggingFace `transformers` / `datasets`, PyTorch FSDP, DeepSpeed and Triton.
We thank the authors and maintainers of these projects.

## License

Released under the [Apache License 2.0](LICENSE).
