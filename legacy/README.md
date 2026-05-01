# Legacy: Monolithic Single-File Training Scripts

This directory contains the **original monolithic entry points** for SparseForge training.
Each script is self-contained (aside from shared utility modules in this same directory)
and can be launched directly via the shell scripts at the project root.

## Contents

| File | Description |
| --- | --- |
| `main_llama.py` | LLaMA-family sparse training (single-file, all logic inline) |
| `main_universal.py` | Multi-architecture sparse training (LLaMA/OPT/GPT-2/Qwen/Mistral/DeepSeek) |
| `eval_wiki_ppl.py` | Standalone evaluation: WikiText-2 PPL + lm_eval harness |
| `sparse_modeling.py` | Core sparse layer (`SparseLinear`) and distillation wrapper |
| `utils.py` | Training utilities (EMA, mask ops, penalties, calibration, eval) |
| `model_factory.py` | Unified model creation interface (auto-detect architecture) |
| `model.py` / `model_*.py` | Per-architecture model definitions |
| `adamw.py` | Custom AdamW optimizer with mask-aware weight decay |
| `triton_block_sparse.py` | Triton kernels for block-sparse matmul |
| `check_sparsity.py` | Checkpoint sparsity analysis tool |
| `evaluate_benchmarks.py` | Batch benchmark evaluation driver |

## How to Launch

These scripts are invoked by the shell launchers at project root:

```bash
# From project root:
bash train_llama.sh 2 0 1        # 2-node LLaMA training
bash train_universal.sh 1 0      # single-node universal training
bash eval_wiki_ppl.sh 1 0        # single-node evaluation
```

The shell scripts handle `cd legacy/` and `torchrun` orchestration automatically.

## Why "Legacy"?

The modular `sparseforge/` package (at project root) provides the same functionality
decomposed into clean, testable submodules. These monolithic scripts remain as the
**battle-tested, numerically verified** training implementation and serve as the
reference for reproducing paper results.
