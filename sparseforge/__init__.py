"""SparseForge: Semi-structured sparse training for large language models.

Official code repository for the SparseForge paper.

Public submodules (import explicitly, not re-exported here to avoid heavy
side-effects at import time):
    - distributed       : multi-node barrier / debug logging / memory helpers
    - cli               : argparse builders for main_llama.py / main_universal.py
    - data_pipeline     : AsyncDataPrefetcher / get_batch / _load_bin
    - optim_utils       : optimizer construction, LR schedule, FSDP optim-sd alignment
    - eval_utils        : estimate_loss
    - state             : TrainState dataclass (reference; main_*.py still uses
                          module-level state, see state.py docstring)
    - model_builders.llama     : FSDP wrap-policy helper + lazy LlamaSparse re-export
    - model_builders.universal : FSDP wrap-policy helper + lazy model_factory re-exports
    - checkpoint        : ckpt discovery / metadata helpers (reference; actual
                          save/load lives in main_*.py, see checkpoint.py docstring)
    - training_loop     : TrainingStage enum + detect_stage() (reference; the
                          actual loop remains in main_*.py)

Note: structured channel-pruning code lives in the top-level ``channel_pruning/``
package of this repository, not under ``sparseforge``.
"""

__version__ = "0.1.0"
