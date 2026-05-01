"""SparseForge: Semi-structured sparse training for large language models.

Official code repository for the SparseForge paper.

Public submodules (import explicitly, not re-exported here to avoid heavy
side-effects at import time):
    - distributed       : multi-node barrier / debug logging / memory helpers
    - cli               : argparse builders for main_llama.py / main_universal.py
    - data_pipeline     : AsyncDataPrefetcher / get_batch / _load_bin
    - optim_utils       : optimizer construction, LR schedule, FSDP optim-sd alignment
    - eval_utils        : estimate_loss
    - channel_pruning_hooks : structured channel-pruning helpers (universal only)
    - state             : TrainState dataclass that threads state across stages
    - model_builders.llama     : build_student / build_teacher / wrap_fsdp for LLaMA
    - model_builders.universal : same, for the universal (multi-arch) entry point
    - checkpoint        : resume / save logic
    - training_loop     : the main `while True` training loop
"""

__version__ = "0.1.0"
