"""
Distributed runtime helpers shared by main_llama.py and main_universal.py.

These utilities are intentionally import-cheap: no heavyweight torch subsystems
are imported eagerly. Behavior is identical to the original inline helpers in
`main_llama_legacy.py` / `main_universal_legacy.py` except that the previously
implicit `master_process` global has been promoted to an explicit argument.
"""
from __future__ import annotations

import atexit
import os
import time
from typing import Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed readiness / barrier
# ---------------------------------------------------------------------------
def _dist_is_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _safe_barrier() -> None:
    if not _dist_is_ready():
        return
    # Note: timeout parameter not supported in older PyTorch versions;
    # NCCL_TIMEOUT environment variable is used instead (set in launch scripts).
    dist.barrier()


# ---------------------------------------------------------------------------
# Multi-node debug logging
# Writes per-rank debug messages to a shared filesystem, since SSH-based
# launchers redirect stdout to per-node log files.
# ---------------------------------------------------------------------------
_DEBUG_LOG_FILE = None
_DEBUG_LOG_ENABLED = False  # Toggle True to enable file logging.


def _init_debug_log() -> None:
    """Initialize per-rank debug log file in shared filesystem."""
    global _DEBUG_LOG_FILE
    if not _DEBUG_LOG_ENABLED:
        return
    try:
        trace_dir = os.environ.get(
            "AST_TRACE_DIR",
            "/apdcephfs/pig_data/Adaptive-Sparse-Trainer/outputs/debug_logs",
        )
        os.makedirs(trace_dir, exist_ok=True)
        rank = dist.get_rank() if _dist_is_ready() else 0
        world_size = dist.get_world_size() if _dist_is_ready() else 1
        node_id = rank // 8
        local_id = rank % 8
        log_path = os.path.join(
            trace_dir, f"rank_{rank:03d}_node{node_id}_local{local_id}.log"
        )
        _DEBUG_LOG_FILE = open(log_path, "w", buffering=1)  # line-buffered
        _debug_log(
            f"=== Debug log initialized: rank={rank}, world_size={world_size}, "
            f"node={node_id}, local={local_id} ==="
        )
    except Exception as e:
        print(f"[WARNING] Failed to init debug log: {e}", flush=True)


def _debug_log(msg: str) -> None:
    """Write debug message to log file only (no stdout, to keep console clean)."""
    try:
        if _DEBUG_LOG_FILE is None:
            return
        rank = dist.get_rank() if _dist_is_ready() else 0
        node_id = rank // 8
        local_id = rank % 8
        timestamp = time.strftime("%H:%M:%S")
        full_msg = f"[{timestamp}][NODE {node_id} | RANK {rank} | LOCAL {local_id}] {msg}"
        _DEBUG_LOG_FILE.write(full_msg + "\n")
        _DEBUG_LOG_FILE.flush()
    except Exception:
        pass


def _close_debug_log() -> None:
    """Close debug log file."""
    global _DEBUG_LOG_FILE
    if _DEBUG_LOG_FILE is not None:
        try:
            _DEBUG_LOG_FILE.close()
        except Exception:
            pass
        _DEBUG_LOG_FILE = None


# Register cleanup for the debug log at interpreter shutdown.
atexit.register(_close_debug_log)


# ---------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------
def log_memory(label: str, master_process: bool = True) -> None:
    """Log current GPU memory usage.

    Parameters
    ----------
    label : str
        Tag to identify this measurement (e.g. "after STAGE 1 load").
    master_process : bool, default True
        Whether the current rank should print. In DDP/FSDP code this is
        typically ``rank == 0``.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        if master_process:
            print(
                f"[MEMORY] {label}: allocated={allocated:.2f}GB, "
                f"reserved={reserved:.2f}GB"
            )


__all__ = [
    "_dist_is_ready",
    "_safe_barrier",
    "_init_debug_log",
    "_debug_log",
    "_close_debug_log",
    "log_memory",
]
