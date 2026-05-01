"""
Data pipeline utilities for SparseForge.

This module contains:
- ``resolve_data_dtype``  : dtype auto-detection (dtype.txt / metadata.json /
                            model_type heuristic) shared by both entry points
- ``load_memmap``         : np.memmap loader with LLaMA-style graceful fallback
- ``load_train_val``      : one-shot train.bin/val.bin loader returning dtypes
- ``AsyncDataPrefetcher`` : background CPU prefetcher (unchanged from legacy)
- ``make_get_batch``      : factory that builds a closure-based ``get_batch``
                            over specific train/val memmaps (no globals)
- ``PREFETCH_ENABLED``    : ``os.environ['DISABLE_PREFETCH'] != '1'`` flag

Behaviour is intentionally identical to ``main_llama_legacy.py`` and
``main_universal_legacy.py``. The only design change is that the previously
implicit globals (``train_data``, ``val_data``, ``block_size``, ``batch_size``,
``device``, ``device_type``, ``VOCAB_SIZE_CHECK``, ``args.dataset``,
``master_process``) are now passed in explicitly.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Prefetch gate (environment-controlled)
# ---------------------------------------------------------------------------
PREFETCH_ENABLED = os.environ.get("DISABLE_PREFETCH", "0") != "1"


# ---------------------------------------------------------------------------
# dtype resolution
# ---------------------------------------------------------------------------
def resolve_data_dtype(
    data_dir: str,
    *,
    variant: str = "llama",
    model_type: Optional[str] = None,
    master_process: bool = True,
) -> np.dtype:
    """Determine the ``np.dtype`` used to memmap ``train.bin`` / ``val.bin``.

    Priority order:
    1. ``data_dir/dtype.txt`` (string ``uint32`` or ``uint16``).
    2. Variant-specific fallback:
        - ``variant='llama'``: check ``data_dir/metadata.json``;
          default uint16 if nothing else.
        - ``variant='universal'``: use ``model_type`` heuristic
          (``qwen`` / ``deepseek_moe`` → uint32; else uint16).
    """
    dtype_file = os.path.join(data_dir, "dtype.txt")
    if os.path.exists(dtype_file):
        with open(dtype_file, "r") as f:
            dtype_name = f.read().strip()
        data_dtype = np.uint32 if dtype_name == "uint32" else np.uint16
        if master_process:
            print(f"[DATA] Using dtype={dtype_name} from {dtype_file}")
        return data_dtype

    if variant == "llama":
        metadata_file = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_file):
            import json as _json
            with open(metadata_file, "r") as f:
                _meta = _json.load(f)
            data_dtype = np.uint32 if _meta.get("dtype") == "uint32" else np.uint16
            if master_process:
                print(f"[DATA] Using dtype={data_dtype.__name__} from metadata.json")
            return data_dtype
        data_dtype = np.uint16
        if master_process:
            print("[DATA] No dtype.txt or metadata.json found, defaulting to uint16")
        return data_dtype

    # variant == 'universal'
    if model_type in ("qwen", "deepseek_moe"):
        data_dtype = np.uint32  # large-vocab models
    else:
        data_dtype = np.uint16  # traditional models
    if master_process:
        print(
            f"[DATA] Auto-detected dtype={data_dtype.__name__} "
            f"for model_type={model_type}"
        )
    return data_dtype


# ---------------------------------------------------------------------------
# Memmap loading
# ---------------------------------------------------------------------------
def load_memmap(path: str, default_dtype: np.dtype, master_process: bool = True):
    """Load a ``.bin`` memmap, falling back to uint16 if the file size is
    not an integer multiple of ``default_dtype``'s itemsize.

    Returns ``(memmap, effective_dtype)``.
    """
    file_size = os.path.getsize(path)
    dtype_size = np.dtype(default_dtype).itemsize
    if file_size % dtype_size != 0:
        fallback_dtype = np.uint16
        if master_process:
            print(
                f"[DATA] WARNING: {os.path.basename(path)} size ({file_size} bytes) "
                f"is not a multiple of {default_dtype.__name__} ({dtype_size} bytes), "
                f"falling back to {fallback_dtype.__name__}"
            )
        return np.memmap(path, dtype=fallback_dtype, mode="r"), fallback_dtype
    return np.memmap(path, dtype=default_dtype, mode="r"), default_dtype


def load_train_val(
    data_dir: str,
    default_dtype: np.dtype,
    *,
    variant: str = "llama",
    master_process: bool = True,
) -> Tuple[np.memmap, np.memmap, np.dtype, np.dtype]:
    """Load ``train.bin`` and ``val.bin`` from ``data_dir``.

    - ``variant='llama'`` uses :func:`load_memmap` (graceful uint16 fallback).
    - ``variant='universal'`` uses a plain ``np.memmap`` (matches legacy).
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if variant == "llama":
        train_data, train_dtype = load_memmap(train_path, default_dtype, master_process)
        val_data, val_dtype = load_memmap(val_path, default_dtype, master_process)
        if master_process:
            print(
                f"[DATA] Loaded train.bin ({len(train_data):,} tokens, "
                f"dtype={train_dtype.__name__}) and val.bin "
                f"({len(val_data):,} tokens, dtype={val_dtype.__name__})"
            )
        return train_data, val_data, train_dtype, val_dtype
    # variant == 'universal' -> plain memmap
    train_data = np.memmap(train_path, dtype=default_dtype, mode="r")
    val_data = np.memmap(val_path, dtype=default_dtype, mode="r")
    return train_data, val_data, default_dtype, default_dtype


# ---------------------------------------------------------------------------
# Async prefetcher (unchanged from legacy)
# ---------------------------------------------------------------------------
class AsyncDataPrefetcher:
    """Background CPU thread that keeps a bounded queue of next-batch tensors.

    Identical behaviour to the original inlined class in the legacy scripts;
    no design changes.
    """

    def __init__(self, data, block_size, batch_size, device, prefetch_count=2):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None

    def _prefetch_worker(self):
        """Background thread: continuously prefetches batches."""
        while not self.stop_event.is_set():
            try:
                # Generate random indices
                ix = torch.randint(
                    len(self.data) - self.block_size - 1, (self.batch_size,)
                )
                # Read from memmap (CPU-side)
                x = torch.stack(
                    [torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
                     for i in ix]
                )
                y = torch.stack(
                    [torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
                     for i in ix]
                )
                # Pin memory to accelerate the host->device copy
                x = x.pin_memory()
                y = y.pin_memory()
                # Push to queue (blocks if full)
                self.queue.put((x, y), timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Prefetcher] Error: {e}")
                break

    def start(self):
        """Start the prefetch thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def get_batch(self):
        """Pop one prefetched batch and move it to ``self.device``.

        Returns ``(x, y)`` or ``None`` if the queue is empty
        (caller should fall back to synchronous loading).
        """
        try:
            x, y = self.queue.get(timeout=10.0)
            # Non-blocking H2D copy
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            return x, y
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# get_batch factory
# ---------------------------------------------------------------------------
def make_get_batch(
    *,
    train_data,
    val_data,
    block_size: int,
    batch_size: int,
    device,
    device_type: str,
    vocab_size_check: Optional[int] = None,
    vocab_size_check_getter: Optional[Callable[[], Optional[int]]] = None,
    dataset_name: str = "",
    student_model_name: str = "",
    prefetcher_getter: Optional[Callable[[], Optional["AsyncDataPrefetcher"]]] = None,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """Build a closure-based ``get_batch(split)`` replacement.

    The returned callable replicates the legacy behaviour exactly:
    - For ``split='train'`` it attempts an asynchronous path via
      ``prefetcher_getter()`` first (if non-None and the global prefetch flag
      is enabled), falling back to a synchronous memmap read on an empty
      queue.
    - For ``split='val'`` it always reads synchronously.
    - A vocab-size guardrail raises ``ValueError`` if any token exceeds
      the current vocab-size value (defaults match the legacy diagnostic
      message).

    Parameters
    ----------
    vocab_size_check : optional int
        Static vocab-size bound.  Evaluated *eagerly* when the closure is
        built; prefer ``vocab_size_check_getter`` if the bound is known
        only later in the program.
    vocab_size_check_getter : callable returning Optional[int]
        Indirection so the caller can install the vocab-size bound *after*
        this closure has been built (matches the legacy ``global
        VOCAB_SIZE_CHECK`` late-binding pattern).  When both this and
        ``vocab_size_check`` are provided the getter wins.
    prefetcher_getter : callable returning Optional[AsyncDataPrefetcher]
        Indirection so callers can install the prefetcher *after* this
        closure has been built (matches the legacy ``global train_prefetcher``
        pattern).
    """

    def _current_vocab_check() -> Optional[int]:
        if vocab_size_check_getter is not None:
            return vocab_size_check_getter()
        return vocab_size_check

    def get_batch(split: str):
        # Async prefetch path (training only)
        if (
            split == "train"
            and PREFETCH_ENABLED
            and prefetcher_getter is not None
        ):
            pref = prefetcher_getter()
            if pref is not None:
                result = pref.get_batch()
                if result is not None:
                    x, y = result
                    vsc = _current_vocab_check()
                    if vsc is not None:
                        max_id = int(x.max().item())
                        if max_id >= int(vsc):
                            raise ValueError(
                                f"Dataset token id out of range: max_id={max_id} "
                                f">= vocab_size={int(vsc)}. "
                            )
                    return x, y
                # fall through to synchronous path

        # Synchronous path (val, or training when prefetch is cold)
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
        )

        # Guardrail: token ids >= vocab_size trigger device-side CUDA asserts.
        # Usually caused by a mismatch between the tokenizer used to prepare
        # the dataset and the one expected by the model.
        vsc = _current_vocab_check()
        if vsc is not None:
            max_id = int(x.max().item())
            if max_id >= int(vsc):
                raise ValueError(
                    f"Dataset token id out of range: max_id={max_id} "
                    f">= vocab_size={int(vsc)}. "
                    f"Your dataset '{dataset_name}' is likely tokenized for a "
                    f"different tokenizer. Re-tokenize with the LLaMA tokenizer "
                    f"or switch to a LLaMA-tokenized dataset. If you want to "
                    f"generate a compatible memmap from data/c4_dataset shards, "
                    f"run: python data/c4_llama/prepare.py "
                    f"--tokenizer {student_model_name}"
                )

        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch


__all__ = [
    "PREFETCH_ENABLED",
    "resolve_data_dtype",
    "load_memmap",
    "load_train_val",
    "AsyncDataPrefetcher",
    "make_get_batch",
]
