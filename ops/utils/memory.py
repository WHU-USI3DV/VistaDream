"""Memory cleanup helpers for long-running interactive processes."""

import ctypes
import gc

import torch


def release_memory():
    """Release unreachable Python objects and cached CPU/CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    # CPython/glibc may retain large freed model buffers in the process heap.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
