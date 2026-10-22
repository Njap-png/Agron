"""Shared helpers for the Agron neural-network zoo.

Everything here is deliberately dependency-light (torch + numpy only) so the
package can be imported from the Gradio app, the training scripts and the
test-suite without pulling in the full Hugging Face stack.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python / numpy / torch RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Model introspection
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def format_parameters(n: int) -> str:
    """Human readable parameter count, e.g. ``50.3M``."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def model_device(model: nn.Module) -> torch.device:
    """Best guess of the device a module lives on."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def dtype_of(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Deterministic weight init used by every Agron architecture.

    Linear layers get a truncated-normal style normal init, embeddings get a
    normal init, and any residual-projection weight is zero-initialised so the
    network starts as an identity map on the residual stream (a well known
    trick that stabilises deep transformers).
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.zero_()


# --------------------------------------------------------------------------- #
# Small maths helpers
# --------------------------------------------------------------------------- #
def cosine_schedule(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0) -> float:
    """Linear warmup followed by cosine decay to ``min_ratio``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_schedule(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0) -> float:
    """Linear warmup followed by linear decay to ``min_ratio``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return max(min_ratio, 1.0 - progress * (1.0 - min_ratio))


def warmup_constant(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0) -> float:
    """Warmup then hold at 1.0 (times ``min_ratio`` afterwards)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    return 1.0


def inverse_sqrt_schedule(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0) -> float:
    """Noam / Transformer-XL style ``1/sqrt(step)`` decay."""
    step = max(step, 1)
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    return min(1.0, math.sqrt(warmup / step) if warmup else 1.0)


def polynomial_schedule(step: int, total: int, warmup: int = 0, power: float = 1.0,
                        min_ratio: float = 0.0) -> float:
    """Polynomial (e.g. ``power=0.5``) decay."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return max(min_ratio, (1.0 - progress) ** power)


def cosine_restart_schedule(step: int, total: int, warmup: int = 0, cycles: int = 1,
                            min_ratio: float = 0.0) -> float:
    """Cosine schedule with ``cycles`` warm restarts (SGDR)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    cycles = max(1, cycles)
    cycle_len = max(1, (total - warmup) // cycles)
    pos = (step - warmup) % cycle_len
    progress = pos / cycle_len
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def wsd_schedule(step: int, total: int, warmup: int = 0, min_ratio: float = 0.0,
                 decay_ratio: float = 0.1) -> float:
    """Warmup-Stable-Decay schedule."""
    if warmup > 0 and step < warmup:
        return (step + 1) / max(1, warmup)
    stable_end = int(total * 0.8)
    if step <= stable_end:
        return 1.0
    progress = (step - stable_end) / max(1, total - stable_end)
    progress = min(max(progress, 0.0), 1.0)
    return max(min_ratio, decay_ratio + (1.0 - decay_ratio) * (1.0 - progress))


# --------------------------------------------------------------------------- #
# JSON / IO
# --------------------------------------------------------------------------- #
def write_json(path: str, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from ``path`` returning ``default`` on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def truncate_to_blocks(token_ids: List[int], block_size: int) -> List[List[int]]:
    """Split a flat token stream into fixed-size blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = []
    for i in range(0, max(0, len(token_ids) - block_size + 1), block_size):
        blocks.append(token_ids[i:i + block_size])
    return blocks
