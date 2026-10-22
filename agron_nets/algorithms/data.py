"""Data-order algorithms: sequence-length curriculum."""

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch

from .base import AlgorithmContext, BaseAlgorithm


class Curriculum(BaseAlgorithm):
    """Sequence-length curriculum.

    Early in training the engine is instructed to use short sequences; the
    usable length then grows to the model maximum.  This is a simple and
    reliable curriculum for language modelling.
    """

    name = "curriculum"
    category = "data"
    description = "Grows the training sequence length from short to full."

    def __init__(self, start_ratio: float = 0.25, end_ratio: float = 1.0, schedule: str = "linear"):
        super().__init__(start_ratio=start_ratio, end_ratio=end_ratio, schedule=schedule)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.schedule = schedule

    def _ratio(self, ctx) -> float:
        total = max(1, ctx.max_steps)
        t = min(1.0, ctx.step / total)
        if self.schedule == "cosine":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.start_ratio + (self.end_ratio - self.start_ratio) * t

    def on_step_begin(self, ctx, batch):
        ratio = self._ratio(ctx)
        input_ids = batch.get("input_ids")
        if input_ids is None or ratio >= 1.0:
            ctx.set("seq_len_ratio", 1.0)
            return batch
        max_len = max(1, int(round(input_ids.size(1) * ratio)))
        trimmed = dict(batch)
        trimmed["input_ids"] = input_ids[:, :max_len]
        if "labels" in batch and batch["labels"] is not None:
            trimmed["labels"] = batch["labels"][:, :max_len]
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            trimmed["attention_mask"] = batch["attention_mask"][:, :max_len]
        ctx.set("seq_len_ratio", ratio)
        return trimmed


class DifficultyCurriculum(BaseAlgorithm):
    """Sorts the dataset by a difficulty proxy (sequence length).

    The engine exposes the active sampler through ``ctx.get("sampler")``.
    """

    name = "difficulty_curriculum"
    category = "data"
    description = "Orders samples from short/easy to long/hard."

    def __init__(self, chunk_size: int = 64, easy_first: bool = True):
        super().__init__(chunk_size=chunk_size, easy_first=easy_first)
        self.chunk_size = chunk_size
        self.easy_first = easy_first

    def on_train_begin(self, ctx):
        dataset = ctx.get("dataset")
        if dataset is None:
            return
        try:
            lengths = [len(x) for x in dataset]
        except Exception:
            return
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=not self.easy_first)
        ctx.set("sample_order", order)
