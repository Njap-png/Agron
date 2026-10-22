"""Regularisation algorithms: label smoothing, mixup, weight-decay schedules."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .base import AlgorithmContext, BaseAlgorithm, compose_loss


class LabelSmoothing(BaseAlgorithm):
    """Recompute the causal-LM loss with label smoothing."""

    name = "label_smoothing"
    category = "regularisation"
    description = "Softens one-hot targets to reduce over-confidence."

    def __init__(self, epsilon: float = 0.1):
        super().__init__(epsilon=epsilon)
        self.epsilon = epsilon

    def compute_loss(self, ctx, batch, outputs, loss):
        labels = batch.get("labels")
        if labels is None:
            return loss
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]).float()
        labels = labels[..., 1:].reshape(-1)
        return F.cross_entropy(logits, labels, label_smoothing=self.epsilon)


class Mixup(BaseAlgorithm):
    """Embedding-space mixup for language models.

    Mixes input embeddings between random pairs and interpolates the
    corresponding cross-entropy terms.
    """

    name = "mixup"
    category = "regularisation"
    description = "Interpolates input embeddings between random pairs."

    def __init__(self, alpha: float = 0.2):
        super().__init__(alpha=alpha)
        self.alpha = alpha

    def train_step(self, ctx, batch):
        batch = self.on_step_begin(ctx, batch)
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        if input_ids is None or labels is None or self.alpha <= 0:
            outputs = ctx.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = self.compute_loss(ctx, batch, outputs, loss)
            loss = compose_loss(ctx, batch, outputs, loss, skip=self)
            loss.backward()
            return loss

        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())
        perm = torch.randperm(input_ids.size(0), device=input_ids.device)
        embeddings = ctx.model.get_input_embeddings()(input_ids)
        mixed = lam * embeddings + (1.0 - lam) * embeddings[perm]

        outputs = ctx.model(inputs_embeds=mixed, attention_mask=batch.get("attention_mask"))
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]).float()
        loss_a = F.cross_entropy(logits, labels[..., 1:].reshape(-1))
        loss_b = F.cross_entropy(logits, labels[perm][..., 1:].reshape(-1))
        loss = lam * loss_a + (1.0 - lam) * loss_b
        loss.backward()
        return loss


class WeightDecaySchedule(BaseAlgorithm):
    """Ramp the optimiser weight decay from ``start`` to ``end`` over training."""

    name = "weight_decay_schedule"
    category = "regularisation"
    description = "Linearly ramps weight decay across the run."

    def __init__(self, start: float = 0.0, end: float = 0.1):
        super().__init__(start=start, end=end)
        self.start = start
        self.end = end

    def on_step_begin(self, ctx, batch):
        if ctx.optimizer is None or ctx.max_steps <= 0:
            return batch
        t = min(1.0, ctx.step / max(1, ctx.max_steps - 1))
        value = self.start + (self.end - self.start) * t
        for group in ctx.optimizer.param_groups:
            group["weight_decay"] = value
        return batch
