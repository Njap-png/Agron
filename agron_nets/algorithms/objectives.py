"""Objective-level algorithms: knowledge distillation and DPO."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import AlgorithmContext, BaseAlgorithm


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combined cross-entropy + KL divergence distillation loss."""
    tau = max(temperature, 1e-6)
    student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    teacher_probs = F.softmax(teacher_logits / tau, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (tau ** 2)
    if labels is None:
        return alpha * kl
    ce = F.cross_entropy(
        student_logits[..., :-1, :].reshape(-1, student_logits.shape[-1]).float(),
        labels[..., 1:].reshape(-1),
    )
    return (1 - alpha) * ce + alpha * kl


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Direct Preference Optimisation loss (Rafailov et al., 2023)."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    return -F.logsigmoid(logits - label_smoothing).mean()


def sequence_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of per-token log probabilities for each sequence in the batch."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=-1)


class Distillation(BaseAlgorithm):
    """Distil a teacher model into the student during training."""

    name = "distillation"
    category = "objective"
    description = "Matches a teacher model's logits via temperature-scaled KL."

    def __init__(self, teacher=None, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__(temperature=temperature, alpha=alpha)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def on_train_begin(self, ctx):
        if self.teacher is None:
            self.teacher = ctx.get("teacher")
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def train_step(self, ctx, batch):
        batch = self.on_step_begin(ctx, batch)
        outputs = ctx.model(**batch)
        student_logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = batch.get("labels")
        if self.teacher is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            return loss
        with torch.no_grad():
            teacher_out = self.teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch.get("attention_mask"))
            teacher_logits = teacher_out["logits"] if isinstance(teacher_out, dict) else teacher_out.logits
        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 temperature=self.temperature, alpha=self.alpha)
        loss.backward()
        return loss


class DPO(BaseAlgorithm):
    """Direct Preference Optimisation over (chosen, rejected) pairs.

    The batch must contain ``chosen_input_ids``, ``rejected_input_ids`` and the
    matching ``*_attention_mask`` tensors.
    """

    name = "dpo"
    category = "objective"
    description = "Trains on preference pairs with the DPO objective."

    def __init__(self, beta: float = 0.1, reference_model=None, label_smoothing: float = 0.0):
        super().__init__(beta=beta, label_smoothing=label_smoothing)
        self.beta = beta
        self.reference_model = reference_model
        self.label_smoothing = label_smoothing

    def on_train_begin(self, ctx):
        if self.reference_model is None:
            self.reference_model = ctx.get("reference_model")

    def train_step(self, ctx, batch):
        chosen_ids = batch.get("chosen_input_ids")
        rejected_ids = batch.get("rejected_input_ids")
        if chosen_ids is None or rejected_ids is None:
            raise ValueError("DPO requires chosen_input_ids and rejected_input_ids in the batch")

        chosen_mask = batch.get("chosen_attention_mask")
        rejected_mask = batch.get("rejected_attention_mask")

        policy_chosen = sequence_log_probs(ctx.model, chosen_ids, chosen_mask)
        policy_rejected = sequence_log_probs(ctx.model, rejected_ids, rejected_mask)

        if self.reference_model is not None:
            with torch.no_grad():
                ref_chosen = sequence_log_probs(self.reference_model, chosen_ids, chosen_mask)
                ref_rejected = sequence_log_probs(self.reference_model, rejected_ids, rejected_mask)
        else:
            ref_chosen = torch.zeros_like(policy_chosen)
            ref_rejected = torch.zeros_like(policy_rejected)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
                        beta=self.beta, label_smoothing=self.label_smoothing)
        loss.backward()
        return loss
