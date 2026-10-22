"""Shared fixtures + helpers for the Agron zoo test-suite."""

from __future__ import annotations

import pytest
import torch

from agron_nets.architectures import build_architecture

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=32,
)


def make_model(arch: str = "transformer", **overrides):
    cfg = dict(TINY)
    cfg.update(overrides)
    model, config = build_architecture(arch, **cfg)
    return model, config


def make_batch(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 64, device="cpu"):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }


class ListLoader:
    """Minimal iterable dataloader yielding pre-made batches."""

    def __init__(self, batches):
        self.batches = list(batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def make_loader(steps: int, batch_size: int = 2, seq_len: int = 8, vocab_size: int = 64):
    return ListLoader([make_batch(batch_size, seq_len, vocab_size) for _ in range(steps)])


@pytest.fixture
def tiny_model():
    model, _ = make_model("transformer")
    return model


@pytest.fixture
def loader():
    return make_loader(8)
