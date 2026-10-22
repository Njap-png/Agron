"""LLaMA wrapper.

The published Agron weights were trained with a stock Hugging Face LLaMA
model, so that architecture is kept as the reference implementation and stays
byte-compatible with ``model.safetensors`` in the repository root.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class AgronLlamaConfig(LlamaConfig):
    """LLaMA config that also carries the Agron bookkeeping fields.

    ``model_type`` stays ``"llama"`` so checkpoints remain interchangeable
    with the stock Hugging Face implementation.
    """

    model_type = "llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AgronLlamaForCausalLM(LlamaForCausalLM):
    """Stock LLaMA decoder exposed through the Agron registry."""

    config_class = AgronLlamaConfig
    base_model_prefix = "model"


def build_llama_config(
    vocab_size: int = 16384,
    hidden_size: int = 512,
    intermediate_size: int = 2048,
    num_hidden_layers: int = 8,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 512,
    **kwargs,
) -> LlamaConfig:
    """Build a LLaMA config with Agron's defaults."""
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=kwargs.pop("bos_token_id", 0),
        eos_token_id=kwargs.pop("eos_token_id", 2),
        pad_token_id=kwargs.pop("pad_token_id", 1),
        tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
        **kwargs,
    )
