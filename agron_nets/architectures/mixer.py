"""Mixer language model.

A length-agnostic variant of MLP-Mixer: tokens are mixed with a causal
depthwise convolution (so the model works for any sequence length, including
single-token generation steps) and channels are mixed with a two-layer MLP.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, RMSNorm, causal_lm_loss


class AgronMixerConfig(AgronConfig):
    model_type = "agron_mixer"

    def __init__(
        self,
        token_kernel: int = 5,
        channel_mlp_hidden: int = 2048,
        mixer_dropout: float = 0.0,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.token_kernel = token_kernel
        self.channel_mlp_hidden = channel_mlp_hidden
        self.mixer_dropout = mixer_dropout
        self.norm_type = norm_type
        super().__init__(**kwargs)


class MixerBlock(nn.Module):
    """Causal token-mixing convolution followed by channel-mixing MLP."""

    def __init__(self, config: AgronMixerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.token_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.token_kernel,
            groups=config.hidden_size,
            padding=0,
        )
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.channel_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.channel_mlp_hidden),
            nn.GELU(),
            nn.Linear(config.channel_mlp_hidden, config.hidden_size),
            nn.Dropout(config.mixer_dropout),
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        mixed = self.norm1(hidden_states).transpose(1, 2)
        mixed = F.pad(mixed, (self.token_conv.kernel_size[0] - 1, 0))
        mixed = self.token_conv(mixed).transpose(1, 2)
        hidden_states = residual + mixed
        return hidden_states + self.channel_mlp(self.norm2(hidden_states))


class AgronMixerForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Attention-free Mixer decoder."""

    config_class = AgronMixerConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = False

    def __init__(self, config: AgronMixerConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MixerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
