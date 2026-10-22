"""Causal convolutional language model.

A stack of depthwise-separable causal convolutions with a gated activation.
Because the receptive field grows linearly with depth it is a cheap
alternative to attention for local sequence modelling.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, RMSNorm, causal_lm_loss


class AgronConvConfig(AgronConfig):
    model_type = "agron_conv"

    def __init__(
        self,
        kernel_size: int = 4,
        conv_expand: int = 2,
        conv_dropout: float = 0.0,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.conv_expand = conv_expand
        self.conv_dropout = conv_dropout
        self.norm_type = norm_type
        super().__init__(**kwargs)


class GatedCausalConv(nn.Module):
    """Depthwise causal convolution with a GLU-style gate."""

    def __init__(self, d_model: int, kernel_size: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_inner, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_inner, d_inner, kernel_size=kernel_size, groups=d_inner, padding=0
        )
        self.pointwise_out = nn.Conv1d(d_inner, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.pointwise_in(x.transpose(1, 2))
        value, gate = h.chunk(2, dim=1)
        value = F.pad(value, (self.kernel_size - 1, 0))
        value = self.depthwise(value)
        value = F.silu(value) * torch.sigmoid(gate)
        if self.dropout > 0 and self.training:
            value = F.dropout(value, p=self.dropout)
        out = self.pointwise_out(value)
        return out.transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, config: AgronConvConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.conv = GatedCausalConv(
            config.hidden_size,
            kernel_size=config.kernel_size,
            expand=config.conv_expand,
            dropout=config.conv_dropout,
        )

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.conv(self.norm(hidden_states))


class AgronConvForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Causal convolution decoder."""

    config_class = AgronConvConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = False

    def __init__(self, config: AgronConvConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([ConvBlock(config) for _ in range(config.num_hidden_layers)])
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
