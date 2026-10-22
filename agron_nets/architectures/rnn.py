"""Recurrent (GRU / LSTM) language models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, RMSNorm, causal_lm_loss


class _RecurrentConfig(AgronConfig):
    """Shared configuration for the recurrent Agron models."""

    def __init__(self, rnn_dropout: float = 0.0, **kwargs):
        self.rnn_dropout = rnn_dropout
        super().__init__(**kwargs)


class _RecurrentForCausalLM(AgronPreTrainedModel):
    """Common plumbing for GRU / LSTM decoders."""

    config_class = _RecurrentConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = False

    rnn_cls = nn.GRU

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rnn = self.rnn_cls(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.num_hidden_layers > 1 else 0.0,
        )
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
        hidden_states, _ = self.rnn(inputs_embeds)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}


class AgronGRUConfig(_RecurrentConfig):
    model_type = "agron_gru"

    def __init__(self, rnn_dropout: float = 0.0, **kwargs):
        super().__init__(rnn_dropout=rnn_dropout, **kwargs)


class AgronGRUForCausalLM(_RecurrentForCausalLM, GenerationMixin):
    """GRU based decoder."""

    config_class = AgronGRUConfig
    rnn_cls = nn.GRU


class AgronLSTMConfig(_RecurrentConfig):
    model_type = "agron_lstm"

    def __init__(self, rnn_dropout: float = 0.0, **kwargs):
        super().__init__(rnn_dropout=rnn_dropout, **kwargs)


class AgronLSTMForCausalLM(_RecurrentForCausalLM, GenerationMixin):
    """LSTM based decoder."""

    config_class = AgronLSTMConfig
    rnn_cls = nn.LSTM
