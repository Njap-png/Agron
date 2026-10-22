"""Pure selective state-space language model (Mamba-lite)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, RMSNorm, causal_lm_loss
from .ssm import MambaLiteBlock


class AgronMambaConfig(AgronConfig):
    model_type = "agron_mamba"

    def __init__(
        self,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.norm_type = norm_type
        super().__init__(**kwargs)


class AgronMambaForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """A decoder-only LM whose entire backbone is a selective SSM stack."""

    config_class = AgronMambaConfig
    base_model_prefix = "agron"
    supports_gradient_checkpointing = False

    def __init__(self, config: AgronMambaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                MambaLiteBlock(
                    config.hidden_size,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
                for _ in range(config.num_hidden_layers)
            ]
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
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
