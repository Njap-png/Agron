"""Hybrid attention / state-space architecture.

Layers alternate between a LLaMA-style causal attention block and a Mamba-lite
selective SSM block, which is the design used by several modern
hybrid-sequence models.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import (
    AgronConfig,
    AgronPreTrainedModel,
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
    build_norm,
    causal_lm_loss,
)
from .ssm import MambaLiteBlock


class AgronHybridConfig(AgronConfig):
    model_type = "agron_hybrid"

    def __init__(
        self,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        ssm_ratio: float = 0.5,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.ssm_ratio = ssm_ratio
        self.norm_type = norm_type
        super().__init__(**kwargs)


class HybridAttentionBlock(nn.Module):
    def __init__(self, config: AgronHybridConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class AgronHybridForCausalLM(AgronPreTrainedModel, GenerationMixin):
    """Decoder that interleaves attention and selective SSM layers."""

    config_class = AgronHybridConfig
    base_model_prefix = "agron"

    def __init__(self, config: AgronHybridConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        layers = nn.ModuleList()
        self.layer_kinds: List[str] = []
        n_ssm = int(round(config.num_hidden_layers * config.ssm_ratio))
        n_ssm = min(max(n_ssm, 0), config.num_hidden_layers)
        ssm_positions = set()
        if n_ssm:
            stride = config.num_hidden_layers / n_ssm
            for i in range(n_ssm):
                ssm_positions.add(min(config.num_hidden_layers - 1, int(round(i * stride))))
        for i in range(config.num_hidden_layers):
            if i in ssm_positions:
                layers.append(
                    MambaLiteBlock(
                        config.hidden_size,
                        d_state=config.d_state,
                        d_conv=config.d_conv,
                        expand=config.expand,
                    )
                )
                self.layer_kinds.append("ssm")
            else:
                layers.append(HybridAttentionBlock(config))
                self.layer_kinds.append("attention")

        self.layers = layers
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
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
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        for kind, layer in zip(self.layer_kinds, self.layers):
            if kind == "attention":
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids
                )
            else:
                hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
