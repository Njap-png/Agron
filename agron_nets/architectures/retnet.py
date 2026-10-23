"""RetNet: retention-based decoder (Sun et al., 2023).

Instead of softmax attention, each head uses a *retention* with an exponential
decay ``gamma`` per head (multi-scale retention):

    out_t = sum_{s<=t} gamma^(t-s) * (q_t . k_s) * v_s

The parallel form below is exact and used for training. ``Retention.recurrent``
implements the equivalent O(1) state update used at inference time.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss


class AgronRetNetConfig(AgronConfig):
    model_type = "agron_retnet"

    def __init__(self, norm_type: str = "rms", **kwargs):
        self.norm_type = norm_type
        super().__init__(**kwargs)


class MultiScaleRetention(nn.Module):
    """Parallel retention with a distinct decay per head."""

    def __init__(self, config: AgronRetNetConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.g_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_norm = build_norm(config.hidden_size, config.rms_norm_eps, "rms")

        # Paper: gamma_h = 1 - 2^(-5-h), keeping every head in (0, 1).
        gammas = [1.0 - 2.0 ** (-5 - h) for h in range(self.num_heads)]
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32), persistent=False)

    def _decay_matrix(self, seq_len: int, device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        diff = pos[:, None] - pos[None, :]  # t - s
        causal = diff >= 0
        log_gamma = torch.log(self.gammas.to(device))
        decay = torch.exp(diff.clamp(min=0)[None, :, :] * log_gamma[:, None, None])
        return decay * causal[None].float()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        decay = self._decay_matrix(seq_len, hidden_states.device).to(scores.dtype)
        out = torch.matmul(scores * decay, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = out * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(self.out_norm(out))

    def recurrent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """O(T) state-space form of the same retention (inference only)."""
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        state = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device,
                            dtype=hidden_states.dtype)
        outs = []
        for t in range(seq_len):
            state = state * self.gammas.to(state.device).view(1, -1, 1, 1)
            state = state + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t] * self.scale, state))
        out = torch.stack(outs, dim=2)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(self.out_norm(out * torch.sigmoid(self.g_proj(hidden_states))))


class RetNetBlock(nn.Module):
    def __init__(self, config: AgronRetNetConfig):
        super().__init__()
        self.retention = MultiScaleRetention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = residual + self.retention(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronRetNetForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronRetNetConfig
    base_model_prefix = "agron_retnet"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronRetNetConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([RetNetBlock(config) for _ in range(config.num_hidden_layers)])
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

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None,
                labels=None, label_smoothing: float = 0.0, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = causal_lm_loss(logits, labels, label_smoothing) if labels is not None else None
        return {"logits": logits, "loss": loss}
