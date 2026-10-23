"""Mamba-2 style decoder: multi-head selective SSM (state space duality).

The recurrence follows the SSD formulation used by Mamba-2.  With heads ``h``
and per-head state dimension ``n`` the input-dependent transition is a scalar
per state channel,

    h_t = a_t * h_{t-1} + B_t x_t,      a_t = exp(-exp(A_log) * softplus(dt_t))
    y_t = C_t . h_t + D * x_t

Heads therefore act like the "SS" matrices of the SSD view: the decay is shared
across the head while ``B`` and ``C`` are per-channel, which is what makes the
chunkwise evaluation a cheap scalar scan.  ``dt``, ``B`` and ``C`` all come from
the current token, so the model is *selective*; the chunked scan keeps the
cumulative decay relative to the chunk start so every exponential is <= 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss

CHUNK = 16


class AgronMamba2Config(AgronConfig):
    model_type = "agron_mamba2"

    def __init__(
        self,
        norm_type: str = "rms",
        chunk_size: int = CHUNK,
        state_size: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **kwargs,
    ):
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        self.state_size = state_size
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        super().__init__(**kwargs)


class SelectiveSSM(nn.Module):
    def __init__(self, config: AgronMamba2Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.state_size = config.state_size
        self.chunk_size = config.chunk_size
        self.d_inner = config.hidden_size
        self.groups = max(1, self.num_heads // 4)

        self.in_proj = nn.Linear(config.hidden_size, 2 * self.d_inner + 2 * self.groups * self.state_size
                                 + self.num_heads, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, groups=self.d_inner,
                              padding=3, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, self.num_heads)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=False)
        self.out_norm = build_norm(self.d_inner, config.rms_norm_eps, "rms")

        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def _chunk_scan(self, a_full, b, x, chunk):
        """Chunked cumsum of the scalar decay; `a_full` is the full-sequence decay."""
        bsz, heads, seq_len, dstate = b.shape
        state = torch.zeros(bsz, heads, self.head_dim, dstate, device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        log_a = a_full.log()

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            n = end - start
            logc = torch.cumsum(log_a[:, :, start:end], dim=-1)             # (B,H,C) <= 0
            rel = torch.exp(logc[:, :, :1] - logc)                          # (B,H,C)

            bx = x[:, :, start:end].unsqueeze(-1) * b[:, :, start:end].unsqueeze(-2)
            cumbx = torch.cumsum(rel.unsqueeze(-1).unsqueeze(-1) * bx, dim=2)

            out[:, :, start:end] = torch.einsum("bhcn,bhcdn->bhcd", b[:, :, start:end], cumbx) \
                + torch.einsum("bhcn,bhdn->bhcd", b[:, :, start:end], state)

            state = state * torch.exp(logc[:, :, -1]).view(bsz, heads, 1, 1) \
                + cumbx[:, :, -1] * torch.exp(logc[:, :, -1] - logc[:, :, 0]).view(bsz, heads, 1, 1)
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        heads, dim, dstate = self.num_heads, self.head_dim, self.state_size

        proj = self.in_proj(hidden_states)
        x, z, b_raw, c_raw, dt_raw = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.groups * dstate, self.groups * dstate, heads],
            dim=-1,
        )
        x = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = torch.nn.functional.silu(x)

        b = b_raw.view(bsz, seq_len, self.groups, dstate).mean(dim=2)      # (B,T,N)
        c = c_raw.view(bsz, seq_len, self.groups, dstate).mean(dim=2)

        dt = torch.nn.functional.softplus(dt_raw + self.dt_bias)           # (B,T,H)
        a = torch.exp(-torch.exp(self.A_log).view(1, 1, heads) * dt)       # (B,T,H) in (0,1)
        a = a.clamp(min=1e-5, max=1.0 - 1e-6).transpose(1, 2)              # (B,H,T)

        xh = x.view(bsz, seq_len, heads, dim).transpose(1, 2)             # (B,H,T,Dk)
        bh = b.transpose(1, 2).unsqueeze(1).expand(-1, heads, -1, -1)     # (B,H,T,N)
        ch = c.transpose(1, 2).unsqueeze(1).expand(-1, heads, -1, -1)

        y = self._chunk_scan(a, bh, xh, self.chunk_size)
        y = y + self.D.view(1, heads, 1, 1) * xh
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_inner)
        y = y * torch.sigmoid(z)
        return self.out_proj(self.out_norm(y))


class Mamba2Block(nn.Module):
    def __init__(self, config: AgronMamba2Config):
        super().__init__()
        self.ssm = SelectiveSSM(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.ssm(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronMamba2ForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronMamba2Config
    base_model_prefix = "agron_mamba2"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronMamba2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Mamba2Block(config) for _ in range(config.num_hidden_layers)])
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
