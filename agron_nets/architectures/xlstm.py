"""xLSTM decoder: stabilized sLSTM cell with exponential gating.

The scalar-memory LSTM cell of the xLSTM paper is used here:

    i_t = exp(i~_t),  f_t = exp(f~_t),  o_t = sigmoid(o~_t),  z_t = tanh(z~_t)
    c_t = f_t c_{t-1} + i_t z_t
    n_t = f_t n_{t-1} + i_t
    h_t = o_t * c_t / max(|n_t|, eps)

Because both gates are exponentials they can overflow, so the cell carries the
log-space stabilizer ``m_t = max(f~_t + m_{t-1}, i~_t)`` and rescales the gates
by ``exp(-m_t)``.  A causal depthwise convolution and a residual gated MLP wrap
the recurrence, following the xLSTM block layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin

from .base import AgronConfig, AgronPreTrainedModel, SwiGLU, build_norm, causal_lm_loss


class AgronxLSTMConfig(AgronConfig):
    model_type = "agron_xlstm"

    def __init__(self, norm_type: str = "rms", conv_kernel: int = 4, **kwargs):
        self.norm_type = norm_type
        self.conv_kernel = conv_kernel
        super().__init__(**kwargs)


class SLSTMCell(nn.Module):
    """Scalar-memory LSTM with exponential gating and stabilizer state."""

    def __init__(self, config: AgronxLSTMConfig):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.conv_kernel = config.conv_kernel
        self.eps = 1e-6

        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size=self.conv_kernel, groups=hidden,
            padding=self.conv_kernel - 1, bias=True,
        )
        self.i_proj = nn.Linear(hidden, hidden)
        self.f_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.z_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_norm = build_norm(hidden, config.rms_norm_eps, "rms")

        # Start close to "remember everything, write nothing".
        nn.init.constant_(self.f_proj.bias, 2.0)
        nn.init.constant_(self.i_proj.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        xc = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        i_tilde = self.i_proj(xc)
        f_tilde = self.f_proj(xc)
        o_gate = torch.sigmoid(self.o_proj(xc))
        z = torch.tanh(self.z_proj(xc))

        cell = torch.zeros(bsz, hidden, device=x.device, dtype=x.dtype)
        normalizer = torch.zeros(bsz, hidden, device=x.device, dtype=x.dtype)
        stabilizer = torch.full((bsz, hidden), -1e4, device=x.device, dtype=x.dtype)

        outs = []
        for t in range(seq_len):
            it, ft = i_tilde[:, t], f_tilde[:, t]
            m_new = torch.maximum(ft + stabilizer, it)
            i_gate = torch.exp(it - m_new)
            f_gate = torch.exp(ft + stabilizer - m_new)
            cell = f_gate * cell + i_gate * z[:, t]
            normalizer = f_gate * normalizer + i_gate
            h = o_gate[:, t] * cell / normalizer.abs().clamp(min=self.eps)
            outs.append(h)
            stabilizer = m_new

        h = torch.stack(outs, dim=1)
        return self.out_proj(self.out_norm(h))


class xLSTMBlock(nn.Module):
    def __init__(self, config: AgronxLSTMConfig):
        super().__init__()
        self.cell = SLSTMCell(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.cell(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AgronxLSTMForCausalLM(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronxLSTMConfig
    base_model_prefix = "agron_xlstm"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronxLSTMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([xLSTMBlock(config) for _ in range(config.num_hidden_layers)])
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
