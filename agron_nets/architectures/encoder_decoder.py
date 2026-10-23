"""Encoder–Decoder architectures for Agron.

Implements a T5/BART style sequence-to-sequence model with:
- Bidirectional encoder (full self-attention)
- Causal decoder with cross-attention to encoder
- RoPE/RMSNorm/SwiGLU compatible with the rest of the Agron zoo
- HuggingFace compatible (``PreTrainedModel`` + ``GenerationMixin``)

Also registers ``encdec`` and ``bart`` style variants.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

from .base import (
    AgronConfig,
    AgronPreTrainedModel,
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
    build_norm,
    causal_lm_loss,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class AgronEncDecConfig(AgronConfig):
    model_type = "agron_encdec"

    def __init__(
        self,
        is_encoder_decoder: bool = True,
        tie_word_embeddings: bool = False,
        encoder_layers: Optional[int] = None,
        decoder_layers: Optional[int] = None,
        cross_attention: bool = True,
        norm_type: str = "rms",
        norm_before: bool = True,
        **kwargs,
    ):
        self.is_encoder_decoder = is_encoder_decoder
        if kwargs.get("decoder_start_token_id") is None:
            self.decoder_start_token_id = kwargs.get("bos_token_id", 0)
        self.encoder_layers = encoder_layers or kwargs.get("num_hidden_layers", 6)
        self.decoder_layers = decoder_layers or kwargs.get("num_hidden_layers", 6)
        self.cross_attention = cross_attention
        self.norm_type = norm_type
        self.norm_before = norm_before
        super().__init__(**kwargs)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class BidirectionalSelfAttention(nn.Module):
    """Full (non-causal) self-attention for the encoder."""

    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        self.self_attn = BidirectionalSelfAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class AgronEncoder(nn.Module):
    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_layers)])
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.gradient_checkpointing = False

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# Decoder with cross-attention
# --------------------------------------------------------------------------- #
class CrossAttention(nn.Module):
    """Cross-attention from decoder queries to encoder keys/values."""

    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, cross_attn_mask: Optional[torch.Tensor] = None):
        bsz, tgt_len, _ = hidden_states.shape
        src_len = encoder_hidden_states.shape[1]

        q = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if cross_attn_mask is not None:
            if cross_attn_mask.dim() == 2:
                cross_attn_mask = cross_attn_mask[:, None, None, :]
            attn = attn.masked_fill(cross_attn_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
        return self.o_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            bias=config.attention_bias,
        )
        self.cross_attn = CrossAttention(config) if config.cross_attention else None
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        self.post_attention_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        if self.cross_attn is not None:
            self.cross_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)
        else:
            self.cross_layernorm = None
        self.final_layernorm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        if self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_layernorm(hidden_states)
            hidden_states = residual + self.cross_attn(hidden_states, encoder_hidden_states, cross_attn_mask)

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class AgronDecoder(nn.Module):
    def __init__(self, config: AgronEncDecConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layers)])
        self.norm = build_norm(config.hidden_size, config.rms_norm_eps, config.norm_type)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if position_ids is None and hidden_states is not None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attn_mask=encoder_attention_mask,
                position_ids=position_ids,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class AgronEncDecForConditionalGeneration(AgronPreTrainedModel, GenerationMixin):
    config_class = AgronEncDecConfig
    base_model_prefix = "agron_encdec"
    supports_gradient_checkpointing = True

    def __init__(self, config: AgronEncDecConfig):
        super().__init__(config)
        self.encoder = AgronEncoder(config)
        self.decoder = AgronDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        if config.tie_word_embeddings:
            self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight
            self.lm_head.weight = self.encoder.embed_tokens.weight

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def generate(self, *args, **kwargs):
        """Self-contained encoder-decoder generation (greedy or sampling)."""
        return self._seq2seq_generate(*args, **kwargs)

    @torch.no_grad()
    def _seq2seq_generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id=None,
        decoder_start_token_id=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required for encoder-decoder generation")
        device = input_ids.device
        bsz = input_ids.shape[0]

        encoder_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        start = decoder_start_token_id
        if start is None:
            start = getattr(self.config, "decoder_start_token_id", None)
        if start is None:
            start = getattr(self.config, "bos_token_id", 0)
        generated = torch.full((bsz, 1), int(start), dtype=torch.long, device=device)

        eos = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", None)
        for _ in range(max_new_tokens):
            hidden = self.decoder(
                input_ids=generated,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=attention_mask,
            )
            logits = self.lm_head(hidden[:, -1, :]).float()

            if do_sample:
                logits = logits / max(temperature, 1e-5)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    mask = cumulative - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_token = sorted_idx.gather(-1, torch.multinomial(probs, 1))
                else:
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if eos is not None and bool((next_token == eos).all()):
                break
        return generated

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        position_ids=None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        if encoder_outputs is None:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
            )
        else:
            encoder_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (list, tuple)) else encoder_outputs

        decoder_hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=position_ids,
        )

        logits = self.lm_head(decoder_hidden_states)
        loss = None
        if labels is not None:
            loss = causal_lm_loss(logits, labels, label_smoothing)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden_states,
        )


# Aliases for convenience
AgronBartForConditionalGeneration = AgronEncDecForConditionalGeneration
AgronT5ForConditionalGeneration = AgronEncDecForConditionalGeneration
