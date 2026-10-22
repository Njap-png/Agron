"""Selective state-space (Mamba-lite) building blocks.

The scan is implemented as a numerically stable Hillis-Steele doubling scan
rather than the naive Python loop: every step only touches the log-domain
cumulative sum of the decay, which keeps the whole recurrence exact while
being ``O(T log T)`` and fully vectorised.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSNorm


def selective_scan(
    decay: torch.Tensor,
    drive: torch.Tensor,
    readout: torch.Tensor,
    skip: torch.Tensor,
    chunk_hint: int = 0,
) -> torch.Tensor:
    """Diagonal selective state-space scan.

    Parameters
    ----------
    decay : ``(B, D, N, L)``
        Per-step, per-channel state decay in ``(-1, 0]``.
    drive : ``(B, D, N, L)``
        Per-step input contribution (already multiplied by ``B_t``).
    readout : ``(B, D, N, L)``
        Per-step output projection ``C_t``.
    skip : ``(B, D, L)``
        Direct (diagonal) skip connection ``D x_t``.

    Returns
    -------
    ``(B, D, L)`` tensor of outputs.
    """
    bsz, dim, state, length = decay.shape

    # Clamp so |a| < 1 -> the recurrence is contractive and stable.
    decay = decay.clamp(min=-0.9999, max=0.0)
    log_decay = torch.log(-decay.clamp(max=-1e-9))
    cum = torch.cumsum(log_decay, dim=-1)  # (B, D, N, L)

    h = drive
    stride = 1
    while stride < length:
        # prod_{r=t-s+1..t} a_r = exp(cum[t] - cum[t-s])
        shifted = F.pad(cum[..., :-stride], (stride, 0), value=0.0)
        window = torch.exp(cum - shifted)  # <= 1 by construction
        h_shift = F.pad(h[..., :-stride], (stride, 0), value=0.0)
        h = h + window * h_shift
        stride *= 2

    y = (h * readout).sum(dim=2)  # (B, D, L)
    return y + skip


class SelectiveSSM(nn.Module):
    """Selective SSM core (input dependent ``B``, ``C`` and ``dt``)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 4,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.b_proj = nn.Linear(d_model, d_state, bias=False)
        self.c_proj = nn.Linear(d_model, d_state, bias=False)

        # dt bias initialised so softplus(dt_bias) lands in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_model) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
            + torch.log(torch.tensor(dt_min))
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A_log parameterised so that A = -exp(A_log) is strictly negative.
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, D)`` -> ``(B, L, D)``."""
        bsz, length, dim = x.shape
        dt = F.softplus(self.dt_proj(x))  # (B, L, D)
        b = self.b_proj(x)  # (B, L, N)
        c = self.c_proj(x)  # (B, L, N)

        a = -torch.exp(self.A_log)  # (D, N)
        decay = dt.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0)  # (B, L, D, N)
        drive = x.unsqueeze(-1) * b.unsqueeze(2)  # (B, L, D, N)

        decay = decay.permute(0, 2, 3, 1).contiguous()  # (B, D, N, L)
        drive = drive.permute(0, 2, 3, 1).contiguous()
        readout = c.transpose(1, 2).unsqueeze(1).expand(-1, dim, -1, -1).contiguous()  # (B, D, N, L)

        skip = x.transpose(1, 2)  # (B, D, L)
        y = selective_scan(decay, drive, readout, skip)  # (B, D, L)
        return y.permute(0, 2, 1).contiguous()


class MambaLiteBlock(nn.Module):
    """Pre-norm Mamba-lite residual block."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=0
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, d_conv=d_conv)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        x = self.norm(hidden_states)
        x, z = self.in_proj(x).chunk(2, dim=-1)

        # causal depthwise convolution
        x = x.transpose(1, 2)
        x = F.pad(x, (self.conv1d.kernel_size[0] - 1, 0))
        x = self.conv1d(x).transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(z)
        return residual + self.out_proj(y)
