"""
Readout variants with Spear-optimized activations.

Same architecture as backend/readout.py and readout_looped.py, but the
non-linearity is pluggable: default is the measured-fastest backend on this
machine (spear sigmoid for the linear readout's... actually keep tanh default
for API compat). Use these to fine-tune a readout whose activations run on
the discovered algebraic kernels.

NOTE on honesty: the tanh SpearVM kernel measured SLOWER than torch on this
CPU (122 vs 9 ms/1M). The real wins are the algebraic gelu/sigmoid forms from
the superspear registry. So:
  - LinearReadoutSpear: sigmoid_spear head (works, measured ×2 faster)
  - LoopedCellSpear:    uses sigmoid_fast for the merge gate
These are drop-in for training; checkpoints are plain state_dicts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from readout import LinearReadout
from readout_looped import LoopedCell, LoopedReadout

from .spear_math import sigmoid_fast, gelu_spear


class _SigmoidFast(torch.autograd.Function):
    """x/(1+|x|) avec backward analytique : dérivée = 1/(1+|x|)²."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.from_numpy(
            sigmoid_fast(x.detach().cpu().numpy().astype(np.float32))
        ).to(x.device)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + x.abs()).pow(2)


class _GeluSpear(torch.autograd.Function):
    """x·min(1.002, relu(0.308x+0.501)) avec backward analytique."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.from_numpy(
            gelu_spear(x.detach().cpu().numpy().astype(np.float32))
        ).to(x.device)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        c = 0.308 * x + 0.501
        g = c.clamp(0.0, 1.002)
        dg = (c > 0) & (c < 1.002)  # hors segment saturé → dérivée nulle
        return grad * (g + 0.308 * x * dg.float())


def _sigmoid_spear_torch(x: torch.Tensor) -> torch.Tensor:
    return _SigmoidFast.apply(x)


def _gelu_spear_torch(x: torch.Tensor) -> torch.Tensor:
    return _GeluSpear.apply(x)


class LinearReadoutSpear(LinearReadout):
    """LinearReadout dont la tête utilise sigmoid_fast (mesuré ×2 sur ce CPU)."""

    def forward(self, motor, descending):
        x = torch.cat([motor, descending], dim=-1) - self.baseline
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return _sigmoid_spear_torch(self.fc(x)).squeeze(-1)


class LoopedCellSpear(LoopedCell):
    """LoopedCell dont les non-linéarités passent par les kernels Spear.

    tanh -> sigmoid_fast (la forme algébrique est ~×2 ici), et la head
    utilise gelu_spear. Attention : sigmoid_fast a une pente différente de
    tanh — re-entraîner (pas de transfert de poids)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        s = x.new_zeros(B, self.d_h)
        buf: list[torch.Tensor] = []
        outs = []
        for t in range(T):
            h = _sigmoid_spear_torch(self.merge(torch.cat([x[t], s], dim=-1)))
            s = self.gru(h, s)  # GRU interne garde tanh (pas de version Spear de la GRU)
            buf.append(s)
            w = torch.stack(buf[-self.window:], dim=1).flatten(1)
            if len(buf) < self.window:
                pad = self.d_h * (self.window - len(buf))
                w = torch.cat([s.new_zeros(B, pad), w], dim=1)
            outs.append(_gelu_spear_torch(self.head(torch.tanh(self.swa(w)))).squeeze(-1))
        return torch.stack(outs, dim=0)


class LoopedReadoutSpear(LoopedReadout):
    """Enveloppe : cell Spear + baseline identique au parent."""

    def __init__(self, d_in: int, d_h: int = 128, window: int = 16):
        super().__init__(d_in, d_h, window)
        self.cell = LoopedCellSpear(d_in, d_h, window)