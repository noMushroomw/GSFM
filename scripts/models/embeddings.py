from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):

    def __init__(self, dim: int, max_period: float = 100.0, min_period: float = 1e-2):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("encoding dim must be even")
        half = dim // 2

        exponents = torch.linspace(0.0, 1.0, half)
        periods = min_period * (max_period / min_period) ** exponents
        self.register_buffer("omega", 2.0 * math.pi / periods, persistent=False)
        self.dim = dim

    def forward(self, x):
        arg = x.unsqueeze(-1) * self.omega.to(x.dtype)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class MLP(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int, zero_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift
