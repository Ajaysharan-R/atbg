"""
Time2Vec: encodes elapsed time between turns so the model treats
10 seconds and 10 days as genuinely different, not the same "1 step".

Reference: Kazemi et al., "Time2Vec: Learning a Vector Representation of Time".
"""

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int = 16):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >= 2 (1 linear + >=1 periodic term)")
        self.out_dim = out_dim

        # One linear (trend) term + (out_dim - 1) periodic terms
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(out_dim - 1))
        self.b = nn.Parameter(torch.randn(out_dim - 1))

    def forward(self, elapsed_seconds: torch.Tensor) -> torch.Tensor:
        """elapsed_seconds: (..., 1) tensor of seconds since the previous turn.
        Missing/unknown gaps should be pre-filled with a sentinel (e.g. median
        gap for that channel) upstream — Time2Vec itself has no null handling."""
        # log1p compresses the huge dynamic range (seconds to days) before
        # the learned periodic terms, which stabilises training a lot.
        t = torch.log1p(elapsed_seconds.clamp(min=0))

        linear = self.w0 * t + self.b0                      # (..., 1)
        periodic = torch.sin(t * self.w + self.b)             # (..., out_dim - 1)
        return torch.cat([linear, periodic], dim=-1)          # (..., out_dim)
