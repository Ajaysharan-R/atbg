"""
Two output heads sitting on top of the GRU memory:
  - SurvivalHead: "how long until the predicted next stage?" (exponential
    survival distribution with censoring support, per spec).
  - RiskHead: "is this conversation an attack?" (focal loss, since positives
    are the rare, important class).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurvivalHead(nn.Module):
    """Models time-to-next-stage as Exponential(rate). rate = softplus(net(memory))
    so it's always positive. Mean predicted time = 1 / rate."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        raw = self.net(memory).squeeze(-1)
        rate = F.softplus(raw) + 1e-6
        return rate  # (B,) — exponential distribution rate parameter

    def expected_time(self, rate: torch.Tensor) -> torch.Tensor:
        return 1.0 / rate

    def loss(self, rate: torch.Tensor, observed_time: torch.Tensor, is_censored: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the exponential distribution, with
        right-censoring support (is_censored=True means "we only know the
        next stage hadn't happened yet by observed_time", e.g. conversation
        ended before the predicted next state occurred)."""
        # Uncensored: NLL = -log(rate) + rate * t
        # Censored:   NLL = rate * t   (log of the survival function S(t) = exp(-rate*t))
        uncensored_nll = -torch.log(rate) + rate * observed_time
        censored_nll = rate * observed_time
        nll = torch.where(is_censored, censored_nll, uncensored_nll)
        return nll.mean()


class RiskHead(nn.Module):
    """Binary attack-probability head, trained with focal loss to handle
    the severe class imbalance real scam conversations have vs. benign."""

    def __init__(self, hidden_dim: int = 256, focal_gamma: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gamma = focal_gamma
        self.alpha = focal_alpha

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        logits = self.net(memory).squeeze(-1)
        return torch.sigmoid(logits)  # (B,) risk probability in [0, 1]

    def focal_loss(self, risk_prob: torch.Tensor, is_attack: torch.Tensor) -> torch.Tensor:
        p_t = torch.where(is_attack.bool(), risk_prob, 1 - risk_prob)
        alpha_t = torch.where(is_attack.bool(), self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - p_t).pow(self.gamma) * torch.log(p_t.clamp(min=1e-8))
        return loss.mean()
