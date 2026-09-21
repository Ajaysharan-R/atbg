"""
TD-TO: Time-Decayed Transition Operator — the project's main technical
contribution.

Predicts P(next behaviour state | current state, GRU memory, elapsed time),
where each of the 12 states has its OWN learned decay rate. Intuition: how
much a given manipulative state "still applies" after a time gap is
state-specific — an "urgency" cue decays fast (its whole point is immediacy,
so if 3 days pass it's stale), while "rapport/trust" decays slowly (trust
built up doesn't evaporate in a few hours the way urgency does).

decay(state, elapsed) = exp(-lambda_state * elapsed)
lambda_state is a learned, strictly-positive per-state parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STATES = 12


class TimeDecayedTransitionOperator(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_states: int = NUM_STATES):
        super().__init__()
        self.num_states = num_states

        # log_lambda so the actual decay rate (via softplus) is always > 0
        # without needing a manual clamp during optimisation.
        self.log_lambda = nn.Parameter(torch.zeros(num_states))

        # memory + current-state one-hot -> transition logits
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + num_states, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_states),
        )

    def decay_rates(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)  # (num_states,) — always positive

    def forward(self, memory: torch.Tensor, current_state_idx: torch.Tensor, elapsed_seconds: torch.Tensor):
        """
        memory:             (B, hidden_dim) — GRU output at the current turn.
        current_state_idx:  (B,) long — index into the 12 anchored states.
        elapsed_seconds:    (B,) — time since the current turn (for predicting
                             the state that will hold once that much time passes).
        Returns: next_state_probs (B, num_states)
        """
        current_onehot = F.one_hot(current_state_idx, self.num_states).float()
        transition_logits = self.transition_net(torch.cat([memory, current_onehot], dim=-1))

        # Per-state decay: states with a high learned lambda lose relevance
        # fast as elapsed time grows; this directly reweights the logits
        # before softmax, rather than just being a post-hoc discount.
        rates = self.decay_rates()                                   # (num_states,)
        decay = torch.exp(-rates.unsqueeze(0) * elapsed_seconds.unsqueeze(1))  # (B, num_states)

        adjusted_logits = transition_logits * decay
        next_state_probs = F.softmax(adjusted_logits, dim=-1)
        return next_state_probs

    def half_life_seconds(self) -> torch.Tensor:
        """ln(2)/lambda per state — a human-readable summary of how fast each
        state's influence decays. Useful for the interpretability write-up."""
        rates = self.decay_rates()
        return torch.log(torch.tensor(2.0)) / rates.clamp(min=1e-6)
