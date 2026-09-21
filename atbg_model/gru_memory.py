"""
GRU temporal memory: carries a running summary of the conversation so far,
so the model doesn't have to reprocess turns 1..N-1 from scratch when turn N
arrives. This is also what makes the <500ms warm streaming-append target in
the spec possible — the stored hidden state IS the memory, reused as-is.
"""

import torch
import torch.nn as nn


class GRUMemory(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, node_features: torch.Tensor, prev_hidden: torch.Tensor | None = None):
        """
        node_features: (1, T, input_dim) — the HGT output for one conversation's
                       turns, in order, batch dim fixed at 1 (memory is per-conversation).
        prev_hidden:   (1, 1, hidden_dim) stored hidden state from the last
                       streaming call, or None for a fresh conversation.
        Returns: (outputs (1, T, hidden_dim), new_hidden (1, 1, hidden_dim))
        """
        outputs, new_hidden = self.gru(node_features, prev_hidden)
        return outputs, new_hidden

    def step(self, single_turn_feature: torch.Tensor, prev_hidden: torch.Tensor):
        """Streaming-append path: process exactly one new turn, reusing the
        stored hidden state instead of recomputing the whole conversation."""
        return self.forward(single_turn_feature.unsqueeze(1), prev_hidden)
