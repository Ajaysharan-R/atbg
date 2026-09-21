"""
Heterogeneous Graph Transformer (HGT) over the per-conversation graph.

Nodes = turns. Edges = {NEXT (sequential), ARTIFACT_LINK (turn -> artifact
node), SAME_SPEAKER (optional)}. Implemented from scratch in pure PyTorch
(no torch_geometric dependency) so the project has one less heavy/finicky
install to fight with on a laptop.

Config matches the spec: 2 layers, 4 attention heads, 256 hidden dim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EDGE_TYPES = ["next", "artifact_link", "same_speaker"]


class HGTLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_edge_types: int = len(EDGE_TYPES)):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_edge_types = num_edge_types

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # A learned scalar bias per (edge_type, head) — this is what makes
        # the attention "heterogeneous": a NEXT edge and an ARTIFACT_LINK
        # edge are allowed to matter differently to each attention head.
        self.edge_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x:         (N, hidden_dim) node features for ONE conversation graph.
        adj:       (N, N) bool tensor, adj[i, j] = True if there's an edge j -> i.
        edge_type: (N, N) long tensor, edge type id for adj[i, j] (ignored where adj is False).
        """
        N = x.size(0)
        residual = x

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # scores[i, j, h] = how much node i attends to node j, head h
        scores = torch.einsum("ihd,jhd->ijh", q, k) / (self.head_dim ** 0.5)   # (N, N, H)

        # Add the heterogeneous edge-type bias, then mask out non-edges
        bias = self.edge_bias[edge_type]                                       # (N, N, H)
        scores = scores + bias
        scores = scores.masked_fill(~adj.unsqueeze(-1), float("-inf"))

        # A node with no incoming edges at all would produce all -inf and
        # NaN after softmax — give it a self-loop fallback.
        no_edges = ~adj.any(dim=1)                                             # (N,)
        if no_edges.any():
            scores[no_edges, no_edges, :] = 0.0
            adj = adj.clone()
            adj[no_edges, no_edges] = True

        attn = F.softmax(scores, dim=1)                                        # softmax over j
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(N, self.hidden_dim)
        out = self.out_proj(out)

        x = self.norm1(residual + out)
        x = self.norm2(x + self.ffn(x))
        return x


class HGT(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [HGTLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj, edge_type)
        return x


def build_conversation_graph(num_turns: int, artifact_turn_pairs: list[tuple[int, int]] | None = None,
                              speakers: list[str] | None = None):
    """
    Builds the adjacency + edge-type tensors for one conversation.
    - Sequential NEXT edges between consecutive turns (both directions).
    - Optional ARTIFACT_LINK edges between two turns that share an artifact
      (pass pairs of turn indices that co-occur on the same artifact).
    - Optional SAME_SPEAKER edges between turns from the same speaker.

    Node ids 0..num_turns-1 are turn nodes (in order).
    """
    N = num_turns
    adj = torch.zeros(N, N, dtype=torch.bool)
    edge_type = torch.zeros(N, N, dtype=torch.long)

    next_id = EDGE_TYPES.index("next")
    artifact_id = EDGE_TYPES.index("artifact_link")
    speaker_id = EDGE_TYPES.index("same_speaker")

    for i in range(N - 1):
        adj[i, i + 1] = True
        edge_type[i, i + 1] = next_id
        adj[i + 1, i] = True
        edge_type[i + 1, i] = next_id

    if artifact_turn_pairs:
        for a, b in artifact_turn_pairs:
            if a < N and b < N and a != b:
                adj[a, b] = True
                edge_type[a, b] = artifact_id
                adj[b, a] = True
                edge_type[b, a] = artifact_id

    if speakers:
        for i in range(N):
            for j in range(N):
                if i != j and speakers[i] == speakers[j] and not adj[i, j]:
                    adj[i, j] = True
                    edge_type[i, j] = speaker_id

    return adj, edge_type
