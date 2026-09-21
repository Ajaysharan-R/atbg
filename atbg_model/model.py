"""
ATBGModel: wires embeddings -> Time2Vec -> AVQ-BS -> HGT -> GRU -> TD-TO
+ SurvivalHead + RiskHead into one forward pass, per the pipeline in the
spec (Section 8.2 / Appendix E.7):

  turns -> BGE-M3 embeddings -> + Time2Vec(elapsed) -> AVQ-BS state codes
        -> heterogeneous graph -> HGT -> GRU memory
        -> TD-TO (next state) + SurvivalHead (ETA) + RiskHead (attack prob)

This file assumes turn embeddings and elapsed-time tensors are already
computed (by ingestion/pipeline.py) — it does NOT call BGE-M3 itself, so
the same model class works whether input came from live text or from a
pre-embedded training batch.
"""

import torch
import torch.nn as nn

from atbg_model.time2vec import Time2Vec
from atbg_model.avq_bs import AnchoredVectorQuantizer, NUM_ANCHORED, BEHAVIOR_STATES
from atbg_model.hgt import HGT, build_conversation_graph
from atbg_model.gru_memory import GRUMemory
from atbg_model.td_to import TimeDecayedTransitionOperator
from atbg_model.heads import SurvivalHead, RiskHead

EMBED_DIM = 1024      # BGE-M3 output size
TIME_DIM = 16          # Time2Vec output size
HIDDEN_DIM = 256       # HGT / GRU hidden size, per spec


class ATBGModel(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, time_dim: int = TIME_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.time2vec = Time2Vec(out_dim=time_dim)
        self.input_proj = nn.Linear(embed_dim + time_dim, hidden_dim)
        self.avq_bs = AnchoredVectorQuantizer(embed_dim=hidden_dim)
        self.hgt = HGT(hidden_dim=hidden_dim, num_heads=4, num_layers=2)
        self.gru = GRUMemory(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.td_to = TimeDecayedTransitionOperator(hidden_dim=hidden_dim, num_states=NUM_ANCHORED)
        self.survival_head = SurvivalHead(hidden_dim=hidden_dim)
        self.risk_head = RiskHead(hidden_dim=hidden_dim)

    def forward(
        self,
        turn_embeddings: torch.Tensor,       # (T, embed_dim) — one conversation
        elapsed_seconds: torch.Tensor,        # (T, 1) — gap before each turn (0 for the first)
        artifact_turn_pairs: list[tuple[int, int]] | None = None,
        speakers: list[str] | None = None,
        anchor_labels: torch.Tensor | None = None,   # (T,) weak/gold behaviour labels, -100 if none
        prev_gru_hidden: torch.Tensor | None = None,  # for streaming-append mode
    ):
        T = turn_embeddings.size(0)

        time_feat = self.time2vec(elapsed_seconds)                          # (T, time_dim)
        x = torch.cat([turn_embeddings, time_feat], dim=-1)                  # (T, embed+time)
        x = self.input_proj(x)                                                # (T, hidden_dim)

        vq_out = self.avq_bs(x.unsqueeze(0), anchor_labels.unsqueeze(0) if anchor_labels is not None else None)
        x_q = vq_out["quantized"].squeeze(0)                                   # (T, hidden_dim)
        code_indices = vq_out["code_indices"].squeeze(0)                       # (T,)

        adj, edge_type = build_conversation_graph(T, artifact_turn_pairs, speakers)
        x_graph = self.hgt(x_q, adj, edge_type)                                 # (T, hidden_dim)

        gru_out, new_hidden = self.gru(x_graph.unsqueeze(0), prev_gru_hidden)   # (1, T, hidden_dim)
        memory = gru_out.squeeze(0)                                              # (T, hidden_dim)

        # Anchored-state index for TD-TO's "current state" input: clamp VQ
        # code indices >= NUM_ANCHORED (the 20 unanchored codes) down to
        # their nearest anchored meaning via the anchor classifier's argmax,
        # so TD-TO always operates on the 12-state taxonomy.
        anchor_logits = vq_out["anchor_logits"].squeeze(0)                        # (T, NUM_ANCHORED)
        current_state_idx = anchor_logits.argmax(dim=-1)                           # (T,)

        # Elapsed time used by TD-TO is "time from this turn to the yet-unseen
        # next one" — at train time this is the ground-truth gap; at inference
        # time on the LAST turn it's unknown, so callers pass an expected/typical
        # value or query the survival head first and feed its estimate back in.
        next_state_probs = self.td_to(memory, current_state_idx, elapsed_seconds.squeeze(-1))

        eta_rate = self.survival_head(memory)                                       # (T,)
        risk_prob = self.risk_head(memory)                                           # (T,)

        return {
            "memory": memory,
            "gru_hidden": new_hidden,
            "code_indices": code_indices,
            "anchor_logits": anchor_logits,
            "current_state_idx": current_state_idx,
            "next_state_probs": next_state_probs,
            "eta_rate": eta_rate,
            "risk_prob": risk_prob,
            "vq_loss": vq_out["loss"],
        }

    def predict_labels(self, output: dict) -> dict:
        """Convenience: turns raw tensor outputs into human-readable labels
        for the LAST turn in the conversation (i.e. the current prediction)."""
        last = -1
        state_idx = output["current_state_idx"][last].item()
        next_idx = output["next_state_probs"][last].argmax().item()
        next_prob = output["next_state_probs"][last, next_idx].item()
        eta = self.survival_head.expected_time(output["eta_rate"][last]).item()
        risk = output["risk_prob"][last].item()

        return {
            "current_state": BEHAVIOR_STATES[state_idx],
            "next_state_pred": BEHAVIOR_STATES[next_idx],
            "next_state_prob": next_prob,
            "eta_seconds": eta,
            "risk_score": risk,
        }
