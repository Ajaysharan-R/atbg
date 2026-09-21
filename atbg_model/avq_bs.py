"""
AVQ-BS: Anchored Vector-Quantized Behaviour States.

Converts a continuous turn embedding into a discrete, interpretable
behaviour-state code. Codebook has 32 codes total; the first 12 are
"anchored" — nudged during training to align with the human behaviour
taxonomy (via the auxiliary classification loss below) so they stay
human-nameable, while the remaining 20 are free to specialise on
patterns the taxonomy doesn't explicitly name.

This is a standard VQ-VAE-style quantizer (van den Oord et al.) with an
added anchoring loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BEHAVIOR_STATES = [
    "benign", "greeting", "rapport_trust", "authority", "fear", "urgency",
    "credential_request", "payment_request", "verification_bypass",
    "isolation", "reciprocity_bait", "exit_closure",
]
NUM_ANCHORED = len(BEHAVIOR_STATES)  # 12
NUM_CODES = 32


class AnchoredVectorQuantizer(nn.Module):
    def __init__(self, embed_dim: int, num_codes: int = NUM_CODES, num_anchored: int = NUM_ANCHORED,
                 commitment_cost: float = 0.25, anchor_cost: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.num_anchored = num_anchored
        self.commitment_cost = commitment_cost
        self.anchor_cost = anchor_cost

        self.codebook = nn.Embedding(num_codes, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        # Auxiliary head: predicts which of the 12 human states a turn is,
        # used only to pull the first 12 codes toward human-recognisable
        # meanings. Codes 12..31 are never targeted by this loss.
        self.anchor_classifier = nn.Linear(embed_dim, num_anchored)

    def forward(self, z: torch.Tensor, anchor_labels: torch.Tensor | None = None):
        """
        z: (B, T, embed_dim) continuous turn embeddings.
        anchor_labels: optional (B, T) long tensor, values in [0, num_anchored)
                       or -100 for "no gold label at this turn" (ignored in loss).
        Returns: quantized (B,T,embed_dim), code_indices (B,T), loss (scalar), logits (B,T,num_anchored)
        """
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)                                   # (B*T, D)

        codebook_w = self.codebook.weight                            # (num_codes, D)
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ codebook_w.t()
            + codebook_w.pow(2).sum(1)
        )                                                             # (B*T, num_codes)
        code_indices = distances.argmin(dim=1)                        # (B*T,)
        quantized = self.codebook(code_indices).view(B, T, D)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # VQ-VAE losses
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Anchoring loss: only on turns that have a human gold/weak label
        anchor_logits = self.anchor_classifier(z)                     # (B, T, num_anchored)
        anchor_loss = torch.tensor(0.0, device=z.device)
        if anchor_labels is not None:
            mask = anchor_labels != -100
            if mask.any():
                anchor_loss = F.cross_entropy(
                    anchor_logits.reshape(-1, self.num_anchored),
                    anchor_labels.reshape(-1),
                    ignore_index=-100,
                )

        total_loss = vq_loss + self.anchor_cost * anchor_loss

        return {
            "quantized": quantized_st,
            "code_indices": code_indices.view(B, T),
            "loss": total_loss,
            "vq_loss": vq_loss,
            "anchor_loss": anchor_loss,
            "anchor_logits": anchor_logits,
        }

    def codebook_usage_entropy(self, code_indices: torch.Tensor) -> float:
        """Interpretability metric (Section 15.3): are all 32 codes actually
        being used, or has the model collapsed onto a handful of them?"""
        counts = torch.bincount(code_indices.reshape(-1), minlength=self.num_codes).float()
        probs = counts / counts.sum().clamp(min=1)
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        return entropy
