"""
Stage B — graph + transition (~6h on a T4 per spec; the longest stage).

Loads Stage A's checkpoint, unfreezes HGT + GRU + TD-TO, and trains the
model to predict the next behaviour state given the conversation so far.
This is where the TD-TO's per-state decay parameters actually learn
something — cross-entropy loss on next_state_probs against the TRUE next
turn's state, at every prefix of every conversation.

Run: python -m training.train_stage_b --data dataset/splits/train.jsonl \
        --init checkpoints/stage_a.pt --epochs 10
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from atbg_model.model import ATBGModel
from training.dataset import ConversationDataset, collate_single


def train_stage_b(data_path: str, init_checkpoint: str, epochs: int = 10, lr: float = 5e-5, device: str = "cpu"):
    model = ATBGModel().to(device)
    if os.path.exists(init_checkpoint):
        checkpoint = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded Stage A weights from {init_checkpoint}")
    else:
        print(f"WARNING: {init_checkpoint} not found, starting Stage B from scratch")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = ConversationDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Stage B epoch {epoch+1}/{epochs}"):
            embeddings = batch["embeddings"].to(device)
            elapsed = batch["elapsed_seconds"].to(device)
            anchor_labels = batch["anchor_labels"].to(device)

            if embeddings.size(0) < 2:
                continue  # need at least 2 turns to have a "next state" target

            out = model(embeddings, elapsed, anchor_labels=anchor_labels)

            # Next-state target at turn i is the TRUE state at turn i+1.
            # Skip positions where the next turn's gold/weak label is missing (-100).
            next_state_targets = anchor_labels[1:]                       # (T-1,)
            next_state_probs = out["next_state_probs"][:-1]              # (T-1, num_states)
            valid_mask = next_state_targets != -100

            if valid_mask.sum() == 0:
                continue

            next_state_loss = F.nll_loss(
                torch.log(next_state_probs[valid_mask].clamp(min=1e-8)),
                next_state_targets[valid_mask],
            )
            loss = next_state_loss + 0.1 * out["vq_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {total_loss / max(len(loader), 1):.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/splits/train.jsonl")
    parser.add_argument("--init", type=str, default="checkpoints/stage_a.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="checkpoints/stage_b.pt")
    args = parser.parse_args()

    model = train_stage_b(args.data, args.init, args.epochs, args.lr, args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "version": "stage_b"}, args.out)
    print(f"Saved {args.out}")
