"""
Stage C — fusion + calibration (~1h on a T4 per spec).

Trains RiskHead (focal loss against conversation-level is_attack) and
SurvivalHead (exponential NLL against observed inter-turn gaps), on top of
Stage B's already-trained encoder/graph/GRU/TD-TO (those stay frozen here
so this stage is fast and focused).

Run: python -m training.train_stage_c --data dataset/splits/train.jsonl \
        --init checkpoints/stage_b.pt --epochs 5
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from atbg_model.model import ATBGModel
from training.dataset import ConversationDataset, collate_single


def train_stage_c(data_path: str, init_checkpoint: str, epochs: int = 5, lr: float = 1e-4, device: str = "cpu"):
    model = ATBGModel().to(device)
    if os.path.exists(init_checkpoint):
        checkpoint = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded Stage B weights from {init_checkpoint}")
    else:
        print(f"WARNING: {init_checkpoint} not found, starting Stage C from scratch")

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("risk_head") or name.startswith("survival_head")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    dataset = ConversationDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Stage C epoch {epoch+1}/{epochs}"):
            embeddings = batch["embeddings"].to(device)
            elapsed = batch["elapsed_seconds"].to(device)
            anchor_labels = batch["anchor_labels"].to(device)
            is_attack = batch["is_attack"].to(device)

            out = model(embeddings, elapsed, anchor_labels=anchor_labels)

            # Risk: use the LAST turn's prediction, target = conversation-level label
            risk_pred = out["risk_prob"][-1:]
            risk_target = is_attack.expand_as(risk_pred)
            risk_loss = model.risk_head.focal_loss(risk_pred, risk_target)

            # Survival: observed gap = elapsed time of the NEXT turn (if any);
            # last turn's gap is censored (conversation ended before "next" happened)
            if embeddings.size(0) > 1:
                observed_time = elapsed[1:].squeeze(-1)
                is_censored = torch.zeros_like(observed_time, dtype=torch.bool)
                is_censored[-1] = True  # conversation's final observed gap is right-censored
                rate = out["eta_rate"][:-1]
                survival_loss = model.survival_head.loss(rate, observed_time, is_censored)
            else:
                survival_loss = torch.tensor(0.0)

            loss = risk_loss + survival_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {total_loss / max(len(loader), 1):.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/splits/train.jsonl")
    parser.add_argument("--init", type=str, default="checkpoints/stage_b.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="checkpoints/atbg_latest.pt")
    args = parser.parse_args()

    model = train_stage_c(args.data, args.init, args.epochs, args.lr, args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "version": "stage_c"}, args.out)
    print(f"Saved {args.out}  <-- this is the checkpoint the API loads by default")
