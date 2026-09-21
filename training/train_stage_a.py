"""
Stage A — state pretraining (~2h on a T4 per spec).

Trains the embedding projection + AVQ-BS quantizer to produce
human-nameable behaviour states, using the anchoring loss against
weak/gold labels. HGT/GRU/TD-TO/heads are NOT trained yet in this stage —
Stage A only needs input_proj + avq_bs to converge before the heavier
graph/temporal machinery is layered on in Stage B.

Run: python -m training.train_stage_a --data dataset/splits/train.jsonl --epochs 5
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from atbg_model.model import ATBGModel
from training.dataset import ConversationDataset, collate_single


def train_stage_a(data_path: str, epochs: int = 5, lr: float = 1e-4, device: str = "cpu"):
    model = ATBGModel().to(device)
    # Freeze everything except input_proj + avq_bs for this stage.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("input_proj") or name.startswith("avq_bs") or name.startswith("time2vec")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    dataset = ConversationDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Stage A epoch {epoch+1}/{epochs}"):
            embeddings = batch["embeddings"].to(device)
            elapsed = batch["elapsed_seconds"].to(device)
            anchor_labels = batch["anchor_labels"].to(device)

            time_feat = model.time2vec(elapsed)
            x = model.input_proj(torch.cat([embeddings, time_feat], dim=-1))
            vq_out = model.avq_bs(x.unsqueeze(0), anchor_labels.unsqueeze(0))
            loss = vq_out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {total_loss / max(len(loader), 1):.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/splits/train.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="checkpoints/stage_a.pt")
    args = parser.parse_args()

    model = train_stage_a(args.data, args.epochs, args.lr, args.device)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "version": "stage_a"}, args.out)
    print(f"Saved {args.out}")
