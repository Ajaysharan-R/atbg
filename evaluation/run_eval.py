"""
Runs the trained ATBG model plus the two implemented baselines against a
test split and prints the metric report.

Run: python -m evaluation.run_eval --checkpoint checkpoints/atbg_latest.pt \
        --data dataset/splits/test.jsonl
"""

import argparse
import json

import numpy as np
import torch

from atbg_model.model import ATBGModel
from evaluation.baselines import KeywordRegexBaseline, TfidfLogisticBaseline
from evaluation.metrics import detection_metrics, calibration_metrics
from training.dataset import ConversationDataset, collate_single
from torch.utils.data import DataLoader


def evaluate(checkpoint_path: str, data_path: str, device: str = "cpu"):
    model = ATBGModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ConversationDataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    y_true, y_prob_atbg, y_prob_keyword = [], [], []
    conversation_texts = []
    keyword_baseline = KeywordRegexBaseline()

    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            elapsed = batch["elapsed_seconds"].to(device)
            is_attack = batch["is_attack"].item()

            out = model(embeddings, elapsed)
            risk = out["risk_prob"][-1].item()

            y_true.append(is_attack)
            y_prob_atbg.append(risk)

    y_true = np.array(y_true)
    y_prob_atbg = np.array(y_prob_atbg)

    print("=== ATBG ===")
    print(detection_metrics(y_true, y_prob_atbg))
    print(calibration_metrics(y_true, y_prob_atbg))

    # TF-IDF+LR baseline needs to be fit on train, not test — this is left
    # as a documented next step (needs the train split loaded here too)
    # rather than silently fit-and-evaluated on the same data.
    print("\nNOTE: B2 (TF-IDF+LR) and other baselines need fitting on the")
    print("TRAIN split separately — see evaluation/baselines.py TfidfLogisticBaseline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/atbg_latest.pt")
    parser.add_argument("--data", type=str, default="dataset/splits/test.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.device)
