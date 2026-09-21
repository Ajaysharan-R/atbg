"""
Ensemble score fusion evaluation.

Runs atbg_latest.pt (original) and real_atbg_final.pt (retrained) on the
same val/test set, fuses their risk scores at multiple weight ratios, and
prints a comparison table so you can pick the best combination.

Usage (on Kaggle):
    cd /kaggle/working/atbg
    python -m evaluation.ensemble_eval \
        --model_a /kaggle/working/atbg_latest.pt \
        --model_b /kaggle/working/real_atbg_final.pt \
        --data    /kaggle/input/datasets/ajaysharanramesh/atbg-real-eval/val.jsonl \
        --device  cuda

Then evaluate the winning weight on the held-out test set:
    python -m evaluation.ensemble_eval \
        --model_a /kaggle/working/atbg_latest.pt \
        --model_b /kaggle/working/real_atbg_final.pt \
        --data    /kaggle/input/datasets/ajaysharanramesh/atbg-real-eval/test.jsonl \
        --device  cuda \
        --weight_a 0.75    # <- use the winning weight from the val run
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from atbg_model.model import ATBGModel
from evaluation.metrics import detection_metrics, calibration_metrics
from training.dataset import ConversationDataset, collate_single

# Weight combinations to test: (weight_for_model_A, weight_for_model_B)
WEIGHT_COMBOS = [
    (1.00, 0.00),  # A only
    (0.75, 0.25),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.25, 0.75),
    (0.00, 1.00),  # B only
]


def collect_scores(model: ATBGModel, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_prob) for all conversations in the loader."""
    y_true, y_prob = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring", leave=False):
            embeddings = batch["embeddings"].to(device)
            elapsed = batch["elapsed_seconds"].to(device)
            out = model(embeddings, elapsed)
            risk = out["risk_prob"][-1].item()
            is_attack = batch["is_attack"].item()
            y_prob.append(risk)
            y_true.append(is_attack)
    return np.array(y_true), np.array(y_prob)


def load_model(path: str, device: str) -> ATBGModel:
    model = ATBGModel().to(device)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {path}  (version: {ckpt.get('version','unknown')})")
    return model


def print_row(label, metrics):
    print(f"  {label:<30} "
          f"AUPRC={metrics['auprc']:.3f}  "
          f"ROC={metrics['roc_auc']:.3f}  "
          f"P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  "
          f"FPR@95={metrics['fpr_at_95_tpr']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a",  required=True, help="Path to atbg_latest.pt (original model)")
    parser.add_argument("--model_b",  required=True, help="Path to real_atbg_final.pt (retrained model)")
    parser.add_argument("--data",     required=True, help="Path to val.jsonl or test.jsonl")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight_a", type=float, default=None,
                        help="If set, evaluate ONLY this single weight combination (for final test-set eval)")
    args = parser.parse_args()

    print(f"\nDevice: {args.device}")
    print(f"Data:   {args.data}\n")

    model_a = load_model(args.model_a, args.device)
    model_b = load_model(args.model_b, args.device)

    dataset = ConversationDataset(args.data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single)
    print(f"\nScoring {len(dataset)} conversations with Model A...")
    y_true, scores_a = collect_scores(model_a, loader, args.device)

    print(f"Scoring {len(dataset)} conversations with Model B...")
    # reset loader (DataLoader is stateful — create a new one)
    loader2 = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single)
    _, scores_b = collect_scores(model_b, loader2, args.device)

    print(f"\n{'='*85}")
    print(f"  {'Weight combo':<30} {'AUPRC':<8} {'ROC-AUC':<8} {'Prec':<7} {'Recall':<8} {'F1':<7} {'FPR@95'}")
    print(f"{'='*85}")

    combos = [(args.weight_a, 1 - args.weight_a)] if args.weight_a is not None else WEIGHT_COMBOS
    best_auprc, best_label = 0.0, ""
    results = {}

    for wa, wb in combos:
        fused = wa * scores_a + wb * scores_b
        label = f"A={wa:.0%} + B={wb:.0%}"
        m = {**detection_metrics(y_true, fused), **calibration_metrics(y_true, fused)}
        results[label] = m
        print_row(label, m)
        if m["auprc"] > best_auprc:
            best_auprc = m["auprc"]
            best_label = label

    print(f"{'='*85}")

    if args.weight_a is None:
        print(f"\n  Best by AUPRC: {best_label}  (AUPRC={best_auprc:.3f})")
        print("\nCalibration metrics for each combo:")
        for label, m in results.items():
            print(f"  {label:<30}  ECE={m['ece']:.3f}  Brier={m['brier_score']:.3f}")

        print("\nNext step: re-run with --weight_a <best_weight> --data test.jsonl for final held-out eval")
        wa_best = combos[[r[0] for r in combos].index(
            max(combos, key=lambda x: results[f"A={x[0]:.0%} + B={x[1]:.0%}"]["auprc"])[0]
        )][0]
        print(f"  (Best weight_a appears to be {wa_best:.2f})")
    else:
        print(f"\nFinal test-set result for weight_a={args.weight_a}")


if __name__ == "__main__":
    main()
