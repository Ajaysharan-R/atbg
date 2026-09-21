"""
The spec's core metric set (Section 15.2 / Appendix E.9). Implements
everything with a standard library or sklearn — no custom C code needed.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
)


def detection_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "auprc": average_precision_score(y_true, y_prob),   # PRIMARY metric per spec
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr_at_95_tpr": _fpr_at_tpr(y_true, y_prob, target_tpr=0.95),
    }


def _fpr_at_tpr(y_true: np.ndarray, y_prob: np.ndarray, target_tpr: float = 0.95) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def earliness_metrics(alert_turn_indices: list[int | None], conversation_lengths: list[int]) -> dict:
    """
    alert_turn_indices[i] = the turn index at which the model FIRST crossed
    the alert threshold for conversation i (None if it never did).
    Earliness Ratio = fraction of the conversation left unseen when the
    alert fired — 1.0 means "fired instantly", 0.0 means "fired on the last turn".
    """
    ratios = []
    for alert_idx, length in zip(alert_turn_indices, conversation_lengths):
        if alert_idx is None or length <= 1:
            continue
        ratios.append(1.0 - (alert_idx / length))

    return {
        "earliness_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "detected_fraction": sum(1 for a in alert_turn_indices if a is not None) / max(len(alert_turn_indices), 1),
    }


def next_state_metrics(y_true_states: np.ndarray, pred_probs: np.ndarray) -> dict:
    """y_true_states: (N,) true next-state index. pred_probs: (N, num_states)."""
    top1 = (pred_probs.argmax(axis=1) == y_true_states).mean()

    top3_preds = np.argsort(-pred_probs, axis=1)[:, :3]
    top3 = np.mean([y_true_states[i] in top3_preds[i] for i in range(len(y_true_states))])

    ranks = []
    for i, true_state in enumerate(y_true_states):
        order = np.argsort(-pred_probs[i])
        rank = int(np.where(order == true_state)[0][0]) + 1
        ranks.append(1.0 / rank)
    mrr = float(np.mean(ranks))

    return {"top1_accuracy": float(top1), "top3_accuracy": float(top3), "mrr": mrr}


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict:
    """ECE (target <= 0.05 per spec) + Brier score."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(bin_confidence - bin_accuracy)

    return {
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def time_prediction_metrics(y_true_seconds: np.ndarray, y_pred_seconds: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true_seconds - y_pred_seconds)))
    return {"mae_seconds": mae}
