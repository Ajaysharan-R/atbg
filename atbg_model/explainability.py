"""
Three of the spec's four explanation mechanisms:
  1. Native trajectory explanation (no model internals needed — just the
     state sequence already produced).
  2. Integrated Gradients (via captum) — importance of turns/features.
  3. Counterfactual search — minimum change to flip the risk prediction
     below the alert threshold, via bounded greedy turn-removal.

(PGExplainer, the 4th method, explains *graph edges* specifically and is
the most involved to wire up well — left as a documented follow-up in
README.md rather than a half-working stub here.)
"""

import copy

import torch

from atbg_model.avq_bs import BEHAVIOR_STATES


def trajectory_explanation(code_history: list[int]) -> str:
    """Turns a sequence of anchored-state indices into a human sentence,
    e.g. 'Risk increased after the sender progressed from authority to urgency.'"""
    if not code_history:
        return "No turns to explain yet."
    states = [BEHAVIOR_STATES[i] for i in code_history]
    path = " → ".join(states)
    escalation_states = {"fear", "urgency", "credential_request", "payment_request", "verification_bypass"}
    escalation_points = [
        states[i] for i in range(1, len(states)) if states[i] in escalation_states and states[i - 1] not in escalation_states
    ]
    summary = f"Behaviour trajectory: {path}."
    if escalation_points:
        summary += f" Risk increased when the conversation moved into '{escalation_points[0]}'."
    return summary


def integrated_gradients_turn_importance(model, turn_embeddings: torch.Tensor, elapsed_seconds: torch.Tensor,
                                          target: str = "risk", steps: int = 32) -> torch.Tensor:
    """Returns a (T,) tensor of per-turn importance scores for either the
    risk score or, with target='next_state', the top predicted next state."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError("captum is required for Integrated Gradients — pip install captum") from e

    baseline = torch.zeros_like(turn_embeddings)

    def forward_fn(embeddings):
        out = model(embeddings, elapsed_seconds)
        if target == "risk":
            return out["risk_prob"][-1:].unsqueeze(0)
        else:
            top_idx = out["next_state_probs"][-1].argmax()
            return out["next_state_probs"][-1:, top_idx].unsqueeze(0)

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(turn_embeddings.unsqueeze(0), baseline.unsqueeze(0), n_steps=steps)
    # sum over the embedding dimension to get one score per turn
    return attributions.squeeze(0).abs().sum(dim=-1)


def counterfactual_search(model, turn_embeddings: torch.Tensor, elapsed_seconds: torch.Tensor,
                           alert_threshold: float = 0.5, max_removed: int = 3) -> dict:
    """Greedy search: repeatedly remove the single remaining turn whose
    removal drops risk the most, until risk crosses below alert_threshold
    or max_removed turns have been tried (bounded model-call budget, per spec)."""
    with torch.no_grad():
        current_embeddings = turn_embeddings.clone()
        current_elapsed = elapsed_seconds.clone()
        removed_indices: list[int] = []

        base_out = model(current_embeddings, current_elapsed)
        base_risk = base_out["risk_prob"][-1].item()

        if base_risk < alert_threshold:
            return {"validity": True, "sparsity": 0, "removed_turn_indices": [], "original_risk": base_risk, "final_risk": base_risk}

        for _ in range(max_removed):
            if current_embeddings.size(0) <= 1:
                break
            best_drop, best_idx = -1.0, None
            for i in range(current_embeddings.size(0)):
                trial_emb = torch.cat([current_embeddings[:i], current_embeddings[i + 1 :]])
                trial_elapsed = torch.cat([current_elapsed[:i], current_elapsed[i + 1 :]])
                out = model(trial_emb, trial_elapsed)
                risk = out["risk_prob"][-1].item()
                drop = base_risk - risk
                if drop > best_drop:
                    best_drop, best_idx, best_risk = drop, i, risk

            removed_indices.append(best_idx)
            current_embeddings = torch.cat([current_embeddings[:best_idx], current_embeddings[best_idx + 1 :]])
            current_elapsed = torch.cat([current_elapsed[:best_idx], current_elapsed[best_idx + 1 :]])
            base_risk = best_risk

            if base_risk < alert_threshold:
                return {
                    "validity": True,
                    "sparsity": len(removed_indices),
                    "removed_turn_indices": removed_indices,
                    "original_risk": None,
                    "final_risk": base_risk,
                }

        return {
            "validity": False,
            "sparsity": len(removed_indices),
            "removed_turn_indices": removed_indices,
            "original_risk": None,
            "final_risk": base_risk,
        }
