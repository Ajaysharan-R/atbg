import torch
from fastapi import APIRouter

from api.schemas import ConversationIn, AnalyzeResponse
from api.services.model_loader import get_model
from ingestion.pipeline import process_conversation
from atbg_model.explainability import trajectory_explanation

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
def analyze_conversation(payload: ConversationIn):
    turns = [
        {"turn_index": t.turn_index, "speaker": t.speaker, "text": t.text, "timestamp": t.timestamp}
        for t in payload.turns
    ]

    processed = process_conversation(turns)
    model_a, model_b, weight_a, model_version, ensemble_mode = get_model()

    with torch.no_grad():
        out_a = model_a(processed["embeddings"], processed["elapsed_seconds"])
        labels = model_a.predict_labels(out_a)

        if ensemble_mode and model_b is not None:
            out_b = model_b(processed["embeddings"], processed["elapsed_seconds"])
            risk_a = out_a["risk_prob"][-1].item()
            risk_b = out_b["risk_prob"][-1].item()
            fused_risk = weight_a * risk_a + (1 - weight_a) * risk_b
            labels["risk_score"] = fused_risk

    explanation_text = trajectory_explanation(out_a["current_state_idx"].tolist())
    flat_artifacts = [a for turn_artifacts in processed["artifacts"] for a in turn_artifacts]

    return AnalyzeResponse(
        conversation_id=payload.external_id or "unsaved",
        up_to_turn_index=len(turns),
        risk_score=labels["risk_score"],
        current_state=labels["current_state"],
        next_state_pred=labels["next_state_pred"],
        next_state_prob=labels["next_state_prob"],
        eta_seconds=labels["eta_seconds"],
        explanation={"trajectory": explanation_text},
        artifacts=flat_artifacts,
        model_version=model_version,
    )
