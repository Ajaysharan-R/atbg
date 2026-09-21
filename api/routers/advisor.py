"""POST /advisor — ATBG Security Advisor endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from api.services.advisor_service import build_response, try_claude_enhancement

router = APIRouter(prefix="/advisor", tags=["advisor"])


class AdvisorRequest(BaseModel):
    conversation: Optional[str] = ""
    risk_score: float
    current_state: str
    next_state: str
    next_state_probability: Optional[float] = None
    trajectory: list[str] = []
    question: str
    indicators: list[str] = []


class AdvisorResponse(BaseModel):
    answer: str
    recommended_actions: list[str]
    risk_level: str
    risk_score: int


@router.post("", response_model=AdvisorResponse)
def advisor(payload: AdvisorRequest):
    # Try Claude enhancement first if API key available
    enhanced = try_claude_enhancement(
        payload.question,
        {
            "risk_score": payload.risk_score,
            "current_state": payload.current_state,
            "next_state": payload.next_state,
            "next_state_probability": payload.next_state_probability,
            "trajectory": payload.trajectory,
            "indicators": payload.indicators,
        }
    )

    result = build_response(
        question=payload.question,
        risk_score=payload.risk_score,
        current_state=payload.current_state,
        next_state=payload.next_state,
        next_state_probability=payload.next_state_probability or 0.0,
        trajectory=payload.trajectory,
        conversation=payload.conversation or "",
        indicators=payload.indicators,
    )

    if enhanced:
        result["answer"] = enhanced

    return AdvisorResponse(**result)
