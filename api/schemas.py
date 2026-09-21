from datetime import datetime
from typing import Optional

from pydantic import BaseModel

# ── Fixed taxonomy (Appendix A of the spec) ─────────────────────
BEHAVIOR_STATES = [
    "benign",
    "greeting",
    "rapport_trust",
    "authority",
    "fear",
    "urgency",
    "credential_request",
    "payment_request",
    "verification_bypass",
    "isolation",
    "reciprocity_bait",
    "exit_closure",
]

SCAM_TYPES = [
    "phishing",
    "bec",
    "romance",
    "investment",
    "tech_support",
    "job_task",
    "sextortion",
    "lottery_419",
    "other",
]

CHANNELS = ["email", "sms", "whatsapp", "telegram", "chat", "voice_transcript", "screenshot", "unknown"]


class TurnIn(BaseModel):
    turn_index: int
    speaker: str
    text: str
    timestamp: Optional[datetime] = None


class ConversationIn(BaseModel):
    external_id: Optional[str] = None
    source: str = "user_upload"
    channel: str = "unknown"
    language: str = "unknown"
    turns: list[TurnIn]


class ArtifactOut(BaseModel):
    artifact_type: str
    value: str


class AnalyzeResponse(BaseModel):
    conversation_id: str
    up_to_turn_index: int
    risk_score: float
    current_state: Optional[str]
    next_state_pred: Optional[str]
    next_state_prob: Optional[float]
    eta_seconds: Optional[float]
    explanation: dict
    artifacts: list[ArtifactOut]
    model_version: str
