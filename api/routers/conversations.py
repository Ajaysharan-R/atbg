import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db import get_db
from api.models_db import Conversation, Turn, Artifact
from api.schemas import ConversationIn
from api.services.redaction import redact_conversation
from api.services.artifact_extraction import extract_artifacts

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("")
def create_conversation(payload: ConversationIn, db: Session = Depends(get_db)):
    raw_texts = [t.text for t in payload.turns]
    redacted_texts = redact_conversation(raw_texts)

    conversation = Conversation(
        external_id=payload.external_id,
        source=payload.source,
        channel=payload.channel,
        language=payload.language,
    )
    db.add(conversation)
    db.flush()  # get conversation.id before creating turns

    for turn_in, redacted_text in zip(payload.turns, redacted_texts):
        turn = Turn(
            conversation_id=conversation.id,
            turn_index=turn_in.turn_index,
            speaker=turn_in.speaker,
            text_redacted=redacted_text,
            timestamp=turn_in.timestamp,
        )
        db.add(turn)
        db.flush()

        for artifact in extract_artifacts(turn_in.text):  # extracted from ORIGINAL text
            db.add(Artifact(turn_id=turn.id, artifact_type=artifact["artifact_type"], value=artifact["value"]))

    db.commit()
    db.refresh(conversation)
    return {"conversation_id": str(conversation.id), "num_turns": len(payload.turns)}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conversation = db.get(Conversation, uuid.UUID(conversation_id))
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    return {
        "id": str(conversation.id),
        "source": conversation.source,
        "channel": conversation.channel,
        "language": conversation.language,
        "is_attack": conversation.is_attack,
        "scam_type": conversation.scam_type,
        "turns": [
            {
                "turn_index": t.turn_index,
                "speaker": t.speaker,
                "text_redacted": t.text_redacted,
                "behavior_state": t.behavior_state,
                "artifacts": [{"type": a.artifact_type, "value": a.value} for a in t.artifacts],
            }
            for t in sorted(conversation.turns, key=lambda t: t.turn_index)
        ],
    }
