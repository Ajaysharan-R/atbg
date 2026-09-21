import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    String,
    Text,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from api.db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String, unique=True, nullable=True)
    source = Column(String, nullable=False)
    channel = Column(String, nullable=False, default="unknown")
    language = Column(String, nullable=False, default="unknown")
    is_attack = Column(Boolean, nullable=True)
    scam_type = Column(String, nullable=True)
    is_synthetic = Column(Boolean, nullable=False, default=False)
    split = Column(String, nullable=True)
    label_source = Column(String, nullable=False, default="none")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    turns = relationship("Turn", back_populates="conversation", cascade="all, delete-orphan")


class Turn(Base):
    __tablename__ = "turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    turn_index = Column(Integer, nullable=False)
    speaker = Column(String, nullable=False)
    text_redacted = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    behavior_state = Column(String, nullable=True)
    behavior_confidence = Column(Float, nullable=True)
    label_source = Column(String, nullable=False, default="none")
    escalation_flag = Column(Boolean, nullable=False, default=False)

    conversation = relationship("Conversation", back_populates="turns")
    artifacts = relationship("Artifact", back_populates="turn", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    turn_id = Column(UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"))
    artifact_type = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    turn = relationship("Turn", back_populates="artifacts")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    up_to_turn_index = Column(Integer, nullable=False)
    risk_score = Column(Float, nullable=False)
    current_state = Column(String, nullable=True)
    next_state_pred = Column(String, nullable=True)
    next_state_prob = Column(Float, nullable=True)
    eta_seconds = Column(Float, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    model_version = Column(String, nullable=False, default="untrained")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
