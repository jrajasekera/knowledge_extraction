"""Pydantic models shared across the memory agent service."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)

ConfidenceLevel = Literal["high", "medium", "low"]


class MessageModel(BaseModel):
    """Represents an inbound Discord message."""

    author_id: str = Field(alias="author_id")
    author_name: str = Field(alias="author_name")
    content: str
    timestamp: datetime
    message_id: str | None = Field(default=None, alias="message_id")
    reference_message_id: str | None = Field(default=None, alias="reference_message_id")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        msg = f"Unsupported timestamp value: {value!r}"
        raise TypeError(msg)

    @computed_field(return_type=str)
    def timestamp_text(self) -> str:  # noqa: D401
        """Timestamp rendered as ISO-8601 string."""
        return self.timestamp.isoformat()


class RetrievalRequest(BaseModel):
    """Schema for POST /api/memory/retrieve requests."""

    messages: list[MessageModel]
    channel_id: str
    max_facts: int | None = None
    max_messages: int | None = None
    max_iterations: int | None = None
    request_id: str | None = None

    @field_validator("messages")
    @classmethod
    def _ensure_messages_non_empty(cls, value: Sequence[MessageModel]) -> Sequence[MessageModel]:
        if not value:
            msg = "messages must contain at least one entry"
            raise ValueError(msg)
        return value


class RetrievalMetadata(BaseModel):
    """Metadata returned with a retrieval response."""

    queries_executed: int
    facts_retrieved: int
    processing_time_ms: int
    iterations_used: int
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    # Novelty metrics (early-stop policy)
    new_facts_last_iteration: int = 0
    novelty_streak_without_gain: int = 0
    unique_facts_seen: int = 0


class RetrievalResponse(BaseModel):
    """Successful retrieval response."""

    facts: list[str]
    messages: list[str]
    context_summary: str
    confidence: ConfidenceLevel
    metadata: RetrievalMetadata


class RetrievalError(BaseModel):
    """Error response schema."""

    error: str
    message: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["healthy", "degraded", "unhealthy"]
    neo4j_connected: bool
    model_loaded: bool
    version: str
    additional: dict[str, Any] = Field(default_factory=dict)


class FactEvidence(BaseModel):
    """Structured representation of a fact evidence payload."""

    source_id: str
    author: str | None = None
    url: HttpUrl | None = None
    snippet: str | None = None
    created_at: datetime | None = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        msg = f"Unsupported datetime value: {value!r}"
        raise TypeError(msg)


class RetrievedFact(BaseModel):
    """Normalized fact representation used internally by the agent."""

    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[FactEvidence] = Field(default_factory=list)
    timestamp: datetime | None = None
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        msg = f"Unsupported timestamp value: {value!r}"
        raise TypeError(msg)

    @model_validator(mode="after")
    def _ensure_person_name(self) -> RetrievedFact:
        if not self.person_name:
            msg = "person_name is required"
            raise ValueError(msg)
        return self
