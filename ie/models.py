from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .types import FactType


class ExtractionFact(BaseModel):
    type: FactType
    subject_id: str
    object_label: str | None = None
    object_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    timestamp: str | None = None
    notes: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class ExtractionResult(BaseModel):
    facts: list[ExtractionFact] = Field(default_factory=list)

    @field_validator("facts")
    @classmethod
    def _unique_facts(cls, value: list[ExtractionFact]) -> list[ExtractionFact]:
        seen = set()
        unique = []
        import json

        for fact in value:
            attr_key = json.dumps(fact.attributes, sort_keys=True)
            key = (fact.type, fact.subject_id, fact.object_label, fact.object_id, attr_key)
            if key in seen:
                continue
            seen.add(key)
            unique.append(fact)
        return unique
